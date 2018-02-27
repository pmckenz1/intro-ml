#!/usr/bin/env python

"""
Generate large database of site counts from coalescent simulations 
based on msprime + toytree for using in machine learning algorithms. 
"""

## make py3 compatible
from __future__ import print_function

## imports
import os
import h5py
import numba
import sklearn
import toyplot
import toytree
import itertools
import numpy as np
import msprime as ms
import datetime
import time
import ipyparallel as ipp
import scipy



class Model(object):
    """
    A coalescent model for returning ms simulations. 
    """
    def __init__(self, 
        tree, 
        admixture_edges=[], 
        Ne=int(1e5), 
        mut=1e-5, 
        nsnps=1000, 
        ntests=100, 
        seed=12345, 
        **kwargs):
        """
        An object for running simulations to attain genotype matrices for many
        independent runs to sample Nrep SNPs. 
  
        Parameters:
        -----------
        tree: (str)
            A newick string representation of a species tree with edges in 
            units of generations.
            
        admixture_edges (list):
            A list of admixture events in the format (source, dest, start, end, rate).

        Ne (int):
            Effective population size (single fixed value currently)

        mut (float):
            Mutation rate.   
        """
        ## init random seed
        np.random.seed(seed)

        ## hidden argument to turn on debugging
        self._debug = [True if kwargs.get("debug") else False][0]

        ## store sim params as attrs
        self.Ne = Ne
        self.mut = mut
        self.nsnps = nsnps
        self.ntests = ntests

        ## parse the input tree
        if isinstance(tree, toytree.tree):
            self.tree = tree
        elif isinstance(tree, str):
            self.tree = toytree.tree(tree)
        else:
            raise TypeError("input tree should be newick str or Toytree object")
        self.ntips = len(self.tree)

        self.namedict = {}
        leaf = 0
        for node in self.tree.tree.traverse():
            if node.is_leaf():
                ## store old name
                self.namedict[str(node.idx)] = node.name
                ## set new name
                node.name = str(node.idx) #leaf
                ## advance counter
                #leaf += 1   
        
        ## parse the input admixture edges. It should a list of tuples, or list
        ## of lists where each element has five values. 
        if admixture_edges:
            if isinstance(admixture_edges, tuple):
                admixture_edges = [admixture_edges]
            if not isinstance(admixture_edges[0], (list, tuple)):
                admixture_edges = [admixture_edges]
        for event in admixture_edges:
            if len(event) != 5:
                raise ValueError(
                    "admixture events should each be a tuple with 5 values")
        self.admixture_edges = admixture_edges

        ## generate migration parameters from the tree and admixture_edges
        ## stores data in memory as self.test_values
        self.get_test_values()

        ## store results (empty until you run .run())
        self.counts = None


    def get_test_values(self): 
        """
        Generates mrates and mtimes arrays for a range of values (ns) where
        migration rate is uniformly sampled, and its start and end points are
        uniformly sampled but contained within 0.05-0.95% of the branch length. 
        Rates are drawn uniformly between 0.0 and 0.95. 
        """
        ## init a dictionary for storing arrays for each admixture scenario
        self.test_values = {}

        ## iterate over events in admixture list
        idx = 0
        for event in self.admixture_edges:

            ## if times and rate were provided then use em.
            if all(event[-3:]):
                mrates = np.repeat(event[4], self.ntests)
                mtimes = np.stack([
                    np.repeat(event[2], self.ntests), 
                    np.repeat(event[3], self.ntests)], axis=1)
                self.test_values[idx] = {"mrates": mrates, "mtimes": mtimes}

            ## otherwise generate uniform values across edges
            else:        
                ## get migration rates from zero to ~full
                minmig = 0.0
                maxmig = 0.5
                mrates = np.random.uniform(minmig, maxmig, self.ntests)

                ## get divergence times from source start to end
                self._intervals = get_all_admix_edges(self.tree)                
                snode = self.tree.tree.search_nodes(idx=event[0])[0]
                dnode = self.tree.tree.search_nodes(idx=event[1])[0]
                interval = self._intervals[snode.idx, dnode.idx]
                edge_min = int(interval[0] * 2. * self.Ne)
                edge_max = int(interval[1] * 2. * self.Ne)
                mtimes = np.sort(
                    np.random.uniform(edge_min, edge_max, self.ntests*2)
                    .reshape((self.ntests, 2)), axis=1).astype(int)
                self.test_values[idx] = {"mrates": mrates, "mtimes": mtimes}
                if self._debug:
                    print("uniform testvals mig:", (edge_min, edge_max), (minmig, maxmig))
            idx += 1


    def plot_test_values(self):
        """
        Returns a toyplot canvas 
        """
        ## setup canvas
        canvas = toyplot.Canvas(height=300, width=500)
        ax0 = canvas.cartesian(
            grid=(1, 2, 0), xlabel="migration durations", ylabel="simulation index")
        ax1 = canvas.cartesian(
            grid=(1, 2, 1), xlabel="proportion migrants", ylabel="frequency")

        if self.test_values.keys():
            ## get values for the first admixture edge
            mtimes = self.test_values[0]["mtimes"]
            mrates = self.test_values[0]["mrates"]
            mt = mtimes[mtimes[:, 1].argsort()]
            boundaries = np.column_stack((mt[:, 0], mt[:, 1]))
            #durations = mtimes[:, 1] - mtimes[:, 0]

            ## plot
            ax0.fill(boundaries, along='y')
            ax1.bars(np.histogram(mrates))# * durations))
            return canvas
        else:
            raise ValueError(
                "No test_values generated. Model object must have admixture edges")


    ## functions to build simulation options 
    def _get_demography(self, idx):
        """
        returns demography scenario based on an input tree and admixture
        edge list with events in the format (source, dest, start, end, rate)
        """
        ## Define demographic events for msprime
        demog = set()

        ## tag min index child for each node, since at the time the node is 
        ## called it may already be renamed by its child index b/c of divergence
        ## events.
        for node in self.tree.tree.traverse():
            if node.children:
                node._schild = min([i.idx for i in node.get_descendants()])
            else:
                node._schild = node.idx

        ## Add divergence events
        for node in self.tree.tree.traverse():
            if node.children:
                dest = min([i._schild for i in node.children])
                source = max([i._schild for i in node.children])
                time = int(node.height * 2. * self.Ne)
                demog.add(ms.MassMigration(time, source, dest))
                if self._debug:
                    print('demog div:', (time, source, dest))
        
        ## Add migration edges
        for key in self.test_values:
            mdict = self.test_values[key]
            time = mdict['mtimes'][idx]
            rate = mdict['mrates'][idx]
            source, dest = self.admixture_edges[key][:2]

            ## rename nodes at time of admix in case divergences renamed them
            snode = self.tree.tree.search_nodes(idx=source)[0]
            dnode = self.tree.tree.search_nodes(idx=dest)[0]

            demog.add(ms.MigrationRateChange(time[0], rate, (snode._schild, dnode._schild)))
            demog.add(ms.MigrationRateChange(time[1], 0, (snode._schild, dnode._schild)))
            if self._debug:
                print('demog mig:', (time[0], time[1], rate, (snode._schild, dnode._schild)))

        ## sort events by time
        demog = sorted(list(demog), key=lambda x: x.time)
        if self._debug:
            print("")
        return demog


    def _get_popconfig(self):
        """
        returns population_configurations for N tips of a tree
        """
        population_configurations = [
            ms.PopulationConfiguration(sample_size=1, initial_size=self.Ne) \
            for ntip in range(self.ntips)]
        return population_configurations

        
    def _simulate(self, idx):
        """
        performs simulations with params varied across input values.
        """       
        ## set up simulation
        sim = ms.simulate(
            num_replicates=self.nsnps * 100,
            mutation_rate=self.mut,
            migration_matrix=np.zeros((self.ntips, self.ntips), dtype=int).tolist(),
            population_configurations=self._get_popconfig(),
            demographic_events=self._get_demography(idx)
        )
        return sim

   
    def mutate_jc(self, geno):
        """
        mutates sites with 1 into a new base in {0, 1, 2, 3}
        """
        for ridx in range(geno.shape[0]):

            ## get an array of starting bases, e.g., [0, 0, 0, 0]
            init = np.repeat(np.random.randint(0, 4), self.ntips)

            ## get the base it will mutate to e.g., 1
            notinit = list(set([0, 1, 2, 3]) - set(init))

            ## change mutated bases to notinit if there is a mut
            if np.sum(geno[ridx]):
                init[geno[ridx]==1] = np.random.choice(notinit)
                geno[ridx] = init
                ## return 1 SNP 
                return geno[0, :]
            else:
                return np.array([])
            
        ## if geno shape is 0
        return np.array([])       
    

    def run(self):
        """
        run and parse results for nsamples simulations.
        """
        ## storage for output
        #self.counts = np.zeros((self.ntests, self.nquarts, 16, 16), dtype=int)
        
        self.nquarts = int(scipy.special.comb(N=self.ntips,k=4))
        self.quarts = np.array(list(itertools.combinations(xrange(self.ntips), 4)))
        self.snps = np.zeros((self.ntests,self.nsnps,self.ntips))
        self.counts = np.zeros((self.ntests, self.nquarts, 16, 16))
        
        for ridx in range(self.ntests):
            ## run simulation for demography idx
            sims = self._simulate(ridx)
            ## this will give us a bunch of small trees to sample from, and we'll parse with next()
            ## and we can just pick snps one for each tree, with shape (1,ntaxa)
            ## we'll do this until we've filled an array of shape (nsnps,ntips)
            
            snparr = np.zeros((self.nsnps,self.ntips), dtype=np.uint16)
            
            ### continue until nsnps are simulated
            fidx = 0
            while fidx < self.nsnps:
                ## get just the first mutation
                bingenos = sims.next().genotype_matrix()
                ## convert to sequence under JC
                sitegenos = self.mutate_jc(bingenos)
                ## count it
                if sitegenos.size:
                    snparr[fidx] = sitegenos
                    fidx += 1
                    
            ## making a full snps array for now
            self.snps[ridx] = snparr
            
            ## keep track for counts index
            quartidx = 0
            
            ## iterator for quartets
            qiter = itertools.combinations(xrange(self.ntips), 4)
            for currquart in qiter:
                ## luckily, because we've renamed taxa with node.idx convention, col numbers correspond to tip label
                quartsnps = snparr[:,currquart]
                carr = np.zeros((self.nsnps, 16, 16),dtype = np.uint64)
                snpidx = 0
                for snp in quartsnps:
                    carr[snpidx] = count(snp)
                    snpidx += 1
                self.counts[ridx, quartidx] = carr.sum(axis=0)
                quartidx += 1
            ### array to store site counts
            #carr = np.zeros((self.nsnps, 16, 16))
            #        carr[fidx] = count(sitegenos)
            #        
            ### fill site counts into 16x16 matrix
            #self.counts[ridx] = carr.sum(axis=0)



## jitted functions for running super fast
@numba.jit(nopython=True)
def count(i):
    """
    return a 16x16 matrix of site counts from sitegenos
    """
    arr = np.zeros((16, 16), dtype=np.uint16)
    arr[(4*i[0])+i[1], (4*i[2])+i[3]] += 1
    return arr



class DataBase(object):
    """
    An object to parallelize simulations over many parameter settings
    and store finished reps in a HDF5 database.
    """
    def __init__(self,
        name,
        workdir,
        treelist,
        Ne,
        mut,
        nsnps,
        ntests,
        seed,
        force=False,
        **kwargs):

        ## identify this set of simulations
        self.name = name
        self.workdir = workdir
        self.path = os.path.join(workdir, self.name+".hdf5")
        self.trees = treelist
        
        def _make_vector(arg):
            "checks input argument length and type"
            if type(arg) is int or type(arg) is float:
                return([arg]*len(self.trees))
            if (type(arg) is list) and (len(arg) is len(self.trees)):
                return(arg)
            else: 
                raise ValueError(
                    "Ne, mut, nsnps, and ntests arguments each must be an \
                     int/float (which will be applied to all trees) or be a \
                     list the same length as the tree list")

        ## Accept arguments of length 1 or of the same length as the treelist
        self.Ne = _make_vector(Ne)
        self.mut = _make_vector(mut)
        self.nsnps = _make_vector(nsnps)
        self.ntests = _make_vector(ntests)
        self.seed = _make_vector(seed)
        self.nquarts = int(scipy.special.comb(N=len(toytree.tree(self.trees[0]).get_tip_labels()), k=4))
        ## make sure workdir exists
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        ## create database in 'w-' mode to prevent overwriting
        if os.path.exists(self.path):
            if force:
                ## exists and destroy it
                if raw_input("Do you really want to overwrite the database?"):
                    os.remove(self.path)
                    self.database = h5py.File(self.path, mode='w')                    
            else:
                ## exists append to it
                self.database = h5py.File(self.path, mode='a')
        else:
            ## does not exist
            self.database = h5py.File(self.path, mode='w-')     

        ## Create datasets for all planned simulations and write
        ## accompanying metadata for sim params in each data set
        self._generate_database()
        self.database.close()


    def _generate_database(self):
        """
        Parses parameters in self.params to create all combinations
        of parameter values to test. Returns the number of the simulations.
        Simulation metadata is appended to datasets. 
        """
        
        ## Does an arguments group already exist in your database file?
        try:
            self.database['args']
        except:
            total_treenum = 0
            args_array = np.empty(shape=[0,13])
            tree_array = np.empty(shape=[0,1])
            argsexists = False
        else:
            total_treenum = len(self.database['args'])
            args_array = self.database['args']
            tree_array = self.database['trees']
            argsexists = True

        # iterate over trees and ...
        for treenum in range(len(self.trees)):
            ## Get each tree
            currtree = toytree.tree(self.trees[treenum])
            
            ## Get each possible admixture event
            admixedges = get_all_admix_edges(currtree)
            intervals = admixedges.values()
            branches = admixedges.keys()
            onetreetests=np.empty(shape=[0,13])
            newtrees=np.empty(shape=[0,1])
            for event in range(len(branches)):
                ## initialize a model -- we'll use this to get parameters for each test on this admixture event
                carr = Model(Ne = self.Ne[treenum],
                             mut = self.mut[treenum],
                             nsnps = self.nsnps[treenum],
                             ntests = self.ntests[treenum],
                             seed = self.seed[treenum],
                             tree = currtree, 
                             admixture_edges=(branches[event][0], branches[event][1], None, None, None))
                ## save relevant parameters for each test
                eventtest = np.column_stack([[total_treenum]*carr.ntests, 
                                np.repeat(np.array([branches[event]]),carr.ntests,axis = 0),
                                np.repeat(np.array([intervals[event]]),carr.ntests,axis = 0),
                                carr.test_values[0]['mtimes'], 
                                carr.test_values[0]['mrates'],
                                [carr.Ne]*carr.ntests,
                                [carr.mut]*carr.ntests,
                                [carr.nsnps]*carr.ntests,
                                [carr.ntests]*carr.ntests,
                                [self.seed[treenum]]*carr.ntests])
                onetreetests=np.vstack([onetreetests,eventtest])
                newtrees = np.vstack([newtrees,np.tile(np.array(self.trees[treenum]),[carr.ntests,1])])

            total_treenum += 1
            ## Add to the overall args array
            args_array = np.vstack([args_array, np.array(onetreetests)])
            tree_array = np.vstack([tree_array, np.array(newtrees)])
            
            ## We should be holding the whole database in Python right now, so we want to add to a blank slate
            self.database.clear()
            self.database.create_dataset("args", data=args_array)
            self.database.create_dataset("trees", data=tree_array)
        ## store result
        self.ndatasets = len(args_array)



    def run(self, force=False):
        
        """
        Distribute simulations across a parallel Client. If continuing
        a previous run then any unfinished simulation will be queued up
        to run. 
        """
        
        def _add_mat(arr, numberdone):
            """
            Add one matrix to the HDF5 'counts' group. Collette book page 39.
            """
            counts_set[numberdone,:,:,:] = arr.astype(int)
            return(numberdone+1)

        def _add_quarts(arr, numberdone):
            """
            Add one matrix to the HDF5 'counts' group. Collette book page 39.
            """
            quarts_set[numberdone,:,:] = arr.astype(int)
            return(numberdone + 1)

        def _done(numberdone,nquarts):
            """
            Resize your HDF5 'counts' group at the end to the same length as filled count matrices. Collette book page 40.
            """
            counts_set.resize((numberdone,nquarts,16,16))
            quarts_set.resize((numberdone,nquarts,4))
            
        
        ## need to get ipyclient feature working
        #run(self, ipyclient, force=False):
        
        
        mydatabase = h5py.File(self.path, mode='r+')
        sizeargs = mydatabase['args'].len()
        
        ## Does a counts group already exist in your database file?
        try:
            mydatabase['counts']
        except:
            ## if 'counts' doesn't exist
            numberdone = 0 # will adjust this at the end of the loop
            countexists = False
            ## initialize the group
            counts_set = mydatabase.create_dataset('counts',(1,self.nquarts, 16,16),maxshape = (None, self.nquarts, 16, 16), chunks = (4,self.nquarts,16,16),dtype=int)
            quarts_set = mydatabase.create_dataset('quarts',(1,self.nquarts,4),maxshape = (None,self.nquarts, 4), chunks=(1,self.nquarts,4),dtype=int)
        else:
            numberdone = len(mydatabase['counts'])
            countexists = True
        

        
        trigger = 1 # will change this to 0 once we are done

        ## initialize client
        c = ipp.Client()
        lbview = c.load_balanced_view()
        
        while trigger:
            argsleft = sizeargs - numberdone # fill this at the beginning of each loop

            if argsleft > 100:
                windowsize = 100
            else:
                windowsize = argsleft

            ## create empty dataset to hold your set of int paras
            argsints = np.empty((windowsize,6),dtype=int)
            ## fill the dataset with the window of values you want (ints)
            mydatabase['args'].read_direct(argsints, np.s_[numberdone:(numberdone+windowsize),[0,1,2,8,10,12]])
            ## create empty dataset to hold your set of float paras
            argsflts = np.empty((windowsize,4),dtype=float)
            ## fill the dataset with the window of values you want (floats)
            mydatabase['args'].read_direct(argsflts, np.s_[numberdone:(numberdone+windowsize),[5,6,7,9]])

            # resize this for writing the current window
            counts_set.resize((len(counts_set)+windowsize,self.nquarts,16,16))
            quarts_set.resize((len(quarts_set)+windowsize,self.nquarts,4))
            
            #start parallel computing part
            
            def parallel_model(trees,argsints,argsflts,windowsize,nquarts):
                """
                This takes parameters for a big window of parameters and runs a model on each parameter sample.
                This is the function to run using ipyparallel
                Returns an array of shape = [windowsize,16,16]
                """
                print("inside model run function")
                import numpy as np
                store_counts_parallel = np.empty([windowsize,nquarts,16,16])
                store_quarts_parallel = np.empty([windowsize,nquarts,4])
                for idx in xrange(windowsize):
                    treenum, sourcebr, destbr, Ne, nsnps, seed = argsints[idx,:]
                    mtimerecent, mtimedistant, mrate, mut = argsflts[idx,:]
                    mod = Model(tree = trees[treenum],
                                admixture_edges = [(sourcebr,destbr,mtimerecent,mtimedistant,mrate)],
                                Ne = Ne,
                                nsnps = nsnps,
                                mut = mut,
                                seed = seed,
                                ntests = 1)
                    mod.run()
                    store_counts_parallel[idx,:,:,:]=mod.counts
                    store_quarts_parallel[idx,:,:]=mod.quarts 
                return store_counts_parallel, store_quarts_parallel
            #return([parallel_model,self.trees,argsints,argsflts,windowsize]) ## for debugging
            
            ## Set client to work
            task = lbview.apply(parallel_model,self.trees,argsints,argsflts,windowsize,self.nquarts)
            start = time.time()
            while 1:
                elapsed = datetime.timedelta(seconds=int(time.time()-start))
                if not task.ready():
                    time.sleep(0.1)
                else:
                    break
            end = time.time()
            print(end-start)
            
            ## Save the results from parallel
            resultsarray, quartsarray = task.result()
            
            ## Now add all of our count matrices to HDF5
            for resultsmatrix, quartsrow in zip(resultsarray, quartsarray):
                _add_quarts(quartsrow,numberdone)
                numberdone = _add_mat(resultsmatrix,numberdone)
            
            _done(numberdone,self.nquarts)
            print(numberdone)
            print(sizeargs)
            ## Exits the loop if we're out of parameter samples in the database 'args' group
            if numberdone == sizeargs:
                trigger = 0
        
        mydatabase.close()
        
        ## wrapper for ipyclient to close nicely when interrupted
        #pass
        return("Done writing database with " + str(numberdone) + " count matrices.")
    
   

### Convenience functions on toytrees
def get_all_admix_edges(ttree):
    """
    Find all possible admixture edges on a tree. Edges are unidirectional, 
    so the source and dest need to overlap in time interval.    
    """
    ## for all nodes map the potential admixture interval
    for snode in ttree.tree.traverse():
        if snode.is_root():
            snode.interval = (None, None)
        else:
            snode.interval = (snode.height, snode.up.height)
    
    ## for all nodes find overlapping intervals
    intervals = {}
    for snode in ttree.tree.traverse():
        for dnode in ttree.tree.traverse():
            if not snode.is_root() and (snode != dnode):
                ## check for overlap
                smin, smax = snode.interval
                dmin, dmax = dnode.interval

                ## find if nodes have interval where admixture can occur
                low_bin = max(smin, dmin)
                top_bin = min(smax, dmax)
                if top_bin > low_bin:
                    aedge = (snode.idx, dnode.idx, low_bin, top_bin)
                    intervals[(snode.idx, dnode.idx)] = (low_bin, top_bin)
    return intervals