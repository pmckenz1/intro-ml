#!/usr/bin/env python

"""
Generate large database of site counts from coalescent simulations 
based on msprime + toytree for using in machine learning algorithms. 
"""

## make py3 compatible
from __future__ import print_function
from builtins import range

## imports
import os
import h5py
import time
import copy
import numba
#import sklearn  # we may want to perform the ML stuff in a separate .py file
import toyplot
import toytree
import datetime
import numpy as np
import msprime as ms
import itertools as itt
import ipyparallel as ipp
from scipy.special import comb

import ipyrad as ip


class Model(object):
    """
    A coalescent model for returning ms simulations. 
    """
    def __init__(self, 
        tree,
        admixture_edges=None,
        Ne=int(1e5),
        mut=1e-5,
        nsnps=1000,
        nreps=100,
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
            A list of admixture events in the format:
            (source, dest, start, end, rate).

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
        self.nreps = nreps

        ## the counts array (result) is filled by .run()
        self.counts = None

        ## parse the input tree
        if isinstance(tree, toytree.tree):
            self.tree = tree
        elif isinstance(tree, str):
            self.tree = toytree.tree(tree)
        else:
            raise TypeError("input tree must be newick str or Toytree object")
        self.ntips = len(self.tree)

        ## store node.name as node.idx, save old names in a dict.
        self.namedict = {}
        for node in self.tree.tree.traverse():
            if node.is_leaf():
                ## store old name
                self.namedict[str(node.idx)] = node.name
                ## set new name
                node.name = str(node.idx)

        ## parse the input admixture edges. It should a list of tuples, or list
        ## of lists where each element has five values. 
        if admixture_edges:
            ## single list or tuple: [a, b, c, d, e] or (a, b, c, d, e)
            if isinstance(admixture_edges[0], (str, int)):
                admixture_edges = [admixture_edges]
        else:
            admixture_edges = []
        for event in admixture_edges:
            if len(event) != 5:
                raise ValueError(
                    "admixture events should each be a tuple with 5 values")
        self.admixture_edges = admixture_edges

        ## generate migration parameters from the tree and admixture_edges
        ## stores data in memory as self.test_values as 'mrates' and 'mtimes'
        self._get_test_values()



    def _get_test_values(self): 
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
            if all((i is not None for i in event[-3:])):
                mrates = np.repeat(event[4], self.nreps)
                mtimes = np.stack([
                    np.repeat(event[2] * 2. * self.Ne, self.nreps), 
                    np.repeat(event[3] * 2. * self.Ne, self.nreps)], axis=1)
                self.test_values[idx] = {"mrates": mrates, "mtimes": mtimes}

            ## otherwise generate uniform values across edges
            else:        
                ## get migration rates from zero to ~full
                minmig = 0.0
                maxmig = 0.5
                mrates = np.random.uniform(minmig, maxmig, self.nreps)

                ## get divergence times from source start to end
                self._intervals = get_all_admix_edges(self.tree)                
                snode = self.tree.tree.search_nodes(idx=event[0])[0]
                dnode = self.tree.tree.search_nodes(idx=event[1])[0]
                interval = self._intervals[snode.idx, dnode.idx]
                edge_min = int(interval[0] * 2. * self.Ne)
                edge_max = int(interval[1] * 2. * self.Ne)
                mtimes = np.sort(
                    np.random.uniform(edge_min, edge_max, self.nreps*2)
                    .reshape((self.nreps, 2)), axis=1).astype(int)
                self.test_values[idx] = {"mrates": mrates, "mtimes": mtimes}
                if self._debug:
                    print("uniform testvals mig:", 
                         (edge_min, edge_max), (minmig, maxmig))
            idx += 1


    def plot_test_values(self):
        """
        Returns a toyplot canvas 
        """
        ## setup canvas
        canvas = toyplot.Canvas(height=250, width=800)

        ax0 = canvas.cartesian(
            grid=(1, 3, 0))
        ax1 = canvas.cartesian(
            grid=(1, 3, 1), 
            xlabel="migration durations", 
            ylabel="simulation index",
            xmin=0, 
            xmax=self.tree.tree.height * 2 * self.Ne)
        ax2 = canvas.cartesian(
            grid=(1, 3, 2), 
            xlabel="proportion migrants", 
            ylabel="frequency")

        ## advance colors for different edges starting from 1
        colors = iter(toyplot.color.Palette())

        ## draw tree
        self.tree.draw(
            tree_style='c', 
            node_labels="idx", 
            tip_labels=False, 
            axes=ax0,
            node_size=18,
            padding=50)
        ax0.show = False

        ## iterate over edges 
        keys = list(self.test_values.keys())
        for tidx in range(len(keys)):
            color = colors.next()            

            ## get values for the first admixture edge
            mtimes = self.test_values[keys[tidx]]["mtimes"]
            mrates = self.test_values[keys[tidx]]["mrates"]
            mt = mtimes[mtimes[:, 1].argsort()]
            boundaries = np.column_stack((mt[:, 0], mt[:, 1]))

            ## plot
            for idx in range(boundaries.shape[0]):
                ax1.fill(
                    boundaries[idx], 
                    (idx, idx), 
                    (idx+0.5, idx+0.5),
                    color=color, 
                    opacity=0.5)
            ax2.bars(np.histogram(mrates, bins=20), color=color, opacity=0.5)

        return canvas


    ## functions to build simulation options 
    def _get_demography(self, idx):
        """
        returns demography scenario based on an input tree and admixture
        edge list with events in the format (source, dest, start, end, rate)
        """
        ## Define demographic events for msprime
        demog = set()

        ## tag min index child for each node, since at the time the node is 
        ## called it may already be renamed by its child index b/c of 
        ## divergence events.
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
            children = (snode._schild, dnode._schild)
            demog.add(ms.MigrationRateChange(time[0], rate, children))
            demog.add(ms.MigrationRateChange(time[1], 0, children))
            if self._debug:
                print('demog mig:', 
                    (round(time[0], 4), round(time[1], 4), 
                     round(rate, 4), children))

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
            ms.PopulationConfiguration(sample_size=1, initial_size=self.Ne)
            for ntip in range(self.ntips)]
        return population_configurations


    def _simulate(self, idx):
        """
        performs simulations with params varied across input values.
        """       
        ## set up simulation
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()
        sim = ms.simulate(
            num_replicates=self.nsnps * 100,  # 100X since some sims are empty
            mutation_rate=self.mut,
            migration_matrix=migmat,
            population_configurations=self._get_popconfig(),
            demographic_events=self._get_demography(idx)
        )
        return sim


    def run(self):
        """
        run and parse results for nsamples simulations.
        """
        ## storage for output
        self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb
        self.counts = np.zeros(
            (self.nreps, self.nquarts, 16, 16), dtype=np.uint64)

        ## iterate over nreps (different sampled simulation parameters)
        for ridx in range(self.nreps):
            ## run simulation for demography ridx
            ## yields a generator of trees to sample from with next()
            ## we select 1 SNP from each tree with shape (1, ntaxa)
            ## repeat until snparr is full with shape (nsnps, ntips)
            sims = self._simulate(ridx)

            ## store results (nsnps, ntips); def. 1000 SNPs
            snparr = np.zeros((self.nsnps, self.ntips), dtype=np.uint16)

            ## continue until all SNPs are sampled from generator
            fidx = 0
            while fidx < self.nsnps:
                ## get genotypes and convert to {0,1,2,3} under JC
                bingenos = sims.next().genotype_matrix()
                sitegenos = mutate_jc(bingenos, self.ntips)
                ## count as 16x16 matrix and store to snparr
                if sitegenos.size:
                    snparr[fidx] = sitegenos
                    fidx += 1

            ## keep track for counts index
            quartidx = 0

            ## iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4), etc.
            qiter = itt.combinations(xrange(self.ntips), 4)
            for currquart in qiter:
                ## cols indices match tip labels b/c we named tips to node.idx
                quartsnps = snparr[:, currquart]
                self.counts[ridx, quartidx] = count_matrix(quartsnps)
                quartidx += 1


## jitted functions for running super fast -----------------
@numba.jit(nopython=True)
def count_matrix(quartsnps):
    """
    return a 16x16 matrix of site counts from snparr
    """
    arr = np.zeros((16, 16), dtype=np.uint64)
    add = np.uint64(1) 
    for idx in range(quartsnps.shape[0]):
        i = quartsnps[idx, :]
        arr[(4*i[0])+i[1], (4*i[2])+i[3]] += add    
    return arr


@numba.jit(nopython=True)
def mutate_jc(geno, ntips):
    """
    mutates sites with 1 into a new base in {0, 1, 2, 3}
    """
    allbases = np.array([0, 1, 2, 3])
    for ridx in np.arange(geno.shape[0]):
        snp = geno[ridx]
        if snp.sum():
            init = np.zeros(ntips, dtype=np.int64)
            init.fill(np.random.choice(allbases))
            notinit = np.random.choice(allbases[allbases != init[0]])
            init[snp.astype(np.bool_)] = notinit
            return init
    return np.zeros(0, dtype=np.int64)  # return dtypes must match


class DataBase(object):
    """
    An object to parallelize simulations over many parameter settings
    and store finished reps in a HDF5 database. 

    Parameters:
    -----------
    name: str
        The name that will be used in the saved database file (<name>.hdf5)

    workdir: str
        The location where the database file will be saved, or loaded from 
        if continuing an analysis from a checkpoint. 

    tree: newick or toytree
        A fixed topology to use for all simulations. Edge lengths are fixed
        unless the argument 'edge_function' is used, in which case edge lengths 
        are drawn from a distribution.

    edge_function: None or dict (default=None)
        If an edge_function argument is entered then edge lengths on the 
        topology are drawn from one of the supported distributions. The 
        following options are available: 

        {
         "jitter": percentage,
         "yule": birthrate
         "birth-death": (birthrate, deathrate)
         "coalescent": None
        }

    nedges: int (default=0)
        The number of admixture edges to add to each tree at a time. All edges
        will be drawn on the tree that can connect any branches which overlap 
        for a nonzero amount of time. A set of nedges is referred to as an 
        admixture event, and ntests*nreps are repeated for each admixture event
        so that the total data points = nevents * ntests * nreps. The number
        of admixture events generated by 'nedges' depends on the shape of the 
        tree and its branch lengths, and so will vary among 

    ntests: int (default=1)
        The number of sampled trees to perform tests across. Sampled trees
        have the topology of the input tree but with branch lengths modified
        according to the function in 'edge_function'. If None then tests repeat
        using the same tree (same effect as nreps). 

    nreps: int (default=100)
        The number of replicate simulations to run per (sampled tree,
        admixture scenario, and parameter set). 

    nsnps: int (default=1000)
        The number of SNPs in each simulation that are used to build the 
        16x16 arrays of phylogenetic invariants for each quartet sample. 

    Ne: int or tuple (default=1e6)
        The effective population size for all edges on the tree. If a single
        value is entered then Ne is fixed across the tree. If a tuple is 
        entered then Ne values are drawn from a uniform distribution between 
        (low, high). 

    seed: int (default=123)
        Set the seed of the random number generator

    force: bool (default=False)
        Force overwrite of existing database file.
    """
    def __init__(self,
        name,
        workdir,
        tree,
        edge_function=None,
        nsnps=1000,
        nedges=0,
        ntests=1,
        nreps=100,
        Ne=1e6,
        seed=123,
        force=False,
        debug=False,
        quiet=False,
        **kwargs):

        ## identify this set of simulations
        self.name = name
        self.workdir = (workdir or 
            os.path.realpath(os.path.join('.', "databases")))
        self.database = os.path.join(workdir, self.name+".hdf5")
        self._db = None  # open/closed file handle of self.database
        self._debug = debug
        self._quiet = quiet

        ## store params
        self.tree = tree
        self.edge_function = (edge_function or {})
        self.nedges = nedges
        self.ntests = ntests        
        self.nreps = nreps
        self.nstored_values = None
        self.nsnps = nsnps
        self.Ne = Ne

        ## store ipcluster information 
        self._ipcluster = {
            "cluster_id": "", 
            "profile": "default",
            "engines": "Local", 
            "quiet": 0, 
            "timeout": 60, 
            "cores": 0, 
            "threads": 2,
            "pids": {},
            }        

        ## a generator that returns branch lengthed trees
        self.tree_generator = self._get_tree_generator()

        ## make sure workdir exists
        if not os.path.exists(workdir):
            os.makedirs(workdir)

        ## create database in 'w-' mode to prevent overwriting
        if os.path.exists(self.database):
            if force:
                ## exists and destroy it
                answer = raw_input(
                    "Do you really want to overwrite the database? (y/n) ")
                if answer in ('yes', 'y', "Y"):
                    os.remove(self.database)
                    self._db = h5py.File(self.database, mode='w')
                else:
                    ## apparently the user didn't mean to use force
                    print('Aborted: remove force argument if not overwriting')
                    return 
            else:
                ## exists append to it
                self._db = h5py.File(self.database, mode='a')
        else:
            ## does not exist
            self._db = h5py.File(self.database, mode='w-')     

        ## Create h5 datasets for these simulations
        if not self._db.get("counts"):
            self._generate_fixed_tree_database()

        ## Fill all params into the database (this inits the Model objects 
        ## which call ._get_test_values() to generate all simulation scenarios
        ## which are then entered into the database for the next nreps sims
        self._fill_fixed_tree_database_labels()

        ## print info about the database in debug mode
        self._debug_report()
        if not self._quiet:
            print("stored {} labels to {}"
                .format(self.nstored_values, self.database))

        ## Close the database. It is now ready to be filled with .run()
        ## which will run until all tests are finished in the database. We 
        ## could then return to this Model object and add more tests later 
        ## if we wanted by using ...<make this>
        self._db.close()


    def _debug_report(self):
        """
        Prints to screen info about the size of the database if debug=True.
        Assumes the self._db handle is open in read-mode
        """
        if self._debug:
            keys = self._db.keys()
            for key in keys:
                print(key, self._db[key].shape)


    def _get_tree_generator(self):
        """
        A generator that infinitely returns trees. If edge_function then the 
        trees are modified to sample edge lengths from a distribution, if not
        then the input tree is simply returned. 
        """
        while 1:
            if self.edge_function == "node_slider":
                yield node_slider(self.tree)
            elif self.edge_function == "poisson":
                #raise NotImplementedError("Not yet supported")
                yield exp_sampler(self.tree)
            else:
                yield self.tree


    def _generate_fixed_tree_database(self):
        """
        Parses parameters in self.params to create all combinations
        of parameter values to test. Returns the number of the simulations.
        Simulation metadata is appended to datasets. 

        Expect that the h5 file self._db is open in w or a mode.
        """

        ## the number of data points will be nreps x the number of events
        ## uses scipy.special.comb
        admixedges = get_all_admix_edges(self.tree)
        nevents = int(comb(N=len(admixedges), k=self.nedges))
        nvalues = self.ntests * self.nreps * nevents
        nquarts = int(comb(N=len(self.tree), k=4))
        self.nstored_values = nvalues

        if self._debug:
            print()
            print('ntests', self.ntests)
            print('nreps', self.nreps)
            print('nevents', nevents)
            print('nvalues', nvalues)
            print('nquarts', nquarts)

        ## make fixed-length string data type
        dt = np.dtype("S"+str(len(self.tree.tree.write())))

        ## store topology, just once
        self._db.create_dataset("topology", 
            shape=(1,),
            dtype=dt)

        ## store node key, just once
        self._db.create_dataset("nodekey", 
            shape=(len([i.idx for i in self.tree.tree.search_nodes()])-len([i 
                for i in self.tree.tree.get_leaves()]),),
            dtype=np.uint32)        

        ## store count matrices
        self._db.create_dataset("counts", 
            shape=(nvalues, nquarts, 16, 16),
            dtype=np.uint32)

        ## store edge lengths: now NODE HEIGHTS
        self._db.create_dataset("edge_lengths", 
            shape=(nvalues, len([i.idx for i in self.tree.tree.search_nodes()])-len([i 
                for i in self.tree.tree.get_leaves()])),
            dtype=np.float64)

        ## store admixture sources and targets in order
        self._db.create_dataset("admix_sources", 
            shape=(nvalues, self.nedges),
            dtype=np.uint8)
        self._db.create_dataset("admix_targets", 
            shape=(nvalues, self.nedges),
            dtype=np.uint8)
        self._db.create_dataset("admix_props", 
            shape=(nvalues, self.nedges),
            dtype=np.float64)
        self._db.create_dataset("admix_tstart", 
            shape=(nvalues, self.nedges),
            dtype=np.float64)
        self._db.create_dataset("admix_tend", 
            shape=(nvalues, self.nedges),
            dtype=np.float64)

        ## store parameters of the simulation
        self._db.create_dataset("Ne",
            shape=(nvalues, 1),
            dtype=np.uint64)


    def _fill_fixed_tree_database_labels(self):
        """
        This iterates across generated trees and creates simulation scenarios
        for nreps iterations for each admixture edge(s) scenario in the tree
        and stores the full parameter information into the hdf5 database.
        """
        ## fill topology first
        self._db["topology"][0]=self.tree.tree.write()

        ## fill in a key for internal nodes
        self._db["nodekey"][:]=np.array(list(set([i.idx 
            for i in self.tree.tree.search_nodes()])-set([i.idx 
            for i in self.tree.tree.get_leaves()])))

        ## iterate until until all tests are sampled
        tidx = 0
        for _ in range(self.ntests):

            ## get the next tree
            itree, idict = self.tree_generator.next()

            ## store edge lengths for labels in node.idx order
            edge_lengths = (idict.values()/max(idict.values()))*2*self.Ne

            ## get all admixture edges that can be drawn on this tree
            admixedges = get_all_admix_edges(itree)

            ## iterate over each possible (edge, interval) item, or pairs or 
            ## triplets, etc., of them depending on the number of admix_edges
            eidx = tidx
            events = itt.combinations(admixedges.items(), self.nedges)
            for event in events:

                ## initalize a Model to sample range of parameters on this edge
                ## model counts array shape: (ntests*nreps, nquarts, 16, 16)
                admixlist = [(i[0][0], i[0][1], None, None, None) 
                    for i in event]

                ## for help 
                if self._debug:
                    print('admixlist', admixlist)

                model = Model(itree, 
                    admixture_edges=admixlist,
                    )

                ## store labels for this sim (1 x nreps)
                self._db["edge_lengths"][eidx:eidx+self.nreps] = edge_lengths
                self._db["Ne"][eidx:eidx+self.nreps] = self.Ne

                ## get labels from admixlist and model.test_values
                for xidx in range(len(admixlist)):
                    sources = np.repeat(admixlist[xidx][0], self.nreps)
                    targets = np.repeat(admixlist[xidx][1], self.nreps)
                    mrates = model.test_values[xidx]["mrates"]
                    mtimes = model.test_values[xidx]["mtimes"]

                    ## store labels for this admix event (nevents x nreps)
                    s, e = eidx, eidx + self.nreps

                    self._db["admix_sources"][s:e, xidx] = sources
                    self._db["admix_targets"][s:e, xidx] = targets
                    self._db["admix_props"][s:e, xidx] = mrates
                    self._db["admix_tstart"][s:e, xidx] = mtimes[:, 0]
                    self._db["admix_tend"][s:e, xidx] = mtimes[:, 1]

                eidx += self.nreps
            tidx += eidx



    ## THE MAIN RUN COMMANDS ----------------------------------------
    ## Distributes parallel jobs and wraps functions for convenient cleanup.
    def run2(self, ipyclient=None, quiet=False):
        """
        Run inference in

        Parameters
        ----------
        ipyclient (ipyparallel.Client object):
            A connected ipyclient object. If ipcluster instance is 
            not running on the default profile then ...
        """

        ## open the database file in read/write, will throw error if not exist
        self._db = h5py.File(self.database, mode='r+')
        self.tree = toytree.tree(self._db['topology'][0])
        ## figure out how many are already done
        numdone = np.sum(np.sum(self._db['counts'],axis = (1,2,3)) > 0)
        startidx = numdone
        ntotal = len(self._db['counts'])
        batsize = 500
        endidx = startidx+min(batsize,(ntotal-startidx))
        ## pull a batch to run
        bat_edge_lengths = self._db['edge_lengths'][startidx:endidx,]
        bat_Ne = self._db['Ne'][startidx:endidx,]
        bat_ad_sources = self._db['admix_sources'][startidx:endidx,]
        bat_ad_targets = self._db['admix_targets'][startidx:endidx,]
        bat_ad_props = self._db['admix_props'][startidx:endidx,]
        bat_ad_tstart = self._db['admix_tstart'][startidx:endidx,]
        bat_ad_tend = self._db['admix_tend'][startidx:endidx,]
        bat_counts = np.zeros(shape=(np.shape(self._db['counts'][startidx:endidx])))

        trees=[]
        for i in range(len(bat_edge_lengths)):
            n=dict(set(zip(*[self._db["nodekey"],bat_edge_lengths[i]])))
            tree=copy.deepcopy(self.tree)
            for leaf in tree.tree.get_leaves():
                for node in leaf.iter_ancestors():
                    set_node_height(node,n[node.idx])
            trees.append(tree)

        ## initialize a bunch of models
        models = []
        for i in range(batsize):
            samp_edges=(bat_ad_sources[i][0],bat_ad_targets[i][0], bat_ad_tstart[i][0], bat_ad_tend[i][0],bat_ad_props[i][0])
            models.append(Model(trees[i], admixture_edges=[samp_edges],Ne = bat_Ne[i]))

        ## wrap the run in a try statement to ensure we properly shutdown
        ## and cleanup on exit or interrupt. 
        inst = None
        try:
            ## find and connect to an ipcluster instance given the information
            ## in the _ipcluster dictionary if a connected client was not given
            if not ipyclient:
                args = self._ipcluster.items() + [("spacer", "")]
                ipyclient = ip.core.parallel.get_client(**dict(args))

            ## print the cluster connection information
            if not quiet:
                ip.cluster_info(ipyclient)

            ## store ipyclient engine pids to the dict so we can 
            ## hard-interrupt them later if assembly is interrupted. 
            ## Only stores pids of engines that aren't busy at this moment, 
            ## otherwise it would block here while waiting to find their pids.
            self._ipcluster["pids"] = {}
            for eid in ipyclient.ids:
                engine = ipyclient[eid]
                if not engine.outstanding:
                    pid = engine.apply(os.getpid).get()
                    self._ipcluster["pids"][eid] = pid   

            ## execute here...
            print("ready to run")

            ## give it a bunch of model objects
            ## expect back a bunch of count matrices



        ## handle exceptions so they will be raised after we clean up below
        except KeyboardInterrupt as inst:
            print("\nKeyboard Interrupt by user. Cleaning up...")

        except Exception as inst:
            print("\nUnknown exception encountered: {}".format(inst))

        ## close client when done or interrupted
        finally:
            try:
                ## save the Assembly
                #self._save()                

                ## can't close client if it was never open
                if ipyclient:

                    ## send SIGINT (2) to all engines
                    ipyclient.abort()
                    time.sleep(1)
                    for engine_id, pid in self._ipcluster["pids"].items():
                        if ipyclient.queue_status()[engine_id]["tasks"]:
                            os.kill(pid, 2)
                        time.sleep(0.25)

            ## if exception during shutdown then we really screwed up
            except Exception as inst2:
                print("warning: error during shutdown:\n{}".format(inst2))





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
            Resize your HDF5 'counts' group at the end to the same length 
            as filled count matrices. Collette book page 40.
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
                for idx in range(windowsize):
                    treenum, sourcebr, destbr, Ne, nsnps, seed = argsints[idx,:]
                    mtimerecent, mtimedistant, mrate, mut = argsflts[idx,:]
                    mod = Model(tree = trees[treenum],
                                admixture_edges = [(sourcebr,destbr,mtimerecent,mtimedistant,mrate)],
                                Ne = Ne,
                                nsnps = nsnps,
                                mut = mut,
                                seed = seed,
                                nreps = 1)
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


def node_slider(ttree):
    """
    Returns a toytree copy with node heights modified while retaining the 
    same topology but not necessarily node branching order. Node heights are
    moved up or down uniformly between their parent and highest child node 
    heights in 'levelorder' from root to tips. The total tree height is 
    retained at 1.0, only relative edge lengths change.

    ## for example run:
    c, a = node_slide(ctree).draw(
        width=400,
        orient='down', 
        node_labels='idx',
        node_size=15,
        tip_labels=False
        );
    a.show = True
    a.x.show = False
    a.y.ticks.show = True
    """
    ctree = copy.deepcopy(ttree)
    for node in ctree.tree.traverse():

        ## slide internal nodes 
        if node.up and node.children:

            ## get min and max slides
            minjit = max([i.dist for i in node.children]) * 0.99
            maxjit = (node.up.height * 0.99) - node.height
            newheight = np.random.uniform(low=-minjit, high=maxjit)

            ## slide children
            for child in node.children:
                child.dist += newheight

            ## slide self to match
            node.dist -= newheight

    ## make max height = 1
    mod = ctree.tree.height
    for node in ctree.tree.traverse():
        node.dist = node.dist / float(mod)

    return ctree

def exp_sampler(ttree, betaval = 1,returndict = "both"):
    """
    Takes an input topology and samples branch lengths
    
    Parameters:
    -----------
    toytreeobj: toytree
        The topology for which we want to generate branch lengths
    betaval: int/float (default=1)
        The value of beta for the exponential distribution used to generate
        branch lengths. This is inverse of rate parameter.
    returndict: str (default=None)
        If "only", returns only a dictionary matching nodes to heights.
        If "both", returns toytree and dictionary as a tuple.
    """
    tree=copy.deepcopy(ttree)
    def set_node_height(node,height):
        childn=node.get_children()
        for child in childn:
            child.dist=height - child.height
    testobj = []
    
    # we'll just append each step on each chain to these lists to keep track of where we've already been
    allnodes = []
    allnodeheights = []
    
    # d is deprecated (it was being returned) and stores the values for each chain
    #d = {}
    
    ## testobj is our traversed full tree
    for i in tree.tree.traverse():
        testobj.append(i)
    
    # start by getting branch lengths for the longest branch
    longbranch=sum([testobj[-1] in i for i in testobj]) # counts the number of subtrees containing last leaf, giving length of longest branch
    longbranchnodes = np.random.exponential(betaval,longbranch) # segment lengths of longest branch
    longbranchnodes[0] = np.random.uniform(low=0.0,high=longbranchnodes[0]) # cut the edge to the leaf with a uniform draw
    
    # get the heights of nodes along this longest (most nodes) branch
    nodeheights = []
    currheight = 0
    for i in longbranchnodes:
        currheight += i
        nodeheights.append(currheight)
    nodeheights = nodeheights[::-1]

    # get indices to accompany long chain
    lcidx = []
    for i in testobj:
        if testobj[-1] in i.get_leaves():
            lcidx.append(i.idx)
    #d['0heights'] = np.array(nodeheights)
    #d['0nodes'] = np.array(lcidx[:-1])
    
    allnodes = allnodes + lcidx[:-1]
    allnodeheights = allnodeheights + nodeheights
    
    # get other necessary chains to parse
    other_chains = []
    for i in range(len(testobj)-1)[::2]:
        if len(testobj[i+1].get_leaves()) > 1:
            other_chains.append(testobj[i+1])

    # now solve
    for chainnum in range(len(other_chains)): # parse the remaining chains one at a time
        otr=other_chains[chainnum]
        # find where this chain connects to the a chain we've already solved
        firstancestor = otr.get_ancestors()[0].idx
        # which nodeheight does this branch from
        paridx=np.argmax(np.array(allnodes) == firstancestor) 
        
        # traverse the new 
        testobj1 = []
        nodes = []
        
        # save a list of nodes
        for i in otr.traverse():
            testobj1.append(i)
        
        # save the nodes that include the end of the chain (because branches out to other chains might not)
        for i in testobj1:
            if testobj1[-1] in i.get_leaves():
                nodes.append(i.idx)
        
        # make node index list to accompany lengths
        lennodes= nodes[:-1] # don't save ending leaf index
        lennodes.insert(0,firstancestor) # make chain list start with ancestor
        
        # figure out how many exponential draws to make for this chain (i.e. # new nodes + 1)
        num_new_branches = sum([testobj1[-1] in i for i in testobj1])+1
        
        # initialize array to hold the draws
        mir_lens = np.zeros((sum([testobj1[-1] in i for i in testobj1])+1))
        # draw until we have a new set of exponential branch lengths that fit the constraints of our tree height
        while not (sum(mir_lens[:(len(mir_lens)-1)]) < allnodeheights[paridx] and (sum(mir_lens) > allnodeheights[paridx])):
            mir_lens = np.random.exponential(betaval,num_new_branches) ## length of longest branch
        
        # now let's save each node value as a height
        mir_lens_heights = np.zeros((len(mir_lens)))
        subsum = 0
        for i in range(len(mir_lens)):
            mir_lens_heights[i] = allnodeheights[paridx] - subsum
            subsum = subsum + mir_lens[i]
            
        # add our new node indices with their heights to the full list
        allnodes = list(allnodes) + list(lennodes)
        allnodeheights = list(allnodeheights) + list(mir_lens_heights)
        
        #d[(str(chainnum+1)+"heights")] = mir_lens_heights
        #d[(str(chainnum+1)+"nodes")] = np.array(lennodes)
    
    # make a final dictionary of node heights, eliminating redundancy
    n = dict(set(zip(*[allnodes,allnodeheights])))
    
    if returndict == "only":
        return n #d
    elif returndict == "both":
        # create the tree object
        for leaf in tree.tree.get_leaves():
            for node in leaf.iter_ancestors():
                set_node_height(node,n[node.idx])
        return (tree,n)
    else:
        # create the tree object
        for leaf in tree.tree.get_leaves():
            for node in leaf.iter_ancestors():
                set_node_height(node,n[node.idx])
            ## make max height = 1
        mod = tree.tree.height
        for node in tree.tree.traverse():
            node.dist = node.dist / float(mod)
        return tree

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
                    intervals[(snode.idx, dnode.idx)] = (low_bin, top_bin)
    return intervals

def set_node_height(node,height):
    """
    set an ete3 node object at a particular height
    """
    childn=node.get_children()
    for child in childn:
        child.dist=height - child.height
