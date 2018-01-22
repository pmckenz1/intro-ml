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



class Model(object):
    """
    A coalescent model for returning ms simulations. 
    """
    def __init__(self, 
        tree, 
        admixture_edges=[], 
        Ne=int(1e5), 
        mut=1e-8, 
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
        self.counts = np.zeros((self.ntests, 16, 16), dtype=int)    
        for ridx in range(self.ntests):
            ## run simulation for demography idx
            sims = self._simulate(ridx)
            
            ## array to store site counts
            carr = np.zeros((self.nsnps, 16, 16))
            
            ## continue until nsnps are simulated
            fidx = 0
            while fidx < self.nsnps:
                ## get just the first mutation
                bingenos = sims.next().genotype_matrix()
                ## convert to sequence under JC
                sitegenos = self.mutate_jc(bingenos)
                ## count it
                if sitegenos.size:
                    carr[fidx] = count(sitegenos)
                    fidx += 1
                    
            ## fill site counts into 16x16 matrix
            self.counts[ridx] = carr.sum(axis=0)



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
    and store finished reps in a HDF5 database    
    """
    def __init__(self, name, workdir="sim-databases/", force=False, **kwargs):
        
        ## identify this set of simulations
        self.name = name
        self.workdir = workdir
        self.path = os.path.join(workdir, self.name+".hdf5")
        
        ## make sure workdir exists
        if not os.path.exists(workdir):
            os.makedirs(workdir)
        
        ## create database in 'w-' mode to prevent overwriting
        if os.path.exists(self.path):
            if force:
                ## exists and destroy it
                if raw_input("Are you sure you want to overwrite the database? "):
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
        self.ndatasets = self._generate_database()

        
    def _generate_database(self):
        """
        Parses parameters in self.params to create all combinations
        of parameter values to test. Returns the number of the simulations.
        Simulation metadata is appended to datasets. 
        """
        return 1
        




    def run(self, ipyclient, force=False):
        """
        Distribute simulations across a parallel Client. If continuing
        a previous run then any unfinished simulation will be queued up
        to run. 
        """
        
        ## wrapper for ipyclient to close nicely when interrupted
        pass
    
   

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

