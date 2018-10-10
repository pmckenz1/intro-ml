#!/usr/bin/env python

"""
Generate large database of site counts from coalescent simulations
based on msprime + toytree for using in machine learning algorithms.
"""

# imports for py3 compatibility
from __future__ import print_function
from builtins import range, input

# imports
import os
import sys
import h5py
import time
import copy
import numba
import toyplot
import toytree
import datetime
import numpy as np
import msprime as ms
import itertools as itt
import ipyparallel as ipp
from scipy.special import comb

from .parallel import cluster_info


######################################################
class SimcatError(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


#######################################################
class Model:
    """
    A coalescent model for returning ms simulations.
    """
    def __init__(
        self, 
        tree,
        admixture_edges=None,
        theta=0.01,
        nsnps=1000,
        ntests=10,
        nreps=1,
        seed=None,
        debug=False,
        mut=1e-8
        ):
        """
        An object used for demonstration and testing only. The real simulations
        use the similar object Simulator.

        Takes an input topology with edge lengths in coalescent units (2N) 
        entered as either a newick string or as a Toytree.tree object,
        and generates 'ntests' parameter sets for running msprime simulations 
        which are stored in the the '.test_values' dictionary. The .run() 
        command can be used to execute simulations to fill count matrices 
        stored in .counts. 

        Parameters:
        -----------
        tree: (str)
            A newick string or Toytree object of a species tree with edges in
            coalescent units.

        admixture_edges (list):
            A list of admixture events in the format:
            (source, dest, start, end, rate). If start, end, or rate are 
            empty then they will be sampled from possible points given the 
            topology of the tree. 

        theta (int or tuple):
            Mutation parameter.

        nsnps (int):
            Number of unlinked SNPs simulated (e.g., counts is (nsnps, 16, 16))

        ntests (int):
            Number of parameter sets to sample, where _theta is sampled, and 
            for each admixture edge a migration start, migration end, and 
            migration rate is sampled. The counts array is expanded to be 
            (ntests, nsnps, 16, 16)

        seed (int):
            Random number generator
        """
        # init random seed
        if seed:
            np.random.seed(seed)

        # hidden argument to turn on debugging
        self._debug = debug

        # store sim params as attrs
        if isinstance(theta, (float, int)):
            self._rtheta = (theta, theta)
        else:
            self._rtheta = (min(theta), max(theta))

        # fixed _mut; _theta sampled from theta; and _Ne computed for diploid
        self._length = 1000
        self._mut = mut
        self._theta = np.random.uniform(self._rtheta[0], self._rtheta[1])

        # dimension of simulations
        self.nsnps = nsnps
        self.ntests = ntests
        self.nreps = nreps

        # the counts array (result) is filled by .run()
        self.counts = None

        # parse the input tree
        if isinstance(tree, toytree.tree):
            self.tree = tree
        elif isinstance(tree, str):
            self.tree = toytree.tree(tree)
        else:
            raise TypeError("input tree must be newick str or Toytree object")
        self.ntips = len(self.tree)

        ## storage for output
        self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb
        self.counts = np.zeros(
            (self.ntests * self.nreps, self.nquarts, 16, 16), dtype=np.float32)

        # store node.name as node.idx, save old names in a dict.
        self.namedict = {}
        for node in self.tree.tree.traverse():
            if node.is_leaf():
                # store old name
                self.namedict[str(node.idx)] = node.name
                # set new name
                node.name = str(node.idx)

        # parse the input admixture edges. It should a list of tuples, or list
        # of lists where each element has five values.
        if admixture_edges:
            # single list or tuple: [a, b, c, d, e] or (a, b, c, d, e)
            if isinstance(admixture_edges[0], (str, int)):
                admixture_edges = [admixture_edges]
        else:
            admixture_edges = []
        for event in admixture_edges:
            if len(event) != 5:
                raise ValueError(
                    "admixture events should each be a tuple with 5 values")
        self.admixture_edges = admixture_edges
        self.aedges = len(self.admixture_edges)

        ## generate migration parameters from the tree and admixture_edges
        ## stores data in memory as self.test_values as 'mrates' and 'mtimes'
        self._get_test_values()


    @property
    def _Ne(self):
        "Ne is automatically calculated from theta and fixed mut"
        return (self._theta / self._mut) / 4.


    def _get_test_values(self): 
        """
        Generates mrates, mtimes, and thetas arrays for simulations. 

        Migration times are uniformly sampled between start and end points that
        are constrained by the overlap in edge lengths, which is automatically
        inferred from 'get_all_admix_edges()'. migration rates are drawn 
        uniformly between 0.0 and 0.5. thetas are drawn uniformly between 
        theta0 and theta1, and Ne is just theta divided by a constant. 
        """
        ## init a dictionary for storing arrays for each admixture scenario
        self.test_values = {}

        ## store sampled theta values across ntests
        self.test_values["thetas"] = np.random.uniform(
            self._rtheta[0], self._rtheta[1], self.ntests)

        ## store evt: (mrates, mtimes) for each admix event in admixture list
        intervals = get_all_admix_edges(self.tree)
        idx = 0
        for event in self.admixture_edges:

            ## if times and rate were provided then use em.
            if all((i is not None for i in event[-3:])):
                mrates = np.repeat(event[4], self.ntests)

                ## raise an error if mtime is not possible
                ival = intervals[(event[0], event[1])]
                if (event[2] >= ival[0]) and (event[3] <= ival[1]):
                    ## record timing in generations
                    e2 = np.repeat(event[2], self.ntests)
                    e3 = np.repeat(event[3], self.ntests)
                    mtimes = np.stack((e2, e3), axis=1)
                else:
                    print(ival, event)
                    raise Exception("bad migration interval")

                ## store migration arrays
                self.test_values[idx] = {
                    "mrates": mrates, 
                    "mtimes": mtimes, 
                    }

            ## otherwise generate uniform values across edges
            else:        
                ## get migration rates from exponential
                mrates = np.random.exponential(0.1, self.ntests)
                mrates[mrates > 0.99] = 0.99

                ## get divergence times from source start to end                
                snode = self.tree.tree.search_nodes(idx=event[0])[0]
                dnode = self.tree.tree.search_nodes(idx=event[1])[0]
                ival = intervals[snode.idx, dnode.idx]

                ## interval is stored as an int, and is bls in generations
                ui = np.random.uniform(ival[0], ival[1], self.ntests * 2)
                ui = ui.reshape((self.ntests, 2))
                mtimes = np.sort(ui, axis=1)  

                self.test_values[idx] = {
                    "mrates": mrates, 
                    "mtimes": mtimes,
                    }
                if self._debug:
                    print("uniform testvals mig:", 
                          (ival[0], ival[1]), (mrates[0], mrates[1]))
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
            xlabel="simulation index",
            ylabel="migration intervals", 
            ymin=0, 
            ymax=self.tree.tree.height)  # * 2 * self._Ne)
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
        for tidx in range(self.aedges):
            color = colors.next()            

            ## get values for the first admixture edge
            mtimes = self.test_values[tidx]["mtimes"]
            mrates = self.test_values[tidx]["mrates"]
            mt = mtimes[mtimes[:, 1].argsort()]
            boundaries = np.column_stack((mt[:, 0], mt[:, 1]))

            ## plot
            for idx in range(boundaries.shape[0]):
                ax1.fill(
                    boundaries[idx],
                    (idx, idx),
                    (idx + 0.5, idx + 0.5),
                    along='y',
                    color=color, 
                    opacity=0.5)
            ax2.bars(np.histogram(mrates, bins=20), color=color, opacity=0.5)

        return canvas


    ## functions to build simulation options 
    def _get_demography(self):
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
                time = node.height * 2. * self._Ne  
                demog.add(ms.MassMigration(time, source, dest))
                if self._debug:
                    print('demog div:', (int(time), source, dest), 
                        file=sys.stderr)

        ## Add migration edges
        for evt in range(self.aedges):
            rate = self.test_values[evt]['mrates']
            time = self.test_values[evt]['mtimes'][0] * 2. * self._Ne
            print(time)
            source, dest = self.admixture_edges[evt][:2]

            ## rename nodes at time of admix in case divergences renamed them
            snode = self.tree.tree.search_nodes(idx=source)[0]
            dnode = self.tree.tree.search_nodes(idx=dest)[0]
            children = (snode._schild, dnode._schild)
            demog.add(ms.MigrationRateChange(time[0], rate, children))
            demog.add(ms.MigrationRateChange(time[1], 0, children))
            if self._debug:
                print(
                    'demog mig:', 
                    (time[0], time[1]), 
                    round(rate, 4), 
                    children,
                    self._Ne, 
                    file=sys.stderr,
                    )

        ## sort events by time
        demog = sorted(list(demog), key=lambda x: x.time)
        print(demog)
        return demog


    def _get_popconfig(self):
        """
        returns population_configurations for N tips of a tree
        """
        population_configurations = [
            ms.PopulationConfiguration(sample_size=1, initial_size=self._Ne)
            for ntip in range(self.ntips)]
        return population_configurations


    def _simulate(self, idx):
        """
        performs simulations with params varied across input values.
        """       
        # store _temp values for this idx simulation, 
        # Ne will be calculated from theta.
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()
        self._theta = self.test_values["thetas"][idx]
        self._mtimes = [self.test_values[evt]['mtimes'][idx] for evt in 
                        range(len(self.admixture_edges))] 
        self._mrates = [self.test_values[evt]['mrates'][idx] for evt in 
                        range(len(self.admixture_edges))]         

        ## build msprime simulation
        #sim = ms.simulate(
        #    length=self._length,
        #    num_replicates=self.nsnps*100,  # 100X since some sims are empty
        #    mutation_rate=self._mut,
        #    migration_matrix=migmat,
        #    population_configurations=self._get_popconfig(),
        #    demographic_events=self._get_demography()
        #)
        ## build msprime simulation
        sim = ms.simulate(
            length=self._length,
            num_replicates=1,  # 100X since some sims are empty
            mutation_rate=self._mut,
            recombination_rate=self.recombination_rate,
            migration_matrix=migmat,
            population_configurations=self._get_popconfig(),
            demographic_events=self._get_demography()
        )
        return sim


    def run(self):
        """
        run and parse results for nsamples simulations.
        """
        ## iterate over ntests (different sampled simulation parameters)
        gidx = 0
        for ridx in range(self.ntests):
            ## run simulation for demography ridx
            ## yields a generator of trees to sample from with next()
            ## we select 1 SNP from each tree with shape (1, ntaxa)
            ## repeat until snparr is full with shape (nsnps, ntips)
            for rep in range(self.nreps):
                sims = self._simulate(ridx)

                ## store results (nsnps, ntips); def. 1000 SNPs
                snparr = np.zeros((self.nsnps, self.ntips), dtype=np.int32)

                ## continue until all SNPs are sampled from generator
                fidx = 0
                while fidx < self.nsnps:
                    ## get genotypes and convert to {0,1,2,3} under JC
                    bingenos = next(sims).genotype_matrix()

                    ## count as 16x16 matrix and store to snparr
                    if bingenos.size:
                        sitegenos = mutate_jc(bingenos, self.ntips)
                        snparr[fidx] = sitegenos
                        fidx += 1

                ## keep track for counts index
                quartidx = 0

                ## iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
                qiter = itt.combinations(range(self.ntips), 4)
                for currquart in qiter:
                    ## cols indices match tip labels b/c we named tips node.idx
                    quartsnps = snparr[:, currquart]
                    self.counts[gidx, quartidx] = count_matrix_float(quartsnps)
                    quartidx += 1

                # scale by max count for this rep
                self.counts[gidx, ...] /= self.counts[gidx, ...].max()
                gidx += 1



#############################################################################
class Simulator:
    """ 
    This is the object that runs on the engines by loading data from the HDF5,
    building the msprime simulations calls, and then calling .run() to fill
    count matrices and return them. 
    """
    def __init__(self, database, slice0, slice1, run=True, debug=False):

        ## debugging
        self._debug = debug
        
        ## location of data
        self.database = database
        self.slice0 = slice0
        self.slice1 = slice1

        ## parameter transformations
        self._mut = 1e-8
        self._recomb = 1e-9
        self._theta = None

        ## open view to the data
        with h5py.File(self.database, 'r') as io5:

            ## sliced data arrays
            self.thetas = io5["thetas"][slice0:slice1]
            self.atstarts = io5["admix_tstarts"][slice0:slice1, ...]
            self.atends = io5["admix_tends"][slice0:slice1, ...]   
            self.asources = io5["admix_sources"][slice0:slice1, ...]
            self.atargets = io5["admix_targets"][slice0:slice1, ...]
            self.aprops = io5["admix_props"][slice0:slice1, ...] 
            self.node_heights = io5["node_heights"][slice0:slice1, ...]

            ## attribute metadata
            self.tree = toytree.tree(io5.attrs["tree"])
            self.nsnps = io5.attrs["nsnps"]
            self.ntips = len(self.tree)
            self.aedges = self.asources.shape[1]

            ## storage for output
            self.nquarts = int(comb(N=self.ntips, k=4))  # scipy.special.comb
            self.nvalues = self.slice1 - self.slice0
            self.counts = np.zeros(
                (self.nvalues, self.nquarts, 16, 16), dtype=np.float32) 

            ## calls run and returns filled counts matrix
            if run:
            	print("supposedly running")
                self.run()


    @property
    def _Ne(self):
        "Ne is automatically calculated from theta and fixed mut"
        return (self._theta / self._mut) / 4.


    def _simulate(self, idx):
        """
        performs simulations with params varied across input values.
        """       
        # store _temp values for this idx simulation, 
        migmat = np.zeros((self.ntips, self.ntips), dtype=int).tolist()
        self._theta = self.thetas[idx]
        self._astarts = self.atstarts[idx]
        self._aends = self.atends[idx]
        self._aprops = self.aprops[idx]
        self._asources = self.asources[idx]
        self._atargets = self.atargets[idx]
        self._node_heights = self.node_heights[idx]

        ## build msprime simulation
        sim = ms.simulate(
            length=100000,                                      # optimize this
            num_replicates=1,  
            mutation_rate=self._mut,
            recombination_rate=self._recomb,
            migration_matrix=migmat,
            population_configurations=self._get_popconfig(),  # just theta
            demographic_events=self._get_demography()         # node heights
        )
        return sim


    def _get_demography(self):
        """
        returns demography scenario based on an input tree and admixture
        edge list with events in the format (source, dest, start, end, rate)
        """
        ## Define demographic events for msprime
        demog = set()

        ## append stored heights to tree nodes as metadata
        cidx = 0
        n_internal_nodes = sum(1 for i in self.tree.tree.traverse())
        for nidx in range(n_internal_nodes):
            node = self.tree.tree.search_nodes(idx=nidx)[0]
            if not node.is_leaf():
                node._height = self._node_heights[cidx]
                cidx += 1

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
                time = node._height * 2. * self._Ne
                demog.add(ms.MassMigration(time, source, dest))
                
                ## debug
                if self._debug:
                    print('demog div:', (int(time), source, dest), 
                        file=sys.stderr)

        ## Add migration edges
        for evt in range(self.aedges):
            rate = self._aprops[evt]
            start = self._astarts[evt] * 2. * self._Ne
            end = self._aends[evt] * 2. * self._Ne
            source = self._asources[evt]
            dest = self._atargets[evt]

            ## rename nodes at time of admix in case divergences renamed them
            snode = self.tree.tree.search_nodes(idx=source)[0]
            dnode = self.tree.tree.search_nodes(idx=dest)[0]
            children = (snode._schild, dnode._schild)
            demog.add(ms.MigrationRateChange(start, rate, children))
            demog.add(ms.MigrationRateChange(end, 0, children))

            ## debug
            if self._debug:
                print(
                    'demog mig:', 
                    (start, end),
                    round(rate, 4), 
                    children,
                    self._Ne,
                    file=sys.stderr,
                    )

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
            ms.PopulationConfiguration(sample_size=1, initial_size=self._Ne)
            for ntip in range(self.ntips)]
        return population_configurations        

    # have this write to a file?
    def run(self):
        """
        run and parse results for nsamples simulations.
        """
        ## iterate over ntests (different sampled simulation parameters)
        for idx in range(self.nvalues):
            sims = self._simulate(idx)

            ## get genotypes and convert to {0,1,2,3} under JC
            bingenos = next(sims).genotype_matrix()

            ## count as 16x16 matrix and store to snparr
            ## snparr is trimmed down to size = nsnps
            snparr = mutate_jc_fullmat(bingenos, self.ntips, self.nsnps)

            ## keep track for counts index
            quartidx = 0

            ## iterator for quartets, e.g., (0, 1, 2, 3), (0, 1, 2, 4)...
            qiter = itt.combinations(range(self.ntips), 4)
            for currquart in qiter:
                ## cols indices match tip labels b/c we named tips node.idx
                quartsnps = snparr[:, currquart]

                self.counts[idx, quartidx] = count_matrix_float(quartsnps)
                quartidx += 1

            # re-scale by max count for this rep
            self.counts[idx, ...] /= self.counts[idx, ...].max()




############################################################################
class DataBase:
    """
    An object to parallelize simulations over many parameter settings
    and store finished reps in a HDF5 database. The number of labeled tests
    is equal to nevents * ntrees * ntests * nreps, where nevents is based
    on the tree toplogy and number of admixture edges drawn on it (nedges). 

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

    edge_function: None or str (default=None)
        If an edge_function argument is entered then edge lengths on the 
        topology are drawn from one of the supported distributions. The 
        following options are available: 

           "node_slider": uniform jittering of input tree node heights.
           "poisson": exponentially distributed waiting times between nodes.

    nedges: int (default=0)
        The number of admixture edges to add to each tree at a time. All edges
        will be drawn on the tree that can connect any branches which overlap 
        for a nonzero amount of time. A set of admixture scenarios 
        generated by drawing nedges on a tree is referred to as nevents, and 
        all possible events will be tested. 
        * Each nedge increases nvalues by nevents * ntrees * ntests * nreps. 

    ntrees: int (default=100)
        The number of sampled trees to perform tests across. Sampled trees
        have the topology of the input tree but with branch lengths modified
        according to the function in 'edge_function'. If None then tests repeat
        using the same tree. 
        * Each ntree increases nvalues by ntests * nreps.

    ntests: int (default=100)
        The number of parameters to draw for each admixture_event described
        by an edge but sampling different durations, magnitudes, and mutation
        rates (theta). For example, (2, 1, None, None, None) could draw 
        (2, 1, 0.1, 0.3, 0.01) and theta=0.1 in one randomly drawn test, 
        and (2, 1, 0.2, 0.4, 0.02) and theta=0.2 in another. 
        * Each ntest increases nvalues by nreps. 

    nreps: int (default=10)
        The number of replicate simulations to run per admixture scenario, 
        sampled tree, and parameter set (nevent, ntree, ntest). Replicate 
        simulations make identical calls to msprime but get variable result
        matrices due to variability in the coalescent process.

    nsnps: int (default=1000)
        The number of SNPs in each simulation that are used to build the 
        16x16 arrays of phylogenetic invariants for each quartet sample. 

    theta: int or tuple (default=0.01)
        The mutation parameter (2*Ne*u), or range of values from which values
        will be uniformly drawn across ntests. 

    seed: int (default=123)
        Set the seed of the random number generator

    force: bool (default=False)
        Force overwrite of existing database file.
    """
    def __init__(
        self,
        name,
        workdir,
        tree,
        edge_function=None,
        nsnps=1000,
        nedges=0,
        ntrees=1,
        ntests=1,
        nreps=1,
        theta=0.01,
        seed=123,
        force=False,
        debug=False,
        quiet=False,
        **kwargs):

        ## database locations
        self.name = name
        self.workdir = (workdir or 
                        os.path.realpath(os.path.join('.', "databases")))
        self.database = os.path.realpath(
            os.path.join(workdir, self.name + ".hdf5"))
        self._checkpoint = None
        self._db = None  # open/closed file handle of self.database
        self._debug = debug
        self._quiet = quiet

        ## store params
        self.theta = theta
        self.tree = tree
        self.edge_function = (edge_function or {})

        ## database label combinations
        self.nedges = nedges
        self.ntrees = ntrees
        self.ntests = ntests        
        self.nreps = nreps
        self.nsnps = nsnps
        self.nstored_values = None

        ## decide on an appropriate chunksize to keep memory load reasonable
        self.chunksize = 1000

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
                ## exists and destroy it.
                answer = input(
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
        if not self._quiet:
            print("stored {} labels to {}"
                  .format(self.nstored_values, self.database))

        ## print info about the database in debug mode
        self._debug_report()

        ## Close the database. It is now ready to be filled with .run()
        ## which will run until all tests are finished in the database. We 
        ## could then return to this Model object and add more tests later 
        ## if we wanted by using ...<make this>
        self._db.close()


    ## not implemented yet.
    def _find_checkpoint(self):
        """
        find last filled database checkpoint, we should probably just store
        this value rather than need to calculate it. If we do calculate it, 
        then we should do it with dask, since this method uses a lot of RAM.
        """
        return np.argmin(
            np.all(np.sum(np.sum(self.counts, axis=1), axis=1) != 0, axis=1))


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
                yield self._node_slider()
            elif self.edge_function == "poisson":
                #raise NotImplementedError("Not yet supported")
                yield self._exp_sampler(self.tree, returndict=None)
            else:
                yield self.tree

    def _node_slider(self):
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
        ctree = copy.deepcopy(self.tree)
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

    def _exp_sampler(self, betaval=1, returndict="only"):
        """

        # NOT WORKING


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
        #ttree
        #tree = tree.tree
        tree = self.tree.copy()
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

    def _generate_fixed_tree_database(self):
        """
        Parses parameters in self.params to create all combinations
        of parameter values to test. Returns the number of the simulations.
        Simulation metadata is appended to datasets. 

        Expect that the h5 file self._db is open in w or a mode.
        """

        ## store the tree as newick with no bls, and using idx for name, 
        ## e.g., ((1,2),(3,4));
        storetree = self.tree.copy()
        for node in storetree.tree.traverse():
            node.name = node.idx
        self._db.attrs["tree"] = storetree.tree.write(format=9)
        self._db.attrs["nsnps"] = self.nsnps

        ## the number of data points will be nreps x the number of events
        ## uses scipy.special.comb
        admixedges = get_all_admix_edges(self.tree)
        nevents = int(comb(N=len(admixedges), k=self.nedges))
        nvalues = nevents * self.ntrees * self.ntests * self.nreps 
        nquarts = int(comb(N=len(self.tree), k=4))
        self.nstored_values = nvalues

        ## store count matrices
        self._db.create_dataset("counts", 
            shape=(nvalues, nquarts, 16, 16),
            dtype=np.float32)

        ## store node key, just once
        self._db.create_dataset("nodekey", 
            shape=(len([i.idx for i in self.tree.tree.search_nodes()])-len([i 
                for i in self.tree.tree.get_leaves()]),),
            dtype=np.uint32)   

        ## store node heights
        internal_nodes = sum(
            [1 for i in self.tree.tree.traverse() if not i.is_leaf()])
        self._db.create_dataset("node_heights",
            shape=(nvalues, internal_nodes),
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
        self._db.create_dataset("admix_tstarts", 
            shape=(nvalues, self.nedges),
            dtype=np.float64)
        self._db.create_dataset("admix_tends", 
            shape=(nvalues, self.nedges),
            dtype=np.float64)

        ## store parameters of the simulation
        self._db.create_dataset("thetas",
            shape=(nvalues,),
            dtype=np.float64)


    def _fill_fixed_tree_database_labels(self):
        """
        This iterates across generated trees and creates simulation scenarios
        for nreps iterations for each admixture edge(s) scenario in the tree
        and stores the full parameter information into the hdf5 database.
        """

        ## (1) ntrees: iterate over each sampled tree (itree)

        admixedges = get_all_admix_edges(self.tree)
        nevents = int(comb(N=len(admixedges), k=self.nedges))
        nvalstree = nevents * self.ntests * self.nreps 

        tidx = 0
        for _ in range(self.ntrees):
            ## sample tree and save new internal node heights in idx order
            itree = next(self.tree_generator)
            #print(itree)
            node_heights = [           
                itree.tree.search_nodes(idx=i)[0].height
                for i in range(sum(1 for i in itree.tree.traverse()))
                if not itree.tree.search_nodes(idx=i)[0].is_leaf()
            ]

            ## get all admixture edges that can be drawn on this tree
            admixedges = get_all_admix_edges(itree)

            ## (2) nevents: iterate over (source, target) items, or pairs or 
            ## triplets of items depending on nedges combinations.
            eidx = tidx*nvalstree
            


            events = itt.combinations(admixedges.items(), self.nedges)
            
            for evt in events:
                ## initalize a Model to sample range of parameters on this edge
                ## model counts array shape: (ntests*nreps, nquarts, 16, 16)
                admixlist = [(i[0][0], i[0][1], None, None, None) for i in evt]

                ## (3) ntests: sample duration, magnitude, and params on edges
                ## model .run() will make array (ntests, nquarts, 16, 16)
                model = Model(itree, 
                    admixture_edges=admixlist, 
                    ntests=self.ntests, 
                    theta=self.theta)

                ## (4) nreps: fill the same param values repeated times
                mdict = model.test_values

                nnn = self.ntests * self.nreps
                sta, end = eidx, eidx + nnn

                ## store node heights same for every test * rep
                self._db["node_heights"][sta:end, :] = node_heights

                ## store thetas same for every rep, but not test (0,0,0,1,1,1)
                thetas = _tile_reps(mdict["thetas"], self.nreps)
                self._db["thetas"][sta:end] = thetas

                ## get labels from admixlist and model.test_values
                for xidx in range(model.aedges):
                    sources = np.repeat(admixlist[xidx][0], nnn)
                    targets = np.repeat(admixlist[xidx][1], nnn)
                    mrates = _tile_reps(mdict[xidx]["mrates"], self.nreps)
                    msta = _tile_reps(mdict[xidx]["mtimes"][:, 0], self.nreps)
                    mend = _tile_reps(mdict[xidx]["mtimes"][:, 1], self.nreps)                    

                    ## store labels for this admix event (nevents x nreps)
                    self._db["admix_sources"][sta:end, xidx] = sources
                    self._db["admix_targets"][sta:end, xidx] = targets
                    self._db["admix_props"][sta:end, xidx] = mrates
                    self._db["admix_tstarts"][sta:end, xidx] = msta
                    self._db["admix_tends"][sta:end, xidx] = mend

                eidx += nnn
            tidx += 1


    def _fill_fixed_tree_database_counts(self, ipyclient):
        """
        Sends jobs to parallel engines to run Simulator.run().
        """

        ## load-balancer for single-threaded execution jobs
        lbview = ipyclient.load_balanced_view()

        ## an iterator to return chunked slices of jobs
        jobs = range(self.checkpoint, self.nstored_values, self.chunksize)
        #print(jobs)
        njobs = int((self.nstored_values - self.checkpoint) / self.chunksize)
        #print(njobs)
        ## start progress bar
        start = time.time()

        ## submit jobs to engines
        print("submitting jobs")
        asyncs = {}
        #for job in jobs:
        #    args = (self.database, job, job + self.chunksize)
        #    asyncs[job] = lbview.apply(Simulator, *args)

        num_engines = len(self._ipcluster["pids"])
        #rounds = float(njobs)/float(num_engines) 
        rounds = int(njobs / num_engines) + (njobs % num_engines > 0) # round up
        #print(rounds)
        for currround in range(rounds):
            #print(currround)
            for roundnum in range(num_engines):
            	if (currround*num_engines + roundnum < njobs):
	                job = jobs[currround*num_engines + roundnum]
	                args = (self.database, job, job + self.chunksize)
	                asyncs[job] = lbview.apply(Simulator, *args)

            ## wait for jobs to finish, catch results as they return and enter 
            ## them into the HDF5 database. This keeps memory low.
            done = self.checkpoint
            while 1:
                ## gather finished jobs
                finished = [i for i, j in asyncs.items() if j.ready()]
                ## iterate over finished list and insert results
                if (len(finished) == len(asyncs)):
                    for job in finished:
                        async = asyncs[job]
                        if async.successful():
                            ## store result
                            done += 1
                            result = async.result().counts
                            with h5py.File(self.database, 'r+') as io5:
                                io5["counts"][job:(job + self.chunksize)] = result
                            ## free up memory from job
                            del asyncs[job]

                        else:
                            raise SimcatError(async.result())

                ## print progress
                self._progress_bar(njobs, done, start, "simulating count matrices")
                ## finished: break loop
                if len(asyncs) == 0:
                    break
                else:
                    time.sleep(0.5)
            print("Done with round: " + str(currround) + " of " + str(rounds))


    ## THE MAIN RUN COMMANDS ----------------------------------------
    def run(self, ipyclient=None, quiet=False):
        """
        Distributes parallel jobs and wraps functions for convenient cleanup.
        Fills all count matrices with simulated data. 

        Parameters
        ----------
        ipyclient (ipyparallel.Client object):
            A connected ipyclient object. If ipcluster instance is 
            not running on the default profile then ...
        """

        ## wrap the run in a try statement to ensure we properly shutdown
        ## and cleanup on exit or interrupt. 
        inst = None
        try:
            ## find and connect to an ipcluster instance given the information
            ## in the _ipcluster dictionary if a connected client was not given
            if not ipyclient:
                ipyclient = ipp.Client()

            ## print the cluster connection information
            if not quiet:
                cluster_info(ipyclient)

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

            ## put checkpointing code here...
            self.checkpoint = 0

            ## execute here...
            self._fill_fixed_tree_database_counts(ipyclient)


        ## handle exceptions so they will be raised after we clean up below
        except KeyboardInterrupt as inst:
            print("\nKeyboard Interrupt by user. Cleaning up...", 
                file=sys.stderr)

        except Exception as inst:
            print("\nUnknown exception encountered: {}".format(inst), 
                file=sys.stderr)

        ## close client when done or interrupted
        finally:
            try:
                ## can't close client if it was never open
                if ipyclient:

                    ## send SIGINT (2) to all engines
                    ipyclient.abort()
                    time.sleep(1)
                    for engine_id, pid in self._ipcluster["pids"].items():
                        if ipyclient.queue_status()[engine_id]["tasks"]:
                            os.kill(pid, 2)
                        time.sleep(0.25)
                    ipyclient.purge_everything()

            ## if exception during shutdown then we really screwed up
            except Exception as inst2:
                print("warning: error during shutdown:\n{}".format(inst2))



    @staticmethod
    def _progress_bar(njobs, nfinished, start, message=""):

        ## measure progress
        if njobs:
            progress = 100 * (nfinished / njobs)
        else:
            progress = 100

        ## build the bar
        hashes = "#" * int(progress / 5.)
        nohash = " " * int(20 - len(hashes))

        ## get time stamp
        elapsed = datetime.timedelta(seconds=int(time.time() - start))

        ## print to stderr
        args = [hashes + nohash, int(progress), elapsed, message]
        print("\r[{}] {:>3}% | {} | {}".format(*args), end="", file=sys.stderr)
        sys.stderr.flush()




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
            if not any([snode.is_root(), dnode.is_root(), dnode == snode]):
                ## check for overlap
                smin, smax = snode.interval
                dmin, dmax = dnode.interval

                ## find if nodes have interval where admixture can occur
                low_bin = np.max([smin, dmin])
                top_bin = np.min([smax, dmax])
                if top_bin > low_bin:
                    intervals[(snode.idx, dnode.idx)] = (low_bin, top_bin)
    return intervals


def _tile_reps(array, nreps):
    "used to fill labels in the database for replicates"
    ts = array.size
    nr = nreps
    result = np.array(
        np.tile(array, nr)
        .reshape((nr, ts))
        .T.flatten())
    return result


############################################################################
## jitted functions for running super fast -----------------
@numba.jit(nopython=True)
def count_matrix_int(quartsnps):
    """
    return a 16x16 matrix of site counts from snparr
    """
    arr = np.zeros((16, 16), dtype=np.uint64)
    add = np.uint64(1) 
    for idx in range(quartsnps.shape[0]):
        i = quartsnps[idx, :]
        arr[(4 * i[0]) + i[1], (4 * i[2]) + i[3]] += add    
    return arr


@numba.jit(nopython=True)
def count_matrix_float(quartsnps):
    """
    return a 16x16 matrix of site counts from snparr
    """
    arr = np.zeros((16, 16), dtype=np.float32)
    add = np.float32(1)
    for idx in range(quartsnps.shape[0]):
        i = quartsnps[idx, :]
        arr[(4 * i[0]) + i[1], (4 * i[2]) + i[3]] += add    
    return arr  # / arr.max()    


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
    # return dtypes must match
    return np.zeros(0, dtype=np.int64)  

@numba.jit(nopython=True)
def mutate_jc_fullmat(geno, ntips, nsnps):
    """
    mutates sites with 1 into a new base in {0, 1, 2, 3}
    """
    snps=np.zeros((nsnps,ntips), dtype=np.int32)
    allbases = np.array([0, 1, 2, 3])
    indices = np.random.choice(geno.shape[0],size=nsnps,replace=False)
    for ridx in np.arange(nsnps):
        snp = geno[indices[ridx]]
        if snp.sum():
            init = np.zeros(ntips, dtype=np.int64)
            init.fill(np.random.choice(allbases))
            notinit = np.random.choice(allbases[allbases != init[0]])
            init[snp.astype(np.bool_)] = notinit
            snps[ridx] = init
    return snps

