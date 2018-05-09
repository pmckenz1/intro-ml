
# intro-ml 


### Simulation and ML model fitting for admixture inference. 
*In Development*


#### Installation
```
## clone the repo
git clone [...]

## cd into repo dir
cd intro-ml

## local development install with pip
pip install -e .
```

#### Usage
See notebooks/ directory.

In particular, `final_workflow.ipynb` demonstrates an example workflow.

#### simcat Model class object
Accepts toytree object and simulation parameters. Produces a stack of site count matrices corresponding to SNP distributions for quartets sampled from the tree.

#### simcat Database class object
Accepts toytree topology and, when run, fills an HDF5 database with a designated simulation regime.