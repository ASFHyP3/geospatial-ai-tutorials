# TerraMind Environment Creation

Make sure to follow the instructions for [initial setup](inital_setup.md) (particularly the conda setup) before following this tutorial.

NAS does not have an off-the-shelf conda environment that we can use for running TerraMind, so we will need to create our own.

1. Load the terramind `environment.yml` file in this directory into your home PFE directory
1. Call `conda env create --prefix $CONDA_INSTALL/terramind --file environment.yml` to create the base environment
1. Activate the environment `source activate terramind`
1. Install terratorch `python -m pip install git+https://github.com/IBM/terratorch.git`
1. Install diffusers `python -m pip install diffusers==0.30.0`

In order, the commands are:
```bash
conda env create --prefix $CONDA_INSTALL/terramind --file environment.yml
source activate terramind
python -m pip install git+https://github.com/IBM/terratorch.git
python -m pip install diffusers==0.30.0
```
