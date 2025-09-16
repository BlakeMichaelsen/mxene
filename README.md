# mxene project
Analyzing how entropy-driven aggregation drives phase separation in mxene-polymer systems

## ⚙️ Installation

Run the following in your terminal:

# bash/zsh
git clone git@github.com:eridanrojas/mxene.git

# then navigate into mxene
cd mxene

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate flowermd-dev

# Launch Jupyter
jupyter notebook


Then, in order to run a small scale simulation, navigate to the example/ directory:

- eric_sim.ipynb = Original eric simulation, we don't use this currently
- sim_no_shrink.ipynb = Our simulation without a shrink step written; will not wrk with dense systems
- sim_with_shrink.ipynb = current working simulation you should use, shrink step is included allowing system to be packed loosely then shrunk to be dense, causing more interactions to occur
- sim_with_randomwalk_chains.ipynb = a future notebook, has a shrink step but also manipulates the geometry of the chains is randomly set so that we may set a dihedral to be something other than zero
- TPSvsN.ipynb = a notebook to graph TPS v. N

To analyze results, navigate to working_example/:

- RDF_ACF_plots.ipynb = we use the Radial distribution function to measure aggregation over time and autocorrelation plots to ensure there are enough independent samples in a given sim
- clustering_plots.ipynb = we use this to determine the magnitude of clustering over time in a simulation. this means both size and amount of clusters, also using independent samples

In order to run a large scale simulation, navigate to signac/:

- submit.sh = running sbatch submit.sh will run sim_with_shrink.py as it currently is, with the configurations present in this file.
- sim_with_shrink.py = simulation file specifically made for running on cluster
