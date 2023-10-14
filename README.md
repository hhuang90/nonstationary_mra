# The repo of nonstationary_mra
Implementation of parallel nonstationary MRA with a minimum reproducible example

# To run the nonstationary MRA C++ code
1. Adjust the values in *Makefile* to provide the paths for the C++ compiler and MPI, OpenMP, MKL, and Armadillo libraries
2. Adjust the values in *user_parameters* when needed
3. Run `make` to generate the executable *MRA*
4. Run **MRA** with MPI

# To explore the data and results for the the minimum reproducible example provided (using Python)
1. Create the environment and install the Jupyter kernel by running `bash create_jupyter_kernel.sh`
2. Open the notebook *minimum_reproducible_example.ipynb* using the created kernel *nonstationary_mra*