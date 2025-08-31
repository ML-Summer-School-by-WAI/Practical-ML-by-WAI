# 1. Create a new conda environment
conda create --name my_conda_env python=3.9

# 2. Activate the conda environment
conda activate my_conda_env

# 3. Install packages (can use conda or pip)
conda install numpy scikit-learn
pip install matplotlib # You can still use pip within a conda env

# 4. List installed packages
conda list

# 5. Deactivate the environment
conda deactivate


