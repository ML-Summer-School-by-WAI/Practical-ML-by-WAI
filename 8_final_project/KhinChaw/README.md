# 1. Initialize Pipenv with Python 3.11

pipenv --python 3.11

# 2. Install main packages

pipenv install opencv-python scipy matplotlib ipython ipykernel

# 3. Install TensorFlow Metal for Mac GPU

pipenv install tensorflow-macos tensorflow-metal --skip-lock

# 4. Activate the virtual environment

pipenv shell

# 5. Install Jupyter in the environment

pipenv install jupyter
