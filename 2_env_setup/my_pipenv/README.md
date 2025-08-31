```bash
# 1. Install pipenv globally
pip install pipenv

# 2. Navigate to your project directory
cd my_pipenv

which python
pipenv install --python Your-path

# 3. Initialize pipenv (creates Pipfile) Not Recommending to use
pipenv --python 3.9

# 4. Install a package (creates virtual environment if it doesn't exist)
pipenv run python hello.py

# install prod packages
pipenv install requests

# 5. Run a script within the pipenv environment
pipenv run python my_script.py

# 6. Spawn a shell within the pipenv environment
pipenv shell

# 7. Exit the pipenv shell
exit

# check py env location without going under the pipenv 
pipenv --py

# install dev packages : only for mac user
pipenv install  --dev tensorflow-macos


```