### Git commands
#### Basic Git commands
```bash
# Initialize a new Git repo
git init

# Clone an existing repo
git clone <repo-url>

# Check current branch and changes
git status

# List branches
git branch

# Create a new branch
git branch <name>

# Switch to a branch
git checkout <name>


#merge another branch into current one
git merge <branch>

```

#### Cleanup
```bash 
# Discard all changes (DANGEROUS)
git reset --hard

# Delete untracked files/directories
git clean -fd
```



#### Merge main into dev
```bash

# Step 1: Checkout to dev branch
git checkout dev

# Step 2: Pull the latest changes from origin
git pull origin dev

# Step 3: Merge main into dev
git merge main

# Step 4 (optional): Push the merged changes to remote dev branch
git push origin dev
```


