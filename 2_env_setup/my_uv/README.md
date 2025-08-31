```bash

# 1. init project
uv init

# 2. Add dependencies
uv add requests pandas


# 3. uv run script
uv run python my_script.py



# 5. Compile a lockfile for reproducibility
uv lock

# 6. Install packages from a lockfile (e.g., on another machine)
uv sync

# 7. List installed packages
uv pip list



```