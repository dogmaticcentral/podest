# Installing Podest from pip

```bash
pip install podest
```

# Installing Podest from the source repo

## Installing poetry
You can create a `.venv` directory using Python's built-in `venv` module, then install Poetry inside that environment and use it as Poetry's in-project virtual environment directory.

Here’s how:

1. Create the virtual environment in your project directory:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

- On Linux/macOS:

```bash
source .venv/bin/activate
```

- On Windows:

```powershell
.venv\Scripts\activate
```

3. With the virtual environment active, install Poetry into this environment:

```bash
pip install poetry
```

4. Now when you run Poetry commands with the `.venv` activated environment, Poetry can use this `.venv` as the virtual environment for the project.

This essentially means you have manually created the virtual environment that Poetry would normally create itself. If you configure Poetry with:

```bash
poetry config virtualenvs.in-project true
```

It normally creates `.venv` automatically, but if you prefer manual control or want to use an already existing `.venv`, this approach works well.

Check with:
```bash
poetry --version
```
To find the exact path of the virtual environment Poetry is using for your project, you can run:
```bash
poetry  env info --path
```

## Installing podest using poetry
```bash
poetry  install
```
