# Inverting knowledge graph mappings

A master thesis by Tijs Van Kampen

## Getting started

Clone the repository with the --recurse-submodules flag to clone the required submodules as well.

```bash
git clone --recurse-submodules https://github.com/TijsVK/Knowledge-graphs-inversion
```

It is recommended to use a virtual environment to install the required packages. Preferences as to what to use vary, so no exact instructions or environments besides a requirements.txt file in /Implementation are provided.

Install the base requirements.

```bash
pip install -r ./Implementation/requirements.txt
```

Build the pyrdf4j package and install it.

```bash
python -m build ./Implementation/pyrdf4j 
pip install ./Implementation/pyrdf4j/dist/pyrdf4j-0.1.0-py3-none-any.whl
```

