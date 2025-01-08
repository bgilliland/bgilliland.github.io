---
title: "Production Code Setup for Machine Learning Models"
excerpt: "An approach to an OOP-style ML Model Workflow with `uv`, `tox`,"
date: 2025-01-08
---

# Production ML Repository Approach
Understanding how to productionalize your own code as a data scientist is an important skill. We are expected to not only understand the statistics and math behind the models we build, but we should be ready to collaborate with data engineers and software engineers to make these models more useful than a one-legged kick boxer. It does no good to let a Jupyter Notebook sit on the shelf with dust collecting. And it slows down time to production if we just hand them a messy notebook that they have to completely refactor (especially if they are not familiar with the model).

To see an example repository with some pre-written code that would be used for a production ML job go to [this repo.](https://github.com/bgilliland/ml-template)

This document will show off a couple ways a data scientist can be ready to productionalize their own code after they've trained their model. Some important concepts include:
* `uv` for package management
* `tox` for more efficient testing and environment management
* `config.yml` for appropriate configuration setting
* `pytest` for robust testing

This is by no means comprehensive. There are many approaches and I am positive that this one could be enhanced several times over. But it is a good start especially if you, like me, do not have a background in engineering and would like somewhere to start.

## `uv` for Package Management
[`uv`](https://docs.astral.sh/uv/) is 'an extremely fast Python package and project manager, written in Rust.' You may be familiar with `poetry` or `pyenv` or `pipenv`. This is functionally very similar but significantly faster due to having the `Rust` backend. It still uses the familiar lockfile structure so it should be a seamless transition if you're using another tool. My workflow is much faster as a result, and this will matter a lot when you're using `tox` (which we will discuss shortly) since anytime you go to test an environment it needs to create it from scratch which includes installing the packages. If you're using `pip` for this, which is the default in `tox`, then it can take a long time depending on your dependencies (ha!). 

Here is a quick example of how you would use it after installing it to your system:

This will create a directory at `myproj` containing a `uv` venv.
```console
uv init myproj
```

Once you're in that environment you can add `Python` libraries to it:
```console
uv add pandas numpy 
```

This will creae a `pyproject.toml` containing the library names and their versions (analogous to a `Pipfile` for `pipenv`) as seen below. Use `uv sync` to install the packages, which will be kept in a `uv.lock` file and this is what will reside in your repository for anyone who needs to run your code. 

Notice that there are some other attributes and descriptions that can be included, one of which is the python version requirement. Additionally, you can have multiple dependency groups which is very useful in testing and splitting your prod and dev environments. In this example, I have 4 total. `dependcies` is the production environment. I only need these ones to go-live. I also have the `dev` dependency group which is what I need in order to do research while training and testing in model development (usually in a Jupyter Notebook.) The last two are for running tests and checking code formatting for production.
```
[project]
name = "forest-covertypes"
version = "0.1.0"
description = "Machine Learning Production Template"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
    "feature-engine>=1.8.2",
    "numpy>=2.0.2",
    "pandas>=2.2.3",
    "pydantic>=2.10.4",
    "scikit-learn>=1.5.2",
    "setuptools>=75.7.0",
    "strictyaml>=1.7.3",
]

[dependency-groups]
dev = [
    "ipykernel>=6.29.5",
    "seaborn>=0.13.2",
    "matplotlib>=3.9.2",

]
testing = [
    "pytest>=8.3.4",
]
typing = [
    "black>=24.10.0",
    "flake8>=7.1.1",
    "isort>=5.13.2",
    "mypy>=1.14.1",
]
```

To activate the virtual environment for when you need to use that specific context such as running `.py` scripts then you can simply run:
```console
uv venv
```