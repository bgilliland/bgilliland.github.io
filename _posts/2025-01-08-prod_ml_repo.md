---
title: "Package Management and Efficient Testing for Machine Learning Models"
excerpt: "An approach to an OOP-style ML Model Workflow with `uv` and `tox`"
date: 2025-01-08
---

## Production ML Repository Approach
Understanding how to productionalize your own code using OOP design as a data scientist is an important skill. We are expected to not only understand the statistics and math behind the models we build, but we should be ready to collaborate with data engineers and software engineers to make these models useful to the business. It does no good to let a Jupyter Notebook sit on the shelf with dust collecting. And it slows down time to production if we just hand them a messy notebook that they have to completely refactor (especially if they are not familiar with the model).

To see an example repository with some pre-written code that could be used for a production ML job go to this [template repo](https://github.com/bgilliland/ml-template) on my GitHub. This does not include Docker files or anything part of the actual deployment process, I will do another post regarding that. This article is meant to discuss code structure and some tools that make testing and package management simpler.

This document will show off a couple ways a data scientist can be ready to productionalize their own code after they've trained their model. Some important concepts include:
* `uv` for package management
* `tox` for efficient testing and environment management

This is by no means comprehensive. There are many approaches and I am positive that this one could be enhanced several times over. But it is a good start especially if you, like me, do not have a background in engineering and would like somewhere to start.

## `uv` for Package Management
Before you even begin training your model you need to ensure that you're working in an environment with exactly the dependencies that you need. This means you cannot be relying on your global installation of Python nor its libraries. Each project is unique in its needs so using virtual environments is critical to reproducibility. Enter `uv`. 

[`uv`](https://docs.astral.sh/uv/) is 'an extremely fast Python package and project manager, written in Rust.' You may be familiar with `poetry` or `pyenv` or `pipenv`. This is functionally very similar but significantly faster due to having the `Rust` backend. It still uses the familiar lockfile structure so it should be a seamless transition if you're using another tool. My workflow is much faster as a result, and this will matter a lot when you're using `tox` (which we will discuss shortly) since anytime you go to test an environment `tox` needs to create it from scratch which includes installing the packages. If you're using `pip` for this, which is the default in `tox`, then it can take a long time depending on your dependencies. 

Here is a quick example of how you would use it after installing it to your system:

`init` will create a directory at `myproj` containing a `uv` venv.
```console
uv init myproj
```

Once you're in that environment you can `add` Python libraries to it:
```console
uv add pandas numpy 
```

This will create a `pyproject.toml` containing the library names and their versions (analogous to a `Pipfile` for `pipenv`) as seen below and it will install those libraries to the `uv.lock` file. If you are pulling someone else's repository and you need to use their exact dependencies then use `uv sync` to install the packages as defined in their `uv.lock`.

Notice that there are some other attributes and descriptions that can be included, one of which is the python version requirement in `requires-python`. This can be changed to whatever your project requires. Additionally, you can have multiple `dependency-groups` which is very useful in testing and splitting your prod and dev environments. In this example, I have 4 total. `dependcies` is the production environment. I only need these ones to go-live. I also have the `dev` dependency group which is what I need in order to do research while training and testing in model development (usually in a Jupyter Notebook.) The last two are for running tests and checking code formatting for production.
```
[project]
name = "uv-example"
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

## `tox` for Testing and Environment Management
`tox` is a great partner to `uv`. They work together seamlessly to use `uv`'s package management capabilities as part of `tox`'s testing environment. The point of `tox` is to basically answer the question "What happens if this job is run in an environment with a different set of dependencies?" Those dependencies include various Python library versions or even different versions of Python itself! Additionally, it allows you as the engineer to run unit tests with specific commands to check an array of various aspects of the codebase including functionalities and formatting.

`tox` references a `tox.toml` document (or alternatively a `tox.ini` but they suggest using TOML for less advanced use-cases) to see what environments you want created when you go to test and what their configurations are. You can see in the below snippet at the top of the file we set some global configurations including what version of `tox` we require, what dependencies this project has (for us we need both the `tox-uv` plugin (discussed shortly) and `uv` itself). We also list the names of the environments that we want tested whenever we agnostically run the job (ie we don't need to specify these environments, it is implied we want them run). 

Notice the commands for the two environments we want to create, `test` and `checks`. These will bring in the packages from the specified `dependency_group`'s from the `pyproject.toml` file that `uv` references and then run a series of commands that you specify. In the `test` environment it will run the `pytest`'s which is all of the various unit tests you've designed in your code base to ensure the model is working as expected. `tox` will create that virtual environment from scratch to perform all of the installations of dependencies from scratch and run the code, mimicking what it would be like if a stranger were to run it on their machine for the first time. In the `checks` environment it will do the same but this time instead of unit testing it will just perform formatting 
```
requires = ["tox>=4.23.2"]
deps = ["tox-uv", "uv"]
env_list = ["test", "checks"]

[env_run_base]

[env.test]
description = "Run test under {base_python}"
with_dev = true
dependency_groups = ["testing"]
commands = [["pytest"]]

[env.checks]
description = "Run checks under {base_python}"
with_dev = "{[test]with_dev}"
dependency_groups = ["typing"]
commands = [["black", "."]]
```

### Using `uv` with `tox`
Out-of-the-box, `tox` does not use `uv`, it uses `pip`. But, there is a plugin just released in late 2024 that makes it possible! The repository for the plug-in is [here](https://github.com/tox-dev/tox-uv). Reference the documentation for installation.