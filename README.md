[![Tests](https://github.com/ml4physics/cyjax/actions/workflows/python-pytest.yml/badge.svg)](https://github.com/ml4physics/cyjax/actions/workflows/python-pytest.yml)
[![Documentation Status](https://readthedocs.org/projects/cyjax/badge/?version=latest)](https://cyjax.readthedocs.io/en/latest/?badge=latest)

# Calabi-Yau metrics with JAX
**CYJAX** is a python library for numerically approximating Calabi-Yau metrics
using machine learning implemented with the [JAX library](https://github.com/google/jax).
It is meant to be accessible both on a top-level and in a modular way, exposing all intermediate lower-level functionality.
The principle application is machine learning for the algebraic ansatz of the Kähler potential from [Donaldson's algorithm](https://doi.org/10.4310/jdg/1090349449).
While this ansatz is more restrictive compared to approximating the metric directly, it automatically satisfies Kählerity and compatibility on patch overlaps.
As of now, this implementation is limited to varieties defined by a single defining equation on one complex projective space.
A generalization to a wider class of cases is planned.

Good places to start are the introduction and the tutorial notebooks on the documentation page.
More background can also be found in this [paper](https://inspirehep.net/literature/1835403).
Some background knowledge on [JAX](https://github.com/google/jax) may be useful.
The networks implemented here use [Flax](https://github.com/google/flax), but any machine learning framework that works with JAX can be used.

Configurable examples of training for the Dwork family and a two-parameter quintic can be found in the `scripts/` folder.
The configuration files use [hydra](https://hydra.cc) and can be used to change the training target, the training routine, and the network architecture.
While the code runs on both CPU and GPU (if JAX is installed to support this), the latter is advisable for large models.

# Installation

You may want to install the code in a new virtual environment.
This can be created using `python -m venv cyjax-env` and activated using `source cyjax-env/bin/activate` from within the terminal at a desired working directory.

Note: If a specific version of [JAX](https://github.com/google/jax) is required, e.g. with GPU support, next follow the instructions [here](https://github.com/google/jax#installation).
Otherwise, by default the CPU version of JAX will be installed.

Installation is easiest using [pip](https://packaging.python.org/en/latest/key_projects/#pip).
First, update it by running: `pip install --upgrade pip setuptools`.
Then, depending on whether the code is intended to be modified for development, do either of the following:
- Simply run `pip install git+https://github.com/ml4physics/cyjax.git`.
- Download (or clone) the repository. 
  Open the root folder of the repository in a terminal and then run `pip install -e .`. 
  The `-e` flag will allow you to modify the code without having to reinstall it. 

To run the scripts, additionally [hydra](https://hydra.cc) is required (`pip install --upgrade hydra-core`).

## Requirements

Currently, the code works with python versions greater or equal version `3.7`, both with or without JAX GPU support.

The required packages, which are listed in `setup.py`, are automatically installed with the above installation process.
Otherwise, they have to be installed manually.

## Building the documentation
This project uses [sphinx](https://www.sphinx-doc.org) to generate the documentation.
Following the installation of the `cyjax` package, install the additional requirements in `docs/requirements.txt`
(`pip install -r requirements-docs.txt`).
Inside the `docs` folder, run `make html` to generate the html version of the documentation, which can afterwards
be found in `docs/build/html`.

# Related work
Another library with a similar goal as this one is [CYMetric](https://github.com/pythoncymetric/cymetric) .
In contrast to CYMetric, here the algebraic ansatz for the Kähler potential from [Donaldson's algorithm](https://doi.org/10.4310/jdg/1090349449) is used.
While this ansatz is more restrictive, it automatically satisfies Kählerity and compatibility on patch overlaps.

# Contributions
Contributions are welcome, please open a pull request or get in touch!

# Reference
If you find this work useful, please cite

```
@article{gerdes2022cyjax,
    title = "{CYJAX: A package for Calabi-Yau metrics with JAX}",
    author = "Gerdes, Mathis and Krippendorf, Sven",
    eprint = "2211.12520",
    archivePrefix = "arXiv",
    primaryClass = "hep-th",
    doi = "10.1088/2632-2153/acdc84",
    journal = "Mach. Learn. Sci. Tech.",
    volume = "4",
    number = "2",
    pages = "025031",
    year = "2023"
}
```
