#!/usr/bin/env bash
set -e
which python
conda install --yes --file requirements/requirements.txt -c conda-forge
conda install --yes cudatoolkit=10.2 -c pytorch
conda install --yes --file requirements/conda_torch_cuda.txt -c pytorch
pip install -r requirements/only_pip_requirements.txt

pre-commit install
