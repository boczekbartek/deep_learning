#!/usr/bin/env bash
set -e

conda install --yes --file requirements/requirements.txt
conda install --yes --file requirements/conda_torch.txt -c pytorch
pip install -r requirements/only_pip_requirements.txt

pre-commit install
