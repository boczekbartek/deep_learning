#!/usr/bin/env bash
set -e

pip install -r requirements/requirements.txt
pip install -r requirements/only_pip_requirements.txt
pip install -r requirements/pip_torch.txt

pre-commit install
