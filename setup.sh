#!/bin/bash

# Exit if any command fails
#set -e
#deactivate

# Create virtual environment
python -m venv .scratch_venv

# Activate the virtual environment
source .scratch_venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Add Jupyter kernel
# python -m ipykernel install --user --name=.scratch_venv --display-name "Python (.scratch_venv)"

#nbstripout --install

# To run, use - source setup.sh - in the command terminal