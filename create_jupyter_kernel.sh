#!/bin/bash

conda env create -f environment.yml --prefix ./environment
conda activate ./environment
python -m ipykernel install --user --name nonstationary_mra