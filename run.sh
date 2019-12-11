#!/bin/sh

#SBATCH -N 2 
export http_proxy="http://proxy.nyit.edu:80"
export https_proxy="http://proxy.nyit.edu:80"
python3 train_models.py
