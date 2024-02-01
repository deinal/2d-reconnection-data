#!/bin/bash
#SBATCH -M vorna
#SBATCH -p test
#SBATCH -c 1
#SBATCH -t 00:05:00
#SBATCH --mem-per-cpu=4G
#SBATCH -o slurm/%j.out


module load SciPy-bundle
export PYTHONPATH=$PYTHONPATH:$HOME/analysator

python flux_function.py