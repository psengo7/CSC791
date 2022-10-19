#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 10:00:00
#SBATCH --gres=gpu:1

source ~/.bashrc

#TODO: PUT ANYTHING HERE
#conda activate csc791

cd /jet/home/psengo/csc791/P3

python3 modelHPO.py
