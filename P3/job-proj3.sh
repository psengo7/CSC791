#!/bin/bash
#sbatch -N 1
#SBATCH -n 1
#SBATCH -p GPU-shared
#SBATCH -t 0:15:00
#SBATCH --gres=gpu:1

source ~/.bashrc

#create new conda environment
module load anaconda3

#activate environment
conda activate csc791

#change directory to current project
cd /jet/home/psengo/CSC791/P3

#run program
python modelHPO.py
