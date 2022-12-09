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

#change directory to repo
cd /jet/home/psengo/CSC791

#install requirements 
pip install -r requirements.txt

#change directory to current project
cd /jet/home/psengo/CSC791/CourseProject

#run program and get pytorch base model, pruned model, and distilled model as pytorch pt files
python optimize.py
