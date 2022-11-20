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
cd /jet/home/psengo/CSC791/P4

#run program and get pytorch base model, pruned model, and distilled model as pytorch pt files
python modelDistillation.py main

#create tvm_model folder
mkdir tvm_files

#get test image  
python modelDistillation.py image

#convert distilled pt model to onnx
python modelDistillation.py toOnnx

#tune distilled model
tvmc tune --target "llvm" --output ./tvm_files/mnist_distill_autotune.json ./pre-trained_model/mnist_distill.onnx

#compile with tuned model
tvmc compile --target "llvm" --tuning-records ./tvm_files/mnist_distill_autotune.json --output ./tvm_files/mnist_autotuned.tar ./pre-trained_model/mnist_distill.onnx

# get inference time of tuned model
tvmc run --inputs ./tvm_files/image.npz --output ./tvm_files/predictions.npz --print-time --repeat 100 ./tvm_files/mnist_autotuned.tar
