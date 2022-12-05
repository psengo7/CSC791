# CSC791
All unit projects for this class are listed below: 
* P1: gives accuracy and inference time of pruned MNIST pretrained model for different pruning methods in nni library.
* P2: gives accuracy and inference time of quantized MNIST pretrained model for different quantization methods in nni library.
* P3: gives hyper parameter optimizations for learning rate and batch size using different HPO tuners. 
* P4: gives accuracy and inference time of pruned + distilled Mnist pre-trained model and also gives inference time for tvm optimized model
* CourseProject: Optimize a superresolution model with nni


# Common bridges cluster commands
* Allocate node 
    * salloc -N 1 -p GPU-shared --gres=gpu:1 -q interactive -t 01:00:00
* See if you have open node
    * squeue | grep $USER
* cancel node
    * scancel <slurm-id>  
* send batch job
    * sbatch job-proj -t 10:00:00