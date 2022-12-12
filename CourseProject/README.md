# OVERVIEW: 
This project uses NNI Optimizations on LorngTang's SRCNN model(https://github.com/Lornatang/SRCNN-PyTorch) to try to achieve greater speedup and compression results. 

# DIRECTORY STRUCTURE:
* Results **(1)**
    * Inference_Image **(2)**
    * Optimized_models **(3)**
        * Onnx
        * Pytorch
    * PSNR **(4)**
    * XGen_Results **(5)**
* SRCNN **(6)**
* optimize_param.py **(7)**
* optimize.py **(8)**

1. Gives a folder containing all the results for each optimized model
2. Gives the output of the low resolution image when passed in each optimized model
3. Gives the optimized model exported as an onnx file and a pth.tar file
4. Gives a text file with the PSNR results for all the optimized models
5. Gives the csv file for the XGen results of all the optimized models
6. Gives the SRCNN original github model from Lorngtang'g github
7. Holds the configurations and optimizations to apply to the SRCNN model
8. Performs the optimizations specified in optimize_param.py on the SRCNN model.

# HOW TO RUN PROGRAM:
1. Login to bridges and git clone current repository into user folder.  
2. (OPTIONAL) Delete all files in Results/ (don't delete folders keep directory structure same) except the following:    
    * In Results/Optimized_models/Pytorch/ keep "pretrained.pth.tar" file
    * In Results/Inference_Image keep "butterfly_lr.png" file
5. load anaconda: "module load anaconda3"
6. create conda environment : "conda create --name courseProj"
7. activate conda environment: "conda activate courseProj"
7. install requirements: "pip install -r requirements.txt"
8. cd into SRCNN directory and Follow instructions to download train/test dataset (https://github.com/Lornatang/SRCNN-PyTorch/tree/main/data)
9. (OPTIONAL) Change optimizations to run in optimize_param.py by modifying optimization_params list
9. run following command on terminal : "python optimize.py"
10. output is in Results folder
11. To run XGen speedup download all the onnx models in Results/Optimized_models/Onnx/ and follow XGen Speed Test Guidelines.


# Models 
In Results folder the following prefixs are listed below: 

* pretrained - gives the base pretrained model for SCRNN
* prune_level_s<x> - gives pruned model using level pruning with sparsity of x
* prune_l1_s<x> - gives pruned model using l1NormPruner with sparsity of x
* pruned_fpgm_s<x> - gives pruned model using FPGMPruner with sparsity of x
* quant_lsq_w<x> - gives quantized model using LsqQuantizer with weight precision x
* quant_bnn_w<x> - gives quantized model using BNNQuantier with weight precision x
* quant_drf_w<x> - gives quantized model using DoReFaQuantier with weight precision x


asdf
test1 - l1norm = .1 + DoReFa =8
test5 - l1norm = .5 + DoReFa =8
test9 -l1norm = .9 + DoReFa =8
test 11 - level =.1 + lsq = 8