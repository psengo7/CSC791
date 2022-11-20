# DIRECTORY STRUCTURE: 

* mnist/
  * main.py **(1)**
* pre-trained_model/   
  * mnist_cnn.pt **(2)**
* tvm_files/ **(3)**
* modelDistillation.py **(4)**
* job-proj4.sh **(5)** 
* README.md

1. Produces pretrained model using MNIST dataset
2. Holds exported pretrained model, pruned model, and distilation model of mnist/main.py
3. Holds tvm model optimization, tar, prediction, and test image.
4. Runs distillation on the model in mnist folder
5. Holds bash script that runs entire distilation and tvm process.


# HOW TO RUN PROGRAM: 

1. Open mnist/main.py and change file location on line 124 and 126 to folder you would like to download MNIST dataset in.
2. Open modelDistillation.py change file location on line 57, 59, 157 to folder you downloaded MNIST dataset in.
3. Run "python3 mnist/main.py" once script finishs it should have created a file in the mnist/ directory called mnist_cnn.pt. This is the exported model after pretraining it with the mnist dataset. 
4. Copy mnist/mnist_cnn.pt to pre-trained_model/ directory (or you can just use the exported model I used that is already in the pre-trained_model folder.)
5. In the P4/ directory run sbatch job-proj4.sh on the bridges login node. 
