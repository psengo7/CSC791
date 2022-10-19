* File structure: 

P1/
├── mnist/
│   └── main.py          #Produces pretrained model using MNIST dataset
├── pre-trained_model/   
│    └── mnist_cnn.pt    #Holds exported pretrained model from mnist/main.py
├── modelPrune.py        #Runs pruning on mnist_cnn.pt model and produces evaluation
├── Report1.pdf          #Gives the report for the project. 
└── README.md


* HOW TO RUN PROGRAM: 
1. Open mnist/main.py and change file location on line 124 and 126 to folder you would like to download MNIST dataset in.
2. Open modelPrune.py change file location on line 18 to folder you downloaded MNIST dataset in.
3. Run "python3 mnist/main.py" once script finishs it should have created a file in the mnist/ directory called mnist_cnn.pt. This is the exported model after pretraining it with the mnist dataset. 
4. Copy mnist/mnist_cnn.pt to pre-trained_model/ directory (or you can just use the exported model I used that is already in the pre=trained_model folder.)
5. In the same directory as modelPrune.py run "python3 modelPrune.py" the 