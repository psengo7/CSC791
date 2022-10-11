import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import time 
from nni.algorithms.compression.pytorch.quantization import LsqQuantizer, QAT_Quantizer, BNNQuantizer
from mnist.main import Net, train
import argparse
#from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

#tests model on the MNIST tests data and gives information on models accuracy and inference time. 
def test(model):
    startTime = time.time()
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset2 = datasets.MNIST('/mnt/beegfs/psengo/CSC791/P1/data', train=False,
                        transform=transform)
    testloader = torch.utils.data.DataLoader(dataset2, batch_size=1000,
                                            shuffle=False, num_workers=2)

    #get accuracy and time 
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
        
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print("Execution time (Seconds): " + str((time.time() - startTime)/total))
    #print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    return  {
        "model": model, 
        "Accuracy": 100 * (correct / total), 
        "Inference Time": (time.time() - startTime)/total
    }

#pruning method and configs that will be run on base model
modelParam = {
    'Model Base': {
        'quantizer': None,
        'config_list': None, 
    },
    'Model 1': {
        'quantizer': LsqQuantizer,
        'config_list': [{
            'quant_types': ['weight', 'input'],
            'quant_bits': {'weight': 8, 'input': 8},
            'op_types': ['Conv2d']
        }] 
    },
    'Model 2': {
        'quantizer': LsqQuantizer,
        'config_list': [{
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_types': ['Conv2d']
        }] 
    },
    'Model 3': {
        'quantizer': QAT_Quantizer,
        'config_list': [{
            'quant_types': ['input', 'weight'],
            'quant_bits': {'input': 8, 'weight': 8},
            'op_types': ['Conv2d']
        }] 
    },
    'Model 4': {
        'quantizer': QAT_Quantizer,
        'config_list': [{
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_types': ['Conv2d']
        }] 
    },
    'Model 5': {
        'quantizer': BNNQuantizer,
        'config_list': [{
            'quant_types': ['input', 'weight'],
            'quant_bits': {'input': 8, 'weight': 8},
            'op_types': ['Conv2d']
        }] 
    },
    'Model 6': {
        'quantizer': BNNQuantizer,
        'config_list': [{
            'quant_types': ['output'],
            'quant_bits': {'output': 8},
            'op_types': ['Conv2d']
        }] 
    },
}

#Will stores the output of each model(model structure, accuracy, inference time) 
modelOutput = {
    'Model Base':None,
    'Model 1':None,
    'Model 2':None,
    'Model 3':None,
    'Model 4':None,
    'Model 5':None,
    'Model 6':None,

}

#training variables 
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

#TODO: rerun model using cuda for speedup?
device = torch.device("cpu")
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset1 = datasets.MNIST('/mnt/beegfs/psengo/CSC791/P1/data', train=True, download=True,
        transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1)

#Apply each quantization/config options in modelParam dictionary on pretrained mnist model.
for key, val in modelParam.items():  
    # define model and load pretrained MNIST model.
    model = Net()
    model.load_state_dict(torch.load("./pre-trained_model/mnist_cnn.pt"))
    print(key)
    
    #perform quantization on model
    if key != 'Model Base':
        print(model)
        #Quantize model
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        quantizer = val['quantizer'](model = model, config_list =val['config_list'], optimizer = optimizer)
        quantizer.compress()
        
        #Fine tune model 
        for epoch in range(3):
            train(args, model, device, train_loader, optimizer, epoch)
        
        #Export model to get calibration config
        model_path = "./log/mnist_model.pth"
        calibration_path = "./log/mnist_calibration.pth"
        calibration_config = quantizer.export_model(model_path, calibration_path)

        #speedup model
        #TODO change input shape
        #input_shape = (1, 32, 3, 1)
        #model = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32)
        #model.compress()
        
    #get models outputs  
    modelOutput[key] = test(model)

for key, val in modelOutput.items(): 
    print()
    print(key + ":")
    if key != "Model Base":
        print("quantizer Used: " + modelParam[key]['quantizer'].__name__)
        print("Configurations Used: " + str(modelParam[key]['config_list']))
    print("Structure: ")
    print(val['model'])
    print("Accuracy: "+ str(val['Accuracy']))
    print("Inference Time: "+ str(val['Inference Time']))

