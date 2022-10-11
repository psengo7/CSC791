from unittest import TestLoader
import torch
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import time 
from nni.algorithms.compression.pytorch.quantization import LsqQuantizer, DoReFaQuantizer, BNNQuantizer, QAT_Quantizer
from mnist.main import Net, train
import argparse
#from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

def test(model, device, test_loader):
    startTime = time.time()
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    total = len(test_loader.dataset)
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
        'Output':{} 
    },
    'Model 1': {
        'quantizer': LsqQuantizer,
        'config_list': [{
            'quant_types': ['input', 'weight'],
            'quant_bits': {'input':8, 'weight': 8},
            'op_types': ['Conv2d']
        }],
        'Output':{} 
    },
    'Model 2': {
        'quantizer': LsqQuantizer,
        'config_list': [{
            'quant_types': ['input', 'weight'],
            'quant_bits': {'input':8, 'weight': 8},
            'op_types': ['Conv2d']
        }],
        'Output':{} 
    },
    'Model 3': {
        'quantizer': DoReFaQuantizer,
        'config_list': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_types': ['Conv2d']
        }],
        'Output':{} 
    },
    'Model 4': {
        'quantizer': DoReFaQuantizer,
        'config_list': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
        }],
        'Output':{} 
    },
    'Model 5': {
        'quantizer': BNNQuantizer,
        'config_list': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_types': ['Conv2d']
        }],
        'Output':{} 
    },
    'Model 6': {
        'quantizer': BNNQuantizer,
        'config_list': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
        }],
        'Output':{} 
    },
}

#Initialize variables variables 
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

use_cuda = True
device = torch.device("cuda")
train_kwargs = {'batch_size': 64}
test_kwargs = {'batch_size': 1000}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
dataset1 = datasets.MNIST('/mnt/beegfs/psengo/CSC791/P1/data', train=True, download=True,
                    transform=transform)
dataset2 = datasets.MNIST('/mnt/beegfs/psengo/CSC791/P1/data', train=False,
                    transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


#Apply each quantization/config options in modelParam dictionary on pretrained mnist model.
for key, val in modelParam.items():  
    # define model and load pretrained MNIST model.
    model = Net().to(device)
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
        torch.save(model.state_dict(), "./log/mnist_model.pth")
        model_path = "./log/mnist_model.pth"
        calibration_path = "./log/mnist_calibration.pth"
        calibration_config = quantizer.export_model(model_path, calibration_path)
        print(calibration_config)
        
        #speedup model
        #TODO change input shape
        #input_shape = (1, 32, 3, 1)
        #model = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32)
        #model.compress()
        
    #get models outputs  
    val['Output'] = test(model,device, test_loader)

for key, val in modelParam.items(): 
    print()
    print(key + ":")
    if key != "Model Base":
        print("quantizer Used: " + val['quantizer'].__name__)
        print("Configurations Used: " + str(val['config_list']))
    print("Structure: ")
    print(val['Output']['model'])
    print("Accuracy: "+ str(val['Output']['Accuracy']))
    print("Inference Time: "+ str(val['Output']['Inference Time']))

