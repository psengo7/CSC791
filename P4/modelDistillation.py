import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision
from torchvision import datasets, transforms
import time 
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner
from nni.compression.pytorch import ModelSpeedup
from mnist.main import Net
from copy import deepcopy
import torch.nn.functional as F
import onnx 
import sys
import numpy as np

#tests model on the MNIST tests data and gives information on models accuracy and inference time. 
def test(model, dataset2, testloader):
    startTime = time.time()
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

    return  {
        "model": model, 
        "Accuracy": 100 * (correct / total), 
        "Inference Time": (time.time() - startTime)/total
    }



def main():
    #STEP 0: PARAMS -define global params
    #Will stores the output of each model(model structure, accuracy, inference time) 
    print("Step 0 started...")
    modelOutput = {
        'Model Base':None,
        'Model Prune':None,
        'Model Distillation':None,
    }

    #get data and loaders for model
    transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    dataset1 = datasets.MNIST('/ocean/projects/cis220070p/psengo/data', train=True, download=True,
                        transform=transform)
    dataset2 = datasets.MNIST('/ocean/projects/cis220070p/psengo/data', train=False,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64,
                                                shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=1000,
                                            shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Step 0 Complete")

    #STEP 1: BASE MODEL-define model and load pretrained MNIST model.
    print("Step 1 started...")
    model = Net()
    model.load_state_dict(torch.load("./pre-trained_model/mnist_cnn.pt"))
    modelOutput['Model Base'] = test(model, dataset2, test_loader)
    print("Step 1 Complete")

    #STEP 2: PRUNING-Apply Pruning on base model 
    print("Step 2 started...")
    config_list = [{
        'sparsity_per_layer': 0.2,
        'op_types': ['Conv2d']
    }]
    pruner = L1NormPruner(model = model, config_list =config_list )
    masked_model, masks = pruner.compress()
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(1000, 1, 28, 28), masks).speedup_model()
    modelOutput['Model Prune'] = test(model, dataset2, test_loader)
    torch.save(model.state_dict(), "./pre-trained_model/mnist_prune.pt")
    print("Step 2 Complete")

    #STEP 3: DISTILLATION - Apply distilation on pruned model
    #original trained model as teacher
    print("Step 3 started...")
    model_t = Net()
    model_t.load_state_dict(torch.load("./pre-trained_model/mnist_cnn.pt"))
    #save pruned model as student model
    model_s = deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_s.parameters(), lr = 1e-3)

    #train student model with teacher model
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        #data shape: 64, 1, 28, 28

        #compute prediction error
        y_s = model_s(data)
        y_t = model_t(data)
        loss_cri = F.cross_entropy(y_s, target)

        # compute loss and backpropogation
        kd_T = 4 #temperature for knowledge distillation
        p_s = F.log_softmax(y_s/kd_T, dim=1)
        p_t = F.softmax(y_t/kd_T, dim=1)
        loss_kd = F.kl_div(p_s, p_t, size_average=False) * (kd_T**2) / y_s.shape[0]

        # total loss
        loss = loss_cri + loss_kd
        optimizer.zero_grad()
        loss.backward()
    
    #test student model
    modelOutput['Model Distillation'] = test(model_s, dataset2, test_loader)
    torch.save(model_s.state_dict(), "./pre-trained_model/mnist_distill.pt")
    exportOnnx(model_s)
    print("Step 3 Complete")

    #STEP 4: OUTPUT - Output results
    for key, val in modelOutput.items(): 
        print()
        print(key + ":")
        print("Structure: ")
        print(val['model'])
        print("Accuracy: "+ str(val['Accuracy']))
        print("Inference Time: "+ str(val['Inference Time']))

def exportOnnx(model):
    #set model to inference mode
    model.eval()
    #create dummy input variable
    x = torch.randn(1, 1, 28, 28, requires_grad = True)
    torch_out = model(x)
    #convert model to onnx
    torch.onnx.export(model,
                      x,
                      "./pre-trained_model/mnist_distill.onnx",
                      export_params = True,
                      opset_version = 10,
                      input_names = ['data'],
                      output_names = ['output'],
                      )

def preprocessData(): 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('/ocean/projects/cis220070p/psengo/data', train=True, download=True,
                        transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=1,
                                                shuffle=True)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data, target
        #convert data to onnx format
        print(data[0])
        print(target)
        #add batch dimension
        i_data = np.expand_dims(data[0], axis=0)
        #print(data) 
        np.savez("./tvm_files/image",data = i_data)
        break

if __name__ == "__main__":
    
    arg = sys.argv[1]
    #arg = "exportVal"
    if arg == "main":
        #run pruning and distillation on mnist model
        main()
    elif arg == "toOnnx":
        #export distilled model to onnx
        model = Net()
        model.load_state_dict(torch.load("./pre-trained_model/mnist_distill.pt")) 
        exportOnnx(model)
    elif arg == "image":
        #get text mnist data
        preprocessData()