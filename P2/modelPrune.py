import torch
import torchvision
from torchvision import datasets, transforms
import time 
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner, FPGMPruner
from nni.compression.pytorch import ModelSpeedup
from mnist.main import Net


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
        'pruner': None,
        'config_list': None, 
    },
    'Model 1': {
        'pruner': L1NormPruner,
        'config_list': [{
            'sparsity_per_layer': 0.2,
            'op_types': ['Conv2d']
        }] 
    },
    'Model 2': {
        'pruner': L1NormPruner,
        'config_list': [{
            'sparsity_per_layer': 0.7,
            'op_types': ['Conv2d']
        }] 
    },
    'Model 3': {
        'pruner': FPGMPruner,
        'config_list': [{
            'sparsity_per_layer': 0.2,
            'op_types': ['Conv2d']
        }] 
    },
    'Model 4': {
        'pruner': FPGMPruner,
        'config_list': [{
            'sparsity_per_layer': 0.7,
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

}

#Apply each pruning/config options in modelParam dictionary on pretrained mnist model.
for key, val in modelParam.items():  
    # define model and load pretrained MNIST model.
    model = Net()
    model.load_state_dict(torch.load("./pre-trained_model/mnist_cnn.pt"))
    
    #perform pruning on model
    if key != 'Model Base':
        pruner = val['pruner'](model = model, config_list =val['config_list'])
        masked_model, masks = pruner.compress()
        pruner._unwrap_model()
        ModelSpeedup(model, torch.rand(1000, 1, 28, 28), masks).speedup_model()

    #get models outputs  
    modelOutput[key] = test(model)

for key, val in modelOutput.items(): 
    print()
    print(key + ":")
    if key != "Model Base":
        print("Pruner Used: " + modelParam[key]['pruner'].__name__)
        print("Configurations Used: " + str(modelParam[key]['config_list']))
    print("Structure: ")
    print(val['model'])
    print("Accuracy: "+ str(val['Accuracy']))
    print("Inference Time: "+ str(val['Inference Time']))


"""# define model and load pretrained MNIST model.
model = Net()
model.load_state_dict(torch.load("./pre-trained_model/mnist_cnn.pt"))
test(model)

#Model pruning  1
config_list = [{
    'sparsity_per_layer': 0.2,
    'op_types': ['Conv2d']
}]

pruner = L1NormPruner(model = model, config_list =config_list)
masked_model, masks = pruner.compress()
pruner._unwrap_model()
ModelSpeedup(model, torch.rand(1000, 1, 28, 28), masks).speedup_model()

test(model)"""