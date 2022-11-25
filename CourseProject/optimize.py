#torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import datasets, transforms
from torchmetrics import PeakSignalNoiseRatio
#nni imports
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import L1NormPruner
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch import ModelSpeedup
#etc
from copy import deepcopy
import time 
import onnx 
import sys
import numpy as np
from onnx2torch import convert
import cv2
from basicsr.archs.rrdbnet_arch import RRDBNet

#from Real_ESRGAN.realesrgan.models.realesrgan_model import RealESRGANModel

#HELPER METHODS
def exportOnnx(model):
    #set model to inference mode
    model.eval()
    #create dummy input variable
    x = torch.randn(1, 1, 28, 28, requires_grad = True)
    torch_out = model(x)
    #convert model to onnx
    torch.onnx.export(model,
                      x,
                      "./pre-trained_model/mnist.onnx",
                      export_params = True,
                      opset_version = 10,
                      input_names = ['data'],
                      output_names = ['output'],
                    )
    
#onnx file to pytorch model
def importOnnx(filename):
    onnx_model = onnx.load(filename)
    model = convert(onnx_model)
    return model 

#MODEL RESULTS

#give psnr of 2 images
def psnrMetric(predImagePth, targetImagePth):
    #read in image path
    predImage = cv2.imread(predImagePth)
    targetImage = cv2.imread(targetImagePth)
    #define tensor transformation
    transform = transforms.ToTensor()
    #transform image to pytorch tensor
    preds = transform(predImage)
    target = transform(targetImage)

    print("pred(" + str(preds.size()) + ")")
    print(preds)
    print("target(" + str(target.size()) + ")")
    print(target)
    #calculate psnr
    psnr = PeakSignalNoiseRatio()
    psnrVal = psnr(preds, target) 
    print(psnrVal)


#Get PSNR score, inference speed(will need to do one for phone), and size of onnx model.
def validate(model, trainer): 
    output ={
        "PSNR": 0,
        "Inference Time":0, 
        "ONNX Model Size":0
    }
    
    return output

#MODEL OPTIMIZATIONS
#prunes model
def prune(model):
    config_list = [{
        'sparsity_per_layer': 0.2,
        'op_types': ['Conv2d']
    }]
    pruner = L1NormPruner(model = model, config_list =config_list )
    masked_model, masks = pruner.compress()
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(1, 3, 64, 64).to("cuda"), masks).speedup_model()
    return model
   
#quantizes model 
def quantization(model):
    #config
    config = [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
    }]

    #Quantize model
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    quantizer = QAT_Quantizer(model = model, config_list =val['config_list'], optimizer = optimizer)
    quantizer.compress()
    
    #Fine tune model 
    for epoch in range(3):
        train(args, model, device, train_loader, optimizer, epoch)
    
    #Export model to get calibration config
    torch.save(model.state_dict(), "./log/mnist_model.pth")
    model_path = "./log/mnist_model.pth"
    calibration_path = "./log/mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    
    #speedup model
    input_shape = (1, 32, 3, 1)
    model = ModelSpeedupTensorRT(model, input_shape, config=calibration_config, batchsize=32)
    model.compress()
    return model

#distills model
def distillation(model):
    #TODO: need dataset(train_loader)
    #original model
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
    return model_s


#takes in onnx model outputs optimized onnx model
if __name__ == "__main__":
    #Get Global Params and get param from user
    operation = 'prune'
    isSave = False
    
    #import model (input = [1,3, 64, 64])
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to("cuda")
    model.load_state_dict(torch.load("./RealESRGAN_x4plus.pth")['params_ema'])

    #if statement for each model optimization to apply to model
    if operation == "prune":
        model = prune(model)
    elif operation == "quantization":
        model = quantization(model)
    elif operation == "distillation":
        model = distilation(model)
    print("DONE")
    #save optimized model as onnx
    if isSave == True: 
        exportOnnx(model)