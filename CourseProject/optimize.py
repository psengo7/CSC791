#torch imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision
import torch.nn.functional as F
from torch.optim import SGD
import torch.optim as optim
from torchvision import datasets, transforms
#nni imports
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import LevelPruner
from nni.algorithms.compression.pytorch.quantization import QAT_Quantizer
from nni.compression.pytorch import ModelSpeedup
#from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
#etc
from copy import deepcopy
import time 
import onnx 
import sys
import numpy as np
from onnx2torch import convert
import cv2
from torchmetrics import PeakSignalNoiseRatio
from torchsr.datasets import Div2K
from basicsr.archs.rrdbnet_arch import RRDBNet
from torchsr.models import rdn
from models import SRCNN
#https://github.com/Coloquinte/torchSR

#from Real_ESRGAN.realesrgan.models.realesrgan_model import RealESRGANModel

#HELPER METHODS
def exportOnnx(model, model_param):
    #set model to inference mode
    model.eval()
    #create dummy input variable
    x = torch.randn(model_param["input_size"], requires_grad = True).to(model_param["device"])
    torch_out = model(x)
    #convert model to onnx
    torch.onnx.export(model,
                      x,
                      "./model_save/optimized_model.onnx",
                      do_constant_folding= True,
                      opset_version = 9,
                      input_names = ['input'],
                      output_names = ['output'],
                    )
    
#onnx file to pytorch model
def importOnnx(filename):
    onnx_model = onnx.load(filename)
    model = convert(onnx_model)
    return model 

#MODEL RESULTS
#used to get sr resolution image manually of model 
def manual(model, model_param):
    #read in image path
    lrImage = cv2.imread("./Images/other/image_blur.jpg")
    hrImage = cv2.imread("./Images/other/image.jpg")
    #define tensor transformation
    transform = transforms.ToTensor()
    #transform image to pytorch tensor
    lr = transform(lrImage).unsqueeze(0).to(model_param["device"])
    hr = transform(hrImage).unsqueeze(0).to(model_param["device"])
    pred_hr = model(lr)

    #calculate psnr
    save_image(pred_hr, './Images/other/image_pred.jpg')
    psnr = PeakSignalNoiseRatio()(pred_hr, hr).item()
    print(psnr)

#retrains model after quantization or pruning
def retrain(model, model_param):
    #look at loading models train/ validation methods. 
    
    model.train()
    dataset = Div2K(root = "/ocean/projects/cis220070p/psengo/data", scale = 2, download = False)
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    for (hr,lr) in dataset:
        hr = transforms.ToTensor()(hr).unsqueeze(0).to(model_param["device"])
        lr = transforms.ToTensor()(lr).unsqueeze(0).to(model_param["device"])
        optimizer.zero_grad()
        pred_hr = model(lr)
        loss = F.nll_loss(pred_hr, hr)
        loss.backward()
        optimizer.step()

    
#Get PSNR score, inference speed(will need to do one for phone), and size of onnx model.
def validate(model): 
    output ={
        "Model Name" :"",
        "PSNR": 0,
        "Inference Time":0, 
        "ONNX Model Size":0
    }
    # load dataset / train with dataset
    count = 0 
    PSNR = 0
    max_count = 1 
    totTime = 0 
    dataset = Div2K(root = "/ocean/projects/cis220070p/psengo/data", scale = 2, download = False)
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,shuffle=False, num_workers=2)
    hr, lr = dataset[0]

    with torch.no_grad():
        for (hr,lr) in dataset:
            hr = transforms.ToTensor()(hr).unsqueeze(0)
            lr = transforms.ToTensor()(lr).unsqueeze(0)

            startTime =time.time()
            pred_hr = model(lr)
            totTime += (time.time()-startTime)
            
            PSNR += PeakSignalNoiseRatio()(pred_hr, hr).item()
            
            save_image(pred_hr, './Images/pred_hr/pred_hr'+str(count)+ '.png')
            save_image(hr, './Images/hr/hr'+str(count)+ '.png')
            save_image(lr, './Images/lr/lr'+str(count)+ '.png')
            count +=1
            if count == max_count: 
                break 
            
    output["Inference Time"] = totTime/count
    output["PSNR"] = PSNR/count
    print("PSNR:" + str(output["PSNR"]))
    print("Inference Time:" + str(output["Inference Time"]))
    return output

#MODEL OPTIMIZATIONS
#prunes model
def prune(model, model_param):
    config_list = [{
        'sparsity_per_layer': 0.1,
        'op_types': ['Conv2d']
    }]
    pruner = LevelPruner(model = model, config_list =config_list )
    masked_model, masks = pruner.compress()
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(model_param["input_size"]).to(model_param["device"]), masks).speedup_model()
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
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    quantizer = QAT_Quantizer(model = model, config_list =config, optimizer = optimizer)
    quantizer.compress()
    
    #Fine tune model 
    for epoch in range(3):
        retrain(model, model_param)
    
    #Export model to get calibration config
    torch.save(model.state_dict(), "./log/mnist_model.pth")
    model_path = "./log/mnist_model.pth"
    calibration_path = "./log/mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    
    #speedup model
    model = ModelSpeedupTensorRT(model, model_param["input_size"], config=calibration_config, batchsize=32)
    model.compress()
    return model

#distills model
def distillation(model_t, model_s):
    #save pruned model as student model
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
    operation = 'manual'
    isSave = False
    
    optimization_param = { 
        'optim function':'',
        'config': ''
    }

    model_param = {
        'device': "cpu",
        'scale':2,
        'input_size':[1,3,64,64]
        }
    
    #import model
    #model = rdn(model_param["scale"], pretrained=True).to(model_param["device"])
    model = SRCNN()
    model.load_state_dict(torch.load("./srcnn_x4.pth"))

    #if statement for each model optimization to apply to model
    if operation == "prune":
        model = prune(model, model_param)
    elif operation == "quantization":
        model = quantization(model)
    elif operation == "distillation":
        model_t = deepcopy(model)
        model.load_state_dict(torch.load("./model_save/model.pt"))
        model = distillation(model_t, model)
    elif operation == "validate":
        model.load_state_dict(torch.load("./model_save/model.pt"))
        validate(model)
    elif operation == "manual":
        #model.load_state_dict(torch.load("./model_save/model.pt"))
        manual(model, model_param)

    #save optimized model as onnx
    if isSave == True:
        torch.save(model.state_dict(), "./model_save/model.pt") 
        exportOnnx(model, model_param)

    print("DONE")