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
from model import SRCNN
#https://github.com/Coloquinte/torchSR
#train method
import os
import shutil
import time
from enum import Enum

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from model import SRCNN

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
                      "./optimize_model/onnx/optimized_model.onnx",
                      do_constant_folding= True,
                      opset_version = 9,
                      input_names = ['input'],
                      output_names = ['output'],
                    )
    
def loadModel(model, model_path):
    #Load checkpoint model
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # Load checkpoint state dict. Extract the fitted model weights
    model_state_dict = model.state_dict()
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)
    # Load the optimizer model
    print("Loaded pretrained model weights.")
    return model, checkpoint


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
    
    #Export model to get calibration config
    torch.save(model.state_dict(), "./log/mnist_model.pth")

    #Fine tuning will be done manually
    """#Fine tune model 
    for epoch in range(3):
        retrain(model, model_param)
    
    
    model_path = "./log/mnist_model.pth"
    calibration_path = "./log/mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    
    #speedup model
    #model = ModelSpeedupTensorRT(model, model_param["input_size"], config=calibration_config, batchsize=32)
    #model.compress()"""
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
    operation = 'export'
    isSave = True
    
    optimization_param = { 
        'optim function':'',
        'config': ''
    }

    model_param = {
        'device': "cpu",
        'scale':2,
        'input_size':[1,1,9,9]
        }

    model_path = "./results/pretrained_models/srcnn_x2-T91-7d6e0623.pth.tar"
    #model_path = "./samples/SRCNN_x2/epoch_0.pth.tar"
    
    #Load pretrained model
    model, checkpoint = loadModel(SRCNN(), model_path)
    #model.load_state_dict(torch.load("./SRCNN-PyTorch/results/pretrained_models/srcnn_x2-T91-7d6e0623.pth.tar"))

    #if statement for each model optimization to apply to model
    if operation == "prune":
        model = prune(model, model_param)
    elif operation == "quantization":
        model = quantization(model)
    elif operation == "distillation":
        model_t = deepcopy(model)
        model.load_state_dict(torch.load("./model_save/model.pt"))
        model = distillation(model_t, model)
    elif operation == "export":
        exportOnnx(model, model_param)
    
    #save optimized model as onnx
    if isSave == True:
       #torch.save(model.state_dict(), "./model_save/model.pt")
        samples_dir = os.path.join("samples", config.exp_name)
        #TODO: may have to change checkpoint epoch to 0 or reduced 
        epoch = checkpoint["epoch"]
        torch.save({"epoch": checkpoint["epoch"],
                    "best_psnr": checkpoint["best_psnr"],
                    'best_ssim': 0.0,
                    "state_dict": model.state_dict(),
                    "optimizer": checkpoint["optimizer"]},
                    os.path.join("optimize_model", "optimize.pth.tar"))
    print("DONE")