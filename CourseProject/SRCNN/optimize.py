#torch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

#nni imports
from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import LevelPruner, L1NormPruner, FPGMPruner, L2NormPruner
from nni.algorithms.compression.pytorch.quantization import LsqQuantizer, DoReFaQuantizer, BNNQuantizer
from nni.compression.pytorch import ModelSpeedup
#from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

#etc
from copy import deepcopy
import os
from model import SRCNN
from train import train, load_dataset, define_loss, define_optimizer
import config
import cv2
import numpy as np
import imgproc
from image_quality_assessment import PSNR, SSIM
from natsort import natsorted

#HELPER METHODS
def exportOnnx(model, model_param, optimization_param):
    #set model to inference mode
    model.eval()
    #create dummy input variable
    x = torch.randn(model_param["input_size"], requires_grad = True).to(model_param["device"])
    torch_out = model(x)
    #convert model to onnx
    torch.onnx.export(model,
                      x,
                      "./optimize_model/onnx/"+optimization_param['output model name']+ ".onnx",
                      do_constant_folding= True,
                      opset_version = 11,
                      input_names = ['input'],
                      output_names = ['output'],
                    )
    print("export finished")
    
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

def trainHelper(model, epoch, model_param, optimization_param):
    optimizer = define_optimizer(model)
    train_prefetcher, test_prefetcher = load_dataset()
    pixel_criterion = define_loss()
    scaler = amp.GradScaler()
    writer = SummaryWriter(os.path.join("samples", "logs", "SRCNN_x2"))

    train(model = model, train_prefetcher= train_prefetcher, pixel_criterion= pixel_criterion, optimizer=optimizer,  epoch = epoch, scaler=scaler, writer=writer)
    
def test(model, model_param, optimization_param):
    # Initialize the super-resolution model
    model = model.to(device=model_param['device'], memory_format=torch.channels_last)
    print("Build SRCNN model successfully.")


    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the sharpness evaluation function
    psnr = PSNR(config.upscale_factor, False)
    ssim = SSIM(config.upscale_factor, False)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=model_param['device'], memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=model_param['device'], memory_format=torch.channels_last, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0
    lr_dir = f"./data/Set5/GTmod12"
    sr_dir = f"./results/test/{config.exp_name}"
    hr_dir = f"./data/Set5/GTmod12"

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(lr_dir, file_names[index])
        sr_image_path = os.path.join(sr_dir, file_names[index])
        hr_image_path = os.path.join(hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image and HR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

        lr_image = imgproc.image_resize(lr_image, 1 / config.upscale_factor)
        lr_image = imgproc.image_resize(lr_image, config.upscale_factor)

        # Get Y channel image data
        lr_y_image = imgproc.bgr2ycbcr(lr_image, True)
        hr_y_image = imgproc.bgr2ycbcr(hr_image, True)

        # Get Cb Cr image data from hr image
        hr_ycbcr_image = imgproc.bgr2ycbcr(hr_image, False)
        _, hr_cb_image, hr_cr_image = cv2.split(hr_ycbcr_image)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_y_tensor = imgproc.image2tensor(lr_y_image, False, True).unsqueeze_(0)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, False, True).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_y_tensor = lr_y_tensor.to(device=model_param['device'], memory_format=torch.channels_last, non_blocking=True)
        hr_y_tensor = hr_y_tensor.to(device=model_param['device'], memory_format=torch.channels_last, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

        # Save image
        sr_y_image = imgproc.tensor2image(sr_y_tensor, False, True)
        sr_y_image = sr_y_image.astype(np.float32) / 255.0
        sr_ycbcr_image = cv2.merge([sr_y_image, hr_cb_image, hr_cr_image])
        sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
        cv2.imwrite(sr_image_path, sr_image * 255.0)

        # Cal IQA metrics
        psnr_metrics += psnr(sr_y_tensor, hr_y_tensor).item()
        ssim_metrics += ssim(sr_y_tensor, hr_y_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # PSNR range value is 0~100
    # SSIM range value is 0~1
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files

    print(f"PSNR: {avg_psnr:4.2f} dB\n"
          f"SSIM: {avg_ssim:4.4f} u")


def inference(model, model_param, optimization_param):
    # Initialize the model
    model = model.to(memory_format=torch.channels_last, device=model_param['device'])

    # Start the verification mode of the model.
    model.eval()

    # Read LR image and HR image
    lr_image = cv2.imread("./figure/butterfly_lr.png", cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

    # Get Y channel image data
    lr_y_image = imgproc.bgr2ycbcr(lr_image, True)

    # Get Cb Cr image data from hr image
    lr_ycbcr_image = imgproc.bgr2ycbcr(lr_image, False)
    _, lr_cb_image, lr_cr_image = cv2.split(lr_ycbcr_image)

    # Convert RGB channel image format data to Tensor channel image format data
    lr_y_tensor = imgproc.image2tensor(lr_y_image, False, False).unsqueeze_(0)

    # Transfer Tensor channel image format data to CUDA device
    lr_y_tensor = lr_y_tensor.to(device=model_param['device'], memory_format=torch.channels_last, non_blocking=True)

    # Only reconstruct the Y channel image data.
    with torch.no_grad():
        sr_y_tensor = model(lr_y_tensor).clamp_(0, 1.0)

    # Save image
    sr_y_image = imgproc.tensor2image(sr_y_tensor, False, False)
    sr_y_image = sr_y_image.astype(np.float32) / 255.0
    sr_ycbcr_image = cv2.merge([sr_y_image, lr_cb_image, lr_cr_image])
    sr_image = imgproc.ycbcr2bgr(sr_ycbcr_image)
    cv2.imwrite("./figure/butterfly_sr_"+optimization_param["output model name"] + ".png", sr_image * 255.0)

    print(f"SR image save to ./figure/butterfly_sr_"+optimization_param["output model name"] + ".png")



#MODEL OPTIMIZATIONS
#prunes model
def prune(model, model_param, optimization_param):
    config_list = [{
        'sparsity_per_layer': 0.1,
        'op_types': ['Conv2d']
    }]
    pruner = optimization_param['optim function'](model = model, config_list = optimization_param['config'] )
    masked_model, masks = pruner.compress()
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(model_param["input_size"]).to(model_param["device"]), masks).speedup_model()
    return model
    
        
#quantizes model 
def quantization(model, model_param, optimization_param):
    #config
    config = [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
    }]

    #Quantize model
    optimizer = define_optimizer(model)
    quantizer = optimization_param["optim function"](model = model, config_list = optimization_param["config"], optimizer = optimizer)
    quantizer.compress()
    
    #Fine tune model 
    #trainHelper(model, epoch= 5)
    
    
    """
    model_path = "./log/mnist_model.pth"
    calibration_path = "./log/mnist_calibration.pth"
    calibration_config = quantizer.export_model(model_path, calibration_path)
    
    #speedup model
    #model = ModelSpeedupTensorRT(model, model_param["input_size"], config=calibration_config, batchsize=32)
    #model.compress()"""
    return model


#takes in onnx model outputs optimized onnx model
if __name__ == "__main__":
    #Get Global Params and get param from user
    operation = ''
    isSave = True
    
    #Model specific parameters (in case you want to switch models)
    model_param = {
        'model_path': "./optimize_model/pytorch/pretrained.pth.tar",
        'device': "cuda",
        'scale':2,
        'input_size':[1,1,9,9]
        }
    
    #quantizer
    """config = [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
    }]"""

    #prune
    """'config': [{
            'sparsity_per_layer': 0.1,
            'op_types': ['Conv2d']
            }]"""

    #Optimization specific parameters
    optimization_param = {
        'output model name': 'pretrained',
        'optim function':FPGMPruner,
        'config': [{
            'sparsity_per_layer': 0.1,
            'op_types': ['Conv2d']
            }]
    }
    
    #Load pretrained model
    model, checkpoint = loadModel(SRCNN(), model_param["model_path"])
    model.to(model_param["device"])
    
    #if statement for each model optimization to apply to model
    if operation == "prune":
        model = prune(model, model_param, optimization_param)
    elif operation == "quantization":
        model = quantization(model, model_param, optimization_param)
        
    #post train model
    #print("POST TRAINING:")
    #trainHelper(model, epoch = 1, model_param, optimization_param)

    #export model as onnx model
    print("\nEXPORT ONNX:")
    exportOnnx(model, model_param, optimization_param)

    #gives inference for low resolution butterfly image
    print("\nMODEL INFERENCE:")
    inference(model, model_param, optimization_param)
    
    #Test/get PSNR for test set
    print("\nTEST:")
    test(model, model_param, optimization_param)

    #save optimized model as pth file
    if isSave == True:
        torch.save({"epoch": checkpoint["epoch"],
                    "best_psnr": checkpoint["best_psnr"],
                    'best_ssim': 0.0,
                    "state_dict": model.state_dict(),
                    "optimizer": checkpoint["optimizer"]},
                    os.path.join("optimize_model/pytorch", optimization_param['output model name'] + ".pth.tar"))
    
    print("DONE")