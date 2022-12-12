#torch imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch import optim

#nni imports
from nni.compression.pytorch import ModelSpeedup
#from nni.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

#etc
import sys
sys.path.insert(0, './SRCNN')
from copy import deepcopy
import os
from SRCNN.model import SRCNN
from SRCNN.train import train, load_dataset, define_loss, define_optimizer
import cv2
import numpy as np
import SRCNN.imgproc as imgproc
from SRCNN.image_quality_assessment import PSNR, SSIM
from natsort import natsorted
from optimize_param import optimization_params, model_param, inference_param




#HELPER METHODS
def export(model, optimization_param, checkpoint):
    #set model to inference mode
    model.eval()
    #create dummy input variable
    x = torch.randn(model_param["input_size"], requires_grad = True).to(model_param["device"])
    torch_out = model(x)
    #convert model to onnx
    torch.onnx.export(model,
                      x,
                      "./Results/Optimized_models/Onnx/"+optimization_param['output file prefix']+ ".onnx",
                      do_constant_folding= True,
                      opset_version = 11,
                      input_names = ['input'],
                      output_names = ['output'],
                    )
    #export pytorch model
    torch.save({"epoch": checkpoint["epoch"],
                        "best_psnr": checkpoint["best_psnr"],
                        'best_ssim': 0.0,
                        "state_dict": model.state_dict(),
                        "optimizer": checkpoint["optimizer"]},
                        os.path.join("Results/Optimized_models/Pytorch", optimization_param['output file prefix'] + ".pth.tar"))
    
    print("export finished")
    
def loadModel():
    #Load checkpoint model
    model = SRCNN()
    checkpoint = torch.load(model_param["model_path"], map_location=lambda storage, loc: storage)
    # Load checkpoint state dict. Extract the fitted model weights
    model_state_dict = model.state_dict()
    new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
    # Overwrite the pretrained model weights to the current model
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)
    # Load the optimizer model
    print("Loaded pretrained model weights.")
    return model, checkpoint

def trainHelper(model, epoch):
    optimizer = define_optimizer(model)
    train_prefetcher, test_prefetcher = load_dataset()
    pixel_criterion = define_loss()
    scaler = amp.GradScaler()
    writer = SummaryWriter(os.path.join("samples", "logs", "SRCNN_x2"))
    train(model = model, train_prefetcher= train_prefetcher, pixel_criterion= pixel_criterion, optimizer=optimizer,  epoch = epoch, scaler=scaler, writer=writer)
    

def test(model, optimization_param):
    # Initialize the super-resolution model
    model = model.to(device=model_param['device'], memory_format=torch.channels_last)
    print("Build SRCNN model successfully.")


    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("SRCNN","results", "test", "SRCNN_x2")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the sharpness evaluation function
    psnr = PSNR(model_param['scale'], False)
    ssim = SSIM(model_param['scale'], False)

    # Set the sharpness evaluation function calculation device to the specified model
    psnr = psnr.to(device=model_param['device'], memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=model_param['device'], memory_format=torch.channels_last, non_blocking=True)

    # Initialize IQA metrics
    psnr_metrics = 0.0
    ssim_metrics = 0.0
    lr_dir = f"./SRCNN/data/Set5/GTmod12"
    sr_dir = f"./SRCNN/results/test/SRCNN_x2"
    hr_dir = f"./SRCNN/data/Set5/GTmod12"

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

        lr_image = imgproc.image_resize(lr_image, 1 / model_param["scale"])
        lr_image = imgproc.image_resize(lr_image, model_param["scale"])

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
    
    #write to file 
    with open('./Results/PSNR/PSNR.txt', 'a') as f: 
        f.write('\n PSNR: ' + f"{avg_psnr:4.2f} dB - " + optimization_param['output file prefix'])

def inference(model, optimization_param):
    # Initialize the model
    model = model.to(memory_format=torch.channels_last, device=model_param['device'])

    # Start the verification mode of the model.
    model.eval()

    # Read LR image and HR image
    lr_image = cv2.imread("./Results/Inference_Image/"+ inference_param["low res file name"] , cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0

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
    cv2.imwrite("./Results/Inference_Image/"+ inference_param["sr prefix name"] + "_"+ optimization_param["output file prefix"] + ".png", sr_image * 255.0)

    print(f"SR image save to ./Results/Inference_Image/"+ inference_param["sr prefix name"] + "_"+ optimization_param["output file prefix"] + ".png")



#MODEL OPTIMIZATIONS
#prunes model
def prune(model, optimization_param):
    pruner = optimization_param['optim function'](model = model, config_list = optimization_param['config'] )
    masked_model, masks = pruner.compress()
    pruner._unwrap_model()
    ModelSpeedup(model, torch.rand(model_param["input_size"]).to(model_param["device"]), masks).speedup_model()
    return model
    
        
#quantizes model 
def quantization(model, optimization_param):
    #Quantize model- post training done outside method
    optimizer = optim.SGD([{"params": model.features.parameters()},
                           {"params": model.map.parameters()},
                           {"params": model.reconstruction.parameters(), "lr": 1e-4 * 0.1}],
                          lr=1e-4,
                          momentum=0.9,
                          weight_decay=1e-4,
                          nesterov=False)
    quantizer = optimization_param["optim function"](model = model, config_list = optimization_param["config"], optimizer = optimizer)
    quantizer.compress()
    return model


#takes in onnx model outputs optimized onnx model
if __name__ == "__main__":
    #remove PSNR log file
    if os.path.exists("./Results/PSNR/PSNR.txt"):
        os.remove("./Results/PSNR/PSNR.txt")

    #gain param of optization to apply
    for optimization_param in optimization_params:
        #Load pretrained model
        model, checkpoint = loadModel()
        model.to(model_param["device"])
        
        #if statement for which model optimization to apply to model
        if optimization_param['operation'] == "prune":
            model = prune(model, optimization_param)
        elif optimization_param['operation'] == "quantization":
            model = quantization(model, optimization_param)
            
        #post train model - some issues on some runs
        #print("POST TRAINING:")
        #trainHelper(model, epoch = 1)

        #export model as onnx and pth.tar model
        print("\nEXPORT MODEL:")
        export(model, optimization_param, checkpoint)

        #gives inference for low resolution butterfly image and output optimized models high res image
        print("\nMODEL INFERENCE:")
        inference(model, optimization_param)
        
        #Test/get PSNR for test set
        print("\nTEST:")
        test(model, optimization_param)
        print("\nModel Completed")
    
    print("\nFinished Program")