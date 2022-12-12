from nni.algorithms.compression.v2.pytorch.pruning.basic_pruner import LevelPruner, L1NormPruner, FPGMPruner
from nni.algorithms.compression.pytorch.quantization import LsqQuantizer, DoReFaQuantizer, BNNQuantizer

inference_param = {
    'sr prefix name': 'butterfly_sr',
    'low res file name': "butterfly_lr.png"
}
#Model specific parameters (in case you want to switch models) - current params are for SRCNN base pretrained model
model_param = {
    'model_path': "./Results/Optimized_models/Pytorch/pretrained.pth.tar",
    'device': "cuda",
    'scale':2,
    'input_size':[1,1,9,9]
}
    

#Each element in the list gives the optimization you want to perform on the base pretrained SRCNN model.
optimization_params = [
    #BASE PRE-TRAINED MODEL
    {
        'operation':'',
        'output file prefix': 'pretrained',
    },
    #PRUNE-LevelPruner
    {
        'operation':'prune',
        'output file prefix': 'prune_level_s1',
        'optim function':LevelPruner,
        'config': [{
            'sparsity_per_layer': 0.1,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_level_s3',
        'optim function':LevelPruner,
        'config': [{
            'sparsity_per_layer': 0.3,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_level_s5',
        'optim function':LevelPruner,
        'config': [{
            'sparsity_per_layer': 0.5,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_level_s7',
        'optim function':LevelPruner,
        'config': [{
            'sparsity_per_layer': 0.7,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_level_s9',
        'optim function':LevelPruner,
        'config': [{
            'sparsity_per_layer': 0.9,
            'op_types': ['Conv2d']
            }]
    },
    #PRUNE - FPGMPruner
    {
        'operation':'prune',
        'output file prefix': 'prune_fpgm_s1',
        'optim function':FPGMPruner,
        'config': [{
            'sparsity_per_layer': 0.1,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_fpgm_s3',
        'optim function':FPGMPruner,
        'config': [{
            'sparsity_per_layer': 0.3,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_fpgm_s5',
        'optim function':FPGMPruner,
        'config': [{
            'sparsity_per_layer': 0.5,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_fpgm_s7',
        'optim function':FPGMPruner,
        'config': [{
            'sparsity_per_layer': 0.7,
            'op_types': ['Conv2d']
            }]
    },
    {
        'operation':'prune',
        'output file prefix': 'prune_fpgm_s9',
        'optim function':FPGMPruner,
        'config': [{
            'sparsity_per_layer': 0.9,
            'op_types': ['Conv2d']
            }]
    },
    #PRUNE - L1NormPruner
    {
        'operation':'prune',
        'output file prefix': 'prune_l1_s1',
        'optim function':L1NormPruner,
        'config': [{
            'sparsity_per_layer': 0.1,
            'op_types': ['Conv2d']
            }]
    },
        {
        'operation':'prune',
        'output file prefix': 'prune_l1_s3',
        'optim function':L1NormPruner,
        'config': [{
            'sparsity_per_layer': 0.3,
            'op_types': ['Conv2d']
            }]
    },
        {
        'operation':'prune',
        'output file prefix': 'prune_l1_s5',
        'optim function':L1NormPruner,
        'config': [{
            'sparsity_per_layer': 0.5,
            'op_types': ['Conv2d']
            }]
    },
        {
        'operation':'prune',
        'output file prefix': 'prune_l1_s7',
        'optim function':L1NormPruner,
        'config': [{
            'sparsity_per_layer': 0.7,
            'op_types': ['Conv2d']
            }]
    },
        {
        'operation':'prune',
        'output file prefix': 'prune_l1_s9',
        'optim function':L1NormPruner,
        'config': [{
            'sparsity_per_layer': 0.9,
            'op_types': ['Conv2d']
            }]
    },
    #QUANTIZATION - LsqQuantizer
    {
        'operation':'quantization',
        'output file prefix': 'quant_lsq_w8',
        'optim function':LsqQuantizer,
        'config': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_types': ['Conv2d']
        }]
    },
    {
        'operation':'quantization',
        'output file prefix': 'quant_lsq_w2',
        'optim function':LsqQuantizer,
        'config': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
        }]
    },
    #QUANTIZATION - BNNQuantizer
    {
        'operation':'quantization',
        'output file prefix': 'quant_bnn_w8',
        'optim function':BNNQuantizer,
        'config': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_types': ['Conv2d']
        }]
    },
    {
        'operation':'quantization',
        'output file prefix': 'quant_bnn_w2',
        'optim function':BNNQuantizer,
        'config': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
        }]
    },
    #QUANTIZATION - DoReFaQuantizer
    {
        'operation':'quantization',
        'output file prefix': 'quant_drf_w8',
        'optim function':DoReFaQuantizer,
        'config': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 8},
            'op_types': ['Conv2d']
        }]
    },
    {
        'operation':'quantization',
        'output file prefix': 'quant_drf_w2',
        'optim function':DoReFaQuantizer,
        'config': [{
            'quant_types': ['weight'],
            'quant_bits': {'weight': 2},
            'op_types': ['Conv2d']
        }]
    } 
]
    