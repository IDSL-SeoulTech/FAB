#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
import sys
import argparse
import copy
import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch import autograd
from torch.cuda.amp import GradScaler
from torch.distributed.elastic.multiprocessing import errors
import torchinfo
from common import (
    DEFAULT_EPOCHS,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_ITERATIONS,
)
from cvnets import EMA, get_model
from cvnets.layers import LinearLayer, MultiHeadAttention
from data import create_test_loader, create_train_val_loader
from engine import Evaluator, Trainer
from loss_fn import build_loss_fn
from optim import build_optimizer
from optim.scheduler import build_scheduler
from options.opts import get_eval_arguments, get_training_arguments
from utils import logger, resources
from utils.checkpoint_utils import load_checkpoint, load_model_state
from utils.common_utils import create_directories, device_setup
from utils.ddp_utils import distributed_init, is_master
from cvnets.modules import InvertedResidual
from cvnets.ptq.layers import QLinear
from convert_insert import assign_ptf

torch.autograd.set_detect_anomaly(True)
input_activations = {}
output_activations = {}



def main(opts, **kwargs):
    dev_id = getattr(opts, "dev.device_id", torch.device("cpu"))
    device = getattr(opts, "dev.device", torch.device("cpu"))        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(current_dir, 'base_weight', 'mobilenetv2-1.00.pt')
    use_distributed = getattr(opts, "ddp.use_distributed")
    is_master_node = is_master(opts)

    # set-up data loaders
    test_loader = create_test_loader(opts)

    # set-up the model
    model = get_model(opts) 

    loaded_model = torch.load(PATH)

    set_indices = set()
    recursive_update(model, loaded_model,set_indices)  # for weight update (initiate pretrained weight)
   
    # memory format
    memory_format = (
        torch.channels_last
        if getattr(opts, "common.channels_last")
        else torch.contiguous_format
    )

    

    model = model.to(device=device, memory_format=memory_format)
    model.eval()
    if getattr(opts, "ddp.use_deprecated_data_parallel"):
        logger.warning(
            "DataParallel is not recommended for training, and is not tested exhaustively. \
                Please use it only for debugging purposes. We will deprecated the support for DataParallel in future and \
                    encourage you to use DistributedDataParallel."
        )
        model = model.to(memory_format=memory_format, device=torch.device("cpu"))
        model = torch.nn.DataParallel(model)
        model = model.to(device=device)
    elif use_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dev_id],
            output_device=dev_id,
            find_unused_parameters=getattr(opts, "ddp.find_unused_params"),
        )
        if is_master_node:
            logger.log("Using DistributedDataParallel for training")
            
    quant = getattr(opts,"quant.quant")
    model.info()
    if(quant):
        print("Calibration....")
        image_list=[]
        calib_iter = getattr(opts,"quant.calib_iter")
        for i,data in enumerate(test_loader):
            if i == calib_iter:
                break
            data = data['samples']
            data = data.to(device)
            image_list.append(data)
        model.open_calibration()
        with torch.no_grad():
                for i, image in enumerate(image_list):
                    if i == len(image_list) - 1:
                        try :
                            model.open_last_calibration()
                        except :
                            model.module.open_last_calibration()
                    output = model(image)
        try:
            model.close_calibration()
            print("calibration close")
        except:
            model.module.close_calibration()
        print("Calibration Done")
        try:
            model.model_quant()
            print("Apply Quantization")
        except:
            model.module.model_quant()
        eval_engine = Evaluator(opts=opts, model=model, test_loader=test_loader)
        eval_engine.run()

        ###################################################
        # Activation Extraction for validate in C level   #
        ###################################################
        # save_index=0
        # print("SAVE_REF_OUTPUT")
        # root_path = 
        # for i,data in enumerate(train_loader):
        #     if(i<10):
        #         data = data['samples']
        #         data = data.to(device)
        #         save_data = data[i]
        #         temp = save_data.to("cuda")
        #         y_temp = model(temp.view(1,3,224,224))
        #         save_target = torch.argmax(y_temp)
        #         print("PREDICTION = ",save_target)
        #         file_path = f"{root_path}/valid_output/{str(save_index)}.txt"
        #         with open(file_path, 'w') as file:
        #             file.write(f"{save_target}\n")
        #         file_path = f"{root_path}/valid_input/{str(save_index)}.txt"
        #         with open(file_path, 'w') as file:
        #             for i in save_data.flatten():
        #                 file.write("{:.20f}\n".format(i))
        #         save_index = save_index+1
        #     else:
        #         break

        # print("SAVE_WEIGHT")
        # root_path = "./weight"
        # weights = model.state_dict()
        # for name, param in weights.items():
        #     flattened_weights = param.cpu().flatten().detach().numpy()
        #     file_path = f"{root_path}/{name}.txt"
        #     with open(file_path, 'w') as file:
        #         for w in flattened_weights:
        #             file.write("{:.20f}\n".format(w))

        # example_input = image_list[0][0].view(1,3,224,224).to(device)
        # y = model(example_input)
        # for name, layer in model.named_modules():
        #    hook_input = layer.register_forward_hook(get_input_activation(name))
        #    hook_output = layer.register_forward_hook(get_output_activation(name))

        # result = model(example_input)
        # result = torch.argmax(result)
        
        # root_path = './activation'
        # print("HOOK END")
        # print(result)
        # save_activations_to_individual_files(input_activations, root_path)
        # save_activations_to_individual_files(output_activations, root_path)
    else:
        eval_engine = Evaluator(opts=opts, model=model, test_loader=test_loader)
        eval_engine.run()


def distributed_worker(i, main, opts, kwargs):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

    ddp_rank = getattr(opts, "ddp.rank", None)
    if ddp_rank is None:  # torch.multiprocessing.spawn
        ddp_rank = kwargs.get("start_rank", 0) + i
        setattr(opts, "ddp.rank", ddp_rank)

    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts, **kwargs)


def main_worker(args: Optional[List[str]] = None, **kwargs):
    opts = get_eval_arguments(args=args)
    
    opts = device_setup(opts)

    node_rank = getattr(opts, "ddp.rank")
    if node_rank < 0:
        logger.error("--rank should be >=0. Got {}".format(node_rank))

    is_master_node = is_master(opts)

    # create the directory for saving results
    save_dir = getattr(opts, "common.results_loc")
    run_label = getattr(opts, "common.run_label")
    exp_dir = "{}/{}".format(save_dir, run_label)
    setattr(opts, "common.exp_loc", exp_dir)
    create_directories(dir_path=exp_dir, is_master_node=is_master_node)

    world_size = getattr(opts, "ddp.world_size")
    num_gpus = getattr(opts, "dev.num_gpus")
    # use DDP if num_gpus is > 1
    use_distributed = True if num_gpus > 1 else False
    setattr(opts, "ddp.use_distributed", use_distributed)

    if num_gpus > 0:
        assert torch.cuda.is_available(), "We need CUDA for training on GPUs."

    # No of data workers = no of CPUs (if not specified or -1)
    n_cpus = resources.cpu_count()
    dataset_workers = getattr(opts, "dataset.workers", -1)

    if use_distributed:
        if world_size == -1:
            logger.log(
                "Setting --ddp.world-size the same as the number of available gpus"
            )
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)
        elif world_size != num_gpus:
            logger.log(
                "--ddp.world-size does not match num. of available GPUs. Got {} !={}".format(
                    world_size, num_gpus
                )
            )
            logger.log("Setting --ddp.world-size={}".format(num_gpus))
            world_size = num_gpus
            setattr(opts, "ddp.world_size", world_size)

        if dataset_workers == -1 or dataset_workers is None:
            setattr(opts, "dataset.workers", n_cpus // world_size)

        start_rank = getattr(opts, "ddp.rank")
        setattr(opts, "ddp.rank", None)
        kwargs["start_rank"] = start_rank
        torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts, kwargs),
            nprocs=num_gpus,
        )
    else:
        if dataset_workers == -1:
            setattr(opts, "dataset.workers", n_cpus)

        # adjust the batch size
        train_bsize = getattr(opts, "dataset.train_batch_size0") * max(1, num_gpus)
        val_bsize = getattr(opts, "dataset.val_batch_size0") * max(1, num_gpus)
        setattr(opts, "dataset.train_batch_size0", train_bsize)
        setattr(opts, "dataset.val_batch_size0", val_bsize)
        setattr(opts, "dev.device_id", None)
        main(opts=opts, **kwargs)


# for segmentation and detection, we follow a different evaluation pipeline that allows to save the results too
def main_worker_segmentation(args: Optional[List[str]] = None, **kwargs):
    from engine.eval_segmentation import main_segmentation_evaluation

    main_segmentation_evaluation(args=args, **kwargs)


def main_worker_detection(args: Optional[List[str]] = None, **kwargs):
    from engine.eval_detection import main_detection_evaluation

    main_detection_evaluation(args=args, **kwargs)


def save_activations_to_individual_files(activations, root_path):
    for name, values in activations.items():
        file_path = f"{root_path}/{name}.txt"
        with open(file_path, 'w') as file:
            for val in values.flatten():
                file.write("{:.20f}\n".format(val))
def get_input_activation(name):
    def hook(model, input, output):
        if isinstance(input, tuple):
            try:
                input_activations[f"{name}_input"] = input[0].cpu().detach().numpy()
            except:
                pass
        else :
            input_activations[f"{name}_input"] = input.cpu().detach().numpy()    
       
    return hook

def get_output_activation(name):
    def hook(model, input, output):
        if isinstance(output,tuple):
            output_activations[f"{name}_output"] = output[0].cpu().detach().numpy()
        else:
            output_activations[f"{name}_output"] = output.cpu().detach().numpy()
    return hook

def recursive_update(model, loaded_state_dict,visited_indices):
    
     for name, module in model.named_children():
        # print(name)
        if isinstance(module, (nn.Conv2d)):
            for param_name in ["weight"]:
                if hasattr(module, param_name):
                    target_param = getattr(module, param_name)
                    for idx, (source_name, source_param) in enumerate(loaded_state_dict.items()):
                        if(
                            idx not in visited_indices and  
                            target_param.shape == source_param.shape and
                            ("conv." in source_name )):
                            target_param.data = source_param.data
                            visited_indices.add(idx)
                            break
                    else:
                        print(f"Warning: No matching parameter found for {name}.{param_name}")
        elif isinstance(module,nn.BatchNorm2d):
             for param_name in ["weight","bias","running_mean","running_var"]:
                if hasattr(module, param_name):
                    target_param = getattr(module, param_name)
                    for idx, (source_name, source_param) in enumerate(loaded_state_dict.items()):
                        if (
                            idx not in visited_indices and  
                            target_param.shape == source_param.shape and
                            ("norm" in source_name)
                        ):
                            target_param.data = source_param.data
                            visited_indices.add(idx)  
                            break
                    else:
                        print(f"Warning: No matching parameter found for {name}.{param_name}")
        elif isinstance(module, QLinear):
        # elif isinstance(module, LinearLayer):        # when you check baseline, uncomment this line and comment : isinstance(module,QLinear):
            for param_name in ["weight","bias"]:
                if hasattr(module, param_name):
                    print(param_name)
                    target_param = getattr(module, param_name)
                    for idx, (source_name, source_param) in enumerate(loaded_state_dict.items()):
                        if (
                            idx not in visited_indices and  
                            target_param.shape == source_param.shape  
                        ):
                            target_param.data = source_param.data
                            visited_indices.add(idx)  
                            break
                    else:
                        print(f"Warning: No matching parameter found for {name}.{param_name}")
        elif isinstance(module, nn.Module):  
            recursive_update(module, loaded_state_dict, visited_indices)


if __name__ == "__main__":
    main_worker()




