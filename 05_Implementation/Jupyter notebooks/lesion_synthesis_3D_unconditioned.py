#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### create config
from dataclasses import dataclass

@dataclass
class TrainingConfig: 
    t1n_target_shape = None # will transform t1n during preprocessing (computationally expensive)
    unet_img_shape = (128,256)
    channels = 1
    train_batch_size = 1
    eval_batch_size = 1 
    num_sorted_samples = 3
    num_epochs = 16
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    evaluate_epochs = 1 # adjust to num_epochs
    evaluate_num_batches = 2 # ~3s/batch. 2.5 min/Evaluation 3D epoch with all batchesr
    evaluate_num_batches_3d = -1 
    deactivate3Devaluation = True
    evaluate_3D_epochs = 1000  # 3 min/Evaluation 3D 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-synthesis-3D"  # the model name locally and on the HF Hub
    dataset_train_path = "./datasets/synthesis/dataset_train/imgs"
    segm_train_path = "./datasets/synthesis/dataset_train/segm"
    masks_train_path = "./datasets/synthesis/dataset_train/masks"
    dataset_eval_path = "./datasets/synthesis/dataset_eval/imgs"
    segm_eval_path = "./datasets/synthesis/dataset_eval/segm"
    masks_eval_path = "./datasets/synthesis/dataset_eval/masks"
    train_only_connected_masks=False  # No Training with lesion masks
    eval_only_connected_masks=False 
    num_inference_steps=50
    log_csv = True
    add_lesion_technique = "other_lesions_99Quantile" # 'mean_intensity', 'other_lesions_1stQuantile', 'other_lesions_mean', 'other_lesions_median', 'other_lesions_3rdQuantile', or 
    add_lesion_mean_intensity = -0.5492 
    intermediate_timestep = 3 # starting from this timesteps. num_inference_steps means the whole pipeline and 1 the last step. 
    mode = "eval" # 'train', 'eval' or "tuning_parameters"
    debug = False     
    jump_length=intermediate_timestep
    jump_n_sample=3
    brightness_augmentation = False

    push_to_hub = False  # whether to upload the saved model to the HF Hub 
    seed = 0
config = TrainingConfig()


# In[2]:


if config.debug:
    config.num_inference_steps = 1
    config.intermediate_timestep = 1
    config.train_batch_size = 1
    config.eval_batch_size = 1
    config.train_only_connected_masks=False
    config.eval_only_connected_masks=False
    config.evaluate_num_batches = 3
    config.deactivate3Devaluation = False
    config.dataset_train_path = "./datasets/synthesis/dataset_eval/imgs"
    config.segm_train_path = "./datasets/synthesis/dataset_eval/segm"
    config.masks_train_path = "./datasets/synthesis/dataset_eval/masks" 
    config.num_sorted_samples = 1


# In[3]:


#setup huggingface accelerate
import torch
import numpy as np
import accelerate
accelerate.commands.config.default.write_basic_config(config.mixed_precision)


# In[4]:


from custom_modules import DatasetMRI2D, DatasetMRI3D, ScaleDecorator
from pathlib import Path
from torchvision import transforms 

#add augmentation
transformations = None
if config.brightness_augmentation:
    transformations = transforms.RandomApply([ScaleDecorator(transforms.ColorJitter(brightness=1))], p=0.5)

#create dataset
datasetTrain = DatasetMRI2D(root_dir_img=Path(config.dataset_train_path), root_dir_segm=Path(config.segm_train_path), only_connected_masks=config.train_only_connected_masks, t1n_target_shape=config.t1n_target_shape, num_sorted_samples=config.num_sorted_samples, transforms=transformations)
datasetEvaluation = DatasetMRI2D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), root_dir_synthesis=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, t1n_target_shape=config.t1n_target_shape, num_sorted_samples=config.num_sorted_samples)
dataset3DEvaluation = DatasetMRI3D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), root_dir_synthesis=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, t1n_target_shape=config.t1n_target_shape)


# ### Training

# In[5]:


#create model
from custom_modules import UNet2DModel

model = UNet2DModel(
    sample_size=config.unet_img_shape,  # the target image resolution
    in_channels=config.channels,  # the number of input channels, 3 for RGB images
    out_channels=config.channels,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "Pseudo3DDownBlock2D",  # a regular ResNet downsampling block
        "Pseudo3DDownBlock2D",
        "Pseudo3DDownBlock2D",
        "Pseudo3DDownBlock2D",
        "Pseudo3DAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "Pseudo3DDownBlock2D",
    ),
    up_block_types=(
        "Pseudo3DUpBlock2D",  # a regular ResNet upsampling block
        "Pseudo3DAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "Pseudo3DUpBlock2D",
        "Pseudo3DUpBlock2D",
        "Pseudo3DUpBlock2D",
        "Pseudo3DUpBlock2D",
    ),
)

config.model = "Pseudo3DUNet2DModel"


# In[6]:


#setup noise scheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


# In[7]:


# setup lr scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import math

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(math.ceil(len(datasetTrain)/config.train_batch_size) * config.num_epochs), # num_iterations per epoch * num_epochs
)
config.lr_scheduler = "cosine_schedule_with_warmup"


# In[8]:


from custom_modules import TrainingUnconditional, GuidedRePaintPipeline, Evaluation2DSynthesis, Evaluation3DSynthesis 
from custom_modules import PipelineFactories

config.conditional_data = "None"

args = {
    "config": config, 
    "model": model, 
    "noise_scheduler": noise_scheduler, 
    "optimizer": optimizer, 
    "lr_scheduler": lr_scheduler, 
    "datasetTrain": datasetTrain, 
    "datasetEvaluation": datasetEvaluation, 
    "dataset3DEvaluation": dataset3DEvaluation, 
    "evaluation2D": Evaluation2DSynthesis,
    "evaluation3D": Evaluation3DSynthesis, 
    "pipelineFactory": PipelineFactories.get_guided_repaint_pipeline, 
    "multi_sample": config.num_sorted_samples > 1,
    "deactivate3Devaluation": config.deactivate3Devaluation,
    "evaluation_pipeline_parameters": {
                "jump_length": config.jump_length,
                "jump_n_sample": config.jump_n_sample,
            }} 
trainingSynthesis = TrainingUnconditional(**args)


# In[ ]:


if config.mode == "train":
    trainingSynthesis.train()


# In[ ]:


if config.mode == "eval":
    trainingSynthesis.config.deactivate3Devaluation = False
    pipeline = GuidedRePaintPipeline.from_pretrained(config.output_dir) 
    trainingSynthesis.evaluate(pipeline)


# In[ ]:


import os
import csv
from itertools import product

if config.mode == "tuning_parameters":
    trainingSynthesis.config.deactivate3Devaluation = False
    pipeline = GuidedRePaintPipeline.from_pretrained(config.output_dir)
    original_output_dir = config.output_dir 
    timesteps = [1, 3, 5]
    resample_step = [1, 3, 5]
    parameters = product(timesteps, resample_step) 
    
    for timestep, resample_step in parameters:
        print("Begin evaluation for timestep ", timestep, " and resample step ", resample_step)
        trainingSynthesis.config.intermediate_timestep = timestep
        trainingSynthesis.config.jump_length = timestep
        trainingSynthesis.config.jump_n_sample = resample_step
        trainingSynthesis.config.output_dir = original_output_dir + "/tuning_parameters/timestep_" + str(timestep) + "_resample_" + str(resample_step)
        if trainingSynthesis.accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            trainingSynthesis.log_meta_logs()
        trainingSynthesis.evaluate(pipeline, deactivate_save_model=True) 

    # plot lpips score vs  
    folder_list = list(Path().glob(original_output_dir + "/tuning_parameters/timestep_*"))
    lpips = []
    labels = [] 
    for folder in folder_list:
        timestep = str(folder).split("_")[-3] 
        resample_step = str(folder).split("_")[-1]
        with open(folder / "metrics.csv", 'r') as fp: 
            _ = fp.readline()
            csv_metrics = fp.readline()
            reader = csv_metrics.split(',') 
            for metric in reader:
                if metric != "":
                    name, value = metric.split(':')
                    if name == "lpips":
                        lpips.append(float(value))
                        labels.append(timestep + "_" + resample_step) 
    plt.clf()
    plt.bar(labels, lpips) 
    plt.savefig(original_output_dir + '/lpips_parameters.png')


# In[1]:
