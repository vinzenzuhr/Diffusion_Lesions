#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, './custom_modules')


# In[2]:


#### create config
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256  # TODO: the generated image resolution 
    channels = 1
    train_batch_size = 4 
    eval_batch_size = 4  
    num_epochs = 350
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    evaluate_epochs = 40 # anpassen auf Anzahl epochs
    evaluate_num_batches = 2 # one batch needs ~130s 
    deactivate3Devaluation = True
    evaluate_3D_epochs = 1000  # one 3D evaluation has 77 slices and needs 166min
    save_model_epochs = 300
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-synthesis-256-unconditioned"  # the model name locally and on the HF Hub
    dataset_train_path = "./datasets/synthesis/dataset_train/imgs"
    segm_train_path = "./datasets/synthesis/dataset_train/segm"
    masks_train_path = "./datasets/synthesis/dataset_train/masks"
    dataset_eval_path = "./datasets/synthesis/dataset_eval/imgs"
    segm_eval_path = "./datasets/synthesis/dataset_eval/segm"
    masks_eval_path = "./datasets/synthesis/dataset_eval/masks"
    train_only_connected_masks=False  # No Training with lesion masks
    eval_only_connected_masks=False 
    num_inference_steps=50
    mode = "train" # train / eval
    debug = True
    #uniform_dataset_path = "./uniform_dataset"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    #hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = False
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()


# In[3]:


if config.debug:
    config.num_inference_steps = 1
    config.train_batch_size = 1
    config.eval_batch_size = 1 
    config.train_only_connected_masks=False
    config.eval_only_connected_masks=False
    config.evaluate_num_batches = 1
    #dataset_train_path = "./dataset_eval/imgs"
    #segm_train_path = "./dataset_eval/segm"
    #masks_train_path = "./dataset_eval/masks"   


# In[4]:


#setup huggingface accelerate
import torch
import numpy as np
import accelerate
accelerate.commands.config.default.write_basic_config(config.mixed_precision)


# In[5]:


from DatasetMRI2D import DatasetMRI2D
from DatasetMRI3D import DatasetMRI3D
from pathlib import Path

#create dataset
datasetTrain = DatasetMRI2D(root_dir_img=Path(config.dataset_train_path), root_dir_segm=Path(config.segm_train_path), only_connected_masks=config.train_only_connected_masks, directDL=False)
datasetEvaluation = DatasetMRI2D(root_dir_img=Path(config.dataset_eval_path), root_dir_segm=Path(config.segm_eval_path), root_dir_masks=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, directDL=False)
dataset3DEvaluation = DatasetMRI3D(root_dir_img=Path(config.dataset_eval_path), root_dir_segm=Path(config.segm_eval_path), root_dir_masks=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, directDL=False)


# In[6]:


#create model
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=config.channels,  # the number of input channels, 3 for RGB images
    out_channels=config.channels,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

config.model = "UNet2DModel"


# In[7]:


#setup noise scheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


# In[8]:


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


# In[9]:


from TrainingUnconditional import TrainingUnconditional
from DDIMGuidedPipeline import DDIMGuidedPipeline
from Evaluation2DSynthesis import Evaluation2DSynthesis
from Evaluation3DSynthesis import Evaluation3DSynthesis 
import PipelineFactories

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
    "pipelineFactory": PipelineFactories.get_ddim_guided_pipeline,
    "deactivate3Devaluation": config.deactivate3Devaluation}
trainingRepaint = TrainingUnconditional(**args)


# In[ ]:


if config.mode == "train":
    trainingRepaint.train()


# In[ ]:


if config.mode == "eval":
    pipeline = DDIMGuidedPipeline.from_pretrained(config.output_dir) 
    trainingRepaint.evaluate(pipeline)


# In[ ]:


print("Finished Training")


# In[1]:
