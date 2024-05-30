#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### create config
from dataclasses import dataclass

@dataclass
class TrainingConfig:  
    unet_img_shape = (256,256)
    channels = 1
    eval_batch_size = 4  
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-filling-256-repaint"  # the model name locally and on the HF Hub
    num_inference_steps=50 
config = TrainingConfig()


# In[2]:



#setup huggingface accelerate
import torch
import numpy as np
import accelerate
accelerate.commands.config.default.write_basic_config(config.mixed_precision)

#create model
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.unet_img_shape,  # the target image resolution
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


# In[6]:


#setup noise scheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


from diffusers import DDIMPipeline
from diffusers.utils import make_image_grid
import os
from accelerate import Accelerator
pipeline = DDIMPipeline.from_pretrained(config.output_dir) 

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
) 

pipeline = accelerator.prepare(pipeline)
pipeline.to(accelerator.device)


image_list = []
for i in torch.arange(10):
    images = pipeline(batch_size=4, num_inference_steps=config.num_inference_steps, return_dict=False)[0]
    image_list.append(images)

path=config.output_dir+"/unconditional_images"
os.makedirs(path, exist_ok=True) 
for idx, images in enumerate(image_list):
    image_grid = make_image_grid(images, rows=2, cols=2)
    image_grid.save(f"{path}/images_{idx}.png")
print("image saved")

