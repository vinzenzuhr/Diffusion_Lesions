#!/usr/bin/env python
# coding: utf-8

# In[1]:


#### create config
from dataclasses import dataclass

@dataclass
class TrainingConfig: 
    t1n_target_shape = None # will transform t1n during preprocessing (computationally expensive)
    unet_img_shape = (256,256)
    channels = 1
    effective_train_batch_size = 32
    train_batch_size = 1
    eval_batch_size = 1  
    num_dataloader_workers = 8
    num_epochs = 300 # 4.5s/iter, 21 min/epoch
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    evaluate_epochs = 15 # anpassen auf Anzahl epochs
    evaluate_num_batches = 4 # one batch needs 4 min 
    evaluate_num_batches_3d = -1 
    deactivate2Devaluation = False
    deactivate3Devaluation = True
    evaluate_3D_epochs = 1000  # one 3D evaluation has 77 slices and needs 166min 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-filling-3D-repaint"  # the model name locally and on the HF Hub
    dataset_train_path = "./datasets/filling/dataset_train/imgs"
    segm_train_path = "./datasets/filling/dataset_train/segm"
    masks_train_path = "./datasets/filling/dataset_train/masks"
    dataset_eval_path = "./datasets/filling/dataset_eval/imgs"
    segm_eval_path = "./datasets/filling/dataset_eval/segm"
    masks_eval_path = "./datasets/filling/dataset_eval/masks"  
    train_only_connected_masks=False  # No Training with lesion masks
    eval_only_connected_masks=False 
    num_inference_steps=50 
    log_csv = False
    mode = "train" # train / eval
    debug = False
    jump_length=8
    jump_n_sample=10 
    brightness_augmentation = False
    #uniform_dataset_path = "./uniform_dataset"
    eval_loss_timesteps=[20,80,140,200,260,320,380,440,560,620,680,740,800,860,920,980]

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    #hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = False
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()


# In[14]:


#setup huggingface accelerate
import torch
import numpy as np
import accelerate
accelerate.commands.config.default.write_basic_config(config.mixed_precision)
#if there are problems with ports then add manually "main_process_port: 0" or another number to yaml file


# In[15]:


from pathlib import Path
import json
with open(Path.home() / ".cache/huggingface/accelerate/default_config.yaml") as f:
    data = json.load(f)
    config.num_processes = data["num_processes"]


# In[16]:


config.num_sorted_samples = int((config.effective_train_batch_size / config.gradient_accumulation_steps) / config.num_processes)


# In[2]:


if config.debug:
    config.num_inference_steps=1
    config.train_batch_size = 1
    config.eval_batch_size = 1 
    config.eval_loss_timesteps = [20]
    config.train_only_connected_masks=False
    config.eval_only_connected_masks=False
    config.evaluate_num_batches=1
    config.dataset_train_path = "./datasets/filling/dataset_eval/imgs"
    config.segm_train_path = "./datasets/filling/dataset_eval/segm"
    config.masks_train_path = "./datasets/filling/dataset_eval/masks"  
    config.num_sorted_samples=1
    config.num_dataloader_workers = 1


# In[17]:


print(f"Start training with batch size {config.train_batch_size}, {config.gradient_accumulation_steps} accumulation steps and {config.num_processes} process(es)")


# In[4]:


from custom_modules import DatasetMRI2D, DatasetMRI3D, ScaleDecorator
from pathlib import Path
from torchvision import transforms 

#add augmentation
transformations = None
if config.brightness_augmentation:
    transformations = transforms.RandomApply([ScaleDecorator(transforms.ColorJitter(brightness=1))], p=0.5)

#create dataset
datasetTrain = DatasetMRI2D(root_dir_img=Path(config.dataset_train_path), root_dir_segm=Path(config.segm_train_path), only_connected_masks=config.train_only_connected_masks, t1n_target_shape =config.t1n_target_shape, num_sorted_samples=config.num_sorted_samples, transforms=transformations, random_sorted_samples=True)
datasetEvaluation = DatasetMRI2D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, t1n_target_shape =config.t1n_target_shape, num_sorted_samples=config.num_sorted_samples)
dataset3DEvaluation = DatasetMRI3D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, t1n_target_shape =config.t1n_target_shape)


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


from custom_modules import TrainingUnconditional, RePaintPipeline, Evaluation2DFilling, Evaluation3DFilling 
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
    "evaluation2D": Evaluation2DFilling,
    "evaluation3D": Evaluation3DFilling, 
    "pipelineFactory": PipelineFactories.get_repaint_pipeline,
    "multi_sample": config.num_sorted_samples > 1,
    "deactivate3Devaluation": config.deactivate3Devaluation,
    "evaluation_pipeline_parameters": {
                "jump_length": config.jump_length,
                "jump_n_sample": config.jump_n_sample,
            }} 

training3Dlesions = TrainingUnconditional(**args)


# In[ ]:


if config.mode == "train":
    training3Dlesions.train()


# In[ ]:


if config.mode == "eval":
    training3Dlesions.config.deactivate3Devaluation = False
    pipeline = RePaintPipeline.from_pretrained(config.output_dir) 
    training3Dlesions.evaluate(pipeline, deactivate_save_model=True)


# In[ ]:


print("Finished Training")


# In[ ]:
