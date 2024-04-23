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
    image_size = 256
    channels = 1
    train_batch_size = 4 
    eval_batch_size = 4  
    num_epochs = 170
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    evaluate_epochs = 5 # adjust to num_epochs
    evaluate_num_batches = -1 # ~3s/batch. 2.5 min/Evaluation 3D epoch with all batchesr
    deactivate3Devaluation = False
    evaluate_3D_epochs = 1000  # 3 min/Evaluation 3D
    save_model_epochs = 100
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
    log_csv = True
    add_lesion_technique = "mean_intensity" # 'mean_intensity' or 'other_lesions'
    intermediate_timestep = 3 # starting from this timesteps. num_inference_steps means the whole pipeline and 1 the last step. 
    mode = "tuning_timestep" # 'train', 'eval' or "tuning_timestep"
    debug = False 

    push_to_hub = False  # whether to upload the saved model to the HF Hub 
    seed = 0
config = TrainingConfig()


# In[3]:


if config.debug:
    config.num_inference_steps = 1
    config.intermediate_timestep = 1
    config.train_batch_size = 1
    config.eval_batch_size = 1
    config.train_only_connected_masks=False
    config.eval_only_connected_masks=False
    config.evaluate_num_batches = 3
    config.deactivate3Devaluation = False
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
datasetEvaluation = DatasetMRI2D(root_dir_img=Path(config.dataset_eval_path), root_dir_segm=Path(config.segm_eval_path), root_dir_masks=Path(config.masks_eval_path), root_dir_synthesis=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, directDL=False)
dataset3DEvaluation = DatasetMRI3D(root_dir_img=Path(config.dataset_eval_path), root_dir_segm=Path(config.segm_eval_path), root_dir_masks=Path(config.masks_eval_path), root_dir_synthesis=Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks, directDL=False)


# ### Finding good intensity of lesions

# In[6]:


from pathlib import Path
import nibabel as nib
from tqdm.auto import tqdm
from DatasetMRI3D import DatasetMRI3D 

#skip to speed up
if False:
    t1w_norm_noskull_list = list(Path("temp/unhealthy_DL+DiReCT_Segmentation").rglob("*T1w_norm_noskull.nii.gz"))
    lesion_list = list(Path("temp/unhealthy_registered_lesions").rglob("*transformed_lesion.nii.gz"))
    
    t1ws = list()
    lesions = list()
    for i in tqdm(range(len(t1w_norm_noskull_list))):
        t1w = nib.load(t1w_norm_noskull_list[i])
        t1w = t1w.get_fdata()
        t1w, _ = DatasetMRI3D.preprocess(t1w)
        lesion = nib.load(lesion_list[i])
        lesion = lesion.get_fdata()
        lesion = DatasetMRI3D._padding(torch.from_numpy(lesion).to(torch.uint8))
        #means.append(t1w[lesion.to(torch.bool)].mean())
        #stds.append(t1w[lesion.to(torch.bool)].std())
        t1ws.append(t1w)
        lesions.append(lesion)
    t1w_big = torch.cat(t1ws)
    lesion_big = torch.cat(lesions)
    
    print("mean lesion intensity: ", t1w_big[lesion_big.to(torch.bool)].mean()) # -0.3572
    print("std lesion intensity: ", t1w_big[lesion_big.to(torch.bool)].std()) # 0.1829
    print("median lesion intensity: ", t1w_big[lesion_big.to(torch.bool)].median()) # 0.1829


# In[7]:


import matplotlib.pyplot as plt 
import random 
 
torch.manual_seed(7) 
random_idx = torch.randint(len(datasetEvaluation)-1, size=(1,)).item()

fig, axis = plt.subplots(1,4, figsize=(16,4)) 
img = datasetEvaluation[random_idx]["gt_image"].squeeze()
img[datasetEvaluation[random_idx]["mask"].to(torch.bool).squeeze()] = -0.5492 # mean-std of lesion intensity
axis[0].imshow(datasetEvaluation[random_idx]["gt_image"].squeeze()/2+0.5)
axis[0].set_axis_off()
axis[1].imshow(datasetEvaluation[random_idx]["mask"].squeeze())
axis[1].set_axis_off()
axis[2].imshow(img/2+0.5)
axis[2].set_axis_off()
axis[3].imshow(datasetEvaluation[random_idx]["synthesis"].squeeze())
axis[3].set_axis_off()
fig.show()


# ### Training

# In[8]:


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


# In[9]:


#setup noise scheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


# In[10]:


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


# In[11]:


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
trainingSynthesis = TrainingUnconditional(**args) 


# In[12]:


if config.mode == "train":
    trainingSynthesis.train()


# In[ ]:


if config.mode == "eval":
    pipeline = DDIMGuidedPipeline.from_pretrained(config.output_dir) 
    trainingSynthesis.evaluate(pipeline)


# In[ ]:


import os

if config.mode == "tuning_timestep":
    pipeline = DDIMGuidedPipeline.from_pretrained(config.output_dir)
    original_output_dir = config.output_dir 
    timesteps = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20]
    for t in timesteps:
        config.intermediate_timestep = t
        config.output_dir = original_output_dir + "_" + str(t)
        trainingSynthesis.config = config
        if trainingSynthesis.accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            trainingSynthesis.log_meta_logs()
        trainingSynthesis.evaluate(pipeline) 


# In[ ]:


print("Finished Training")


# In[ ]:


#create python script for ubelix 
import os

get_ipython().system('jupyter nbconvert --to script "lesion_synthesis_unconditioned.ipynb"')
