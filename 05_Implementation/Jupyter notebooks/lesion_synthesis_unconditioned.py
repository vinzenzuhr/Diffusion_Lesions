#!/usr/bin/env python
# coding: utf-8

# In[4]:


#### create config
from dataclasses import dataclass

@dataclass
class TrainingConfig: 
    t1n_target_shape = None # will transform t1n during preprocessing (computationally expensive)
    unet_img_shape = (128,256)
    channels = 1
    train_batch_size = 4 
    eval_batch_size = 4  
    num_sorted_samples = 1
    num_dataloader_workers = 8
    num_epochs = 192
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    evaluate_epochs = 5 # adjust to num_epochs
    evaluate_num_batches = -1 # ~3s/batch. 2.5 min/Evaluation 3D epoch with all batchesr
    evaluate_num_batches_3d = -1 
    deactivate3Devaluation = True
    evaluate_3D_epochs = 1000  # 3 min/Evaluation 3D 
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
    add_lesion_technique = "other_lesions_median" # 'mean_intensity', 'other_lesions_1stQuantile', 'other_lesions_mean', 'other_lesions_median', 'other_lesions_3rdQuantile', 'other_lesions_99Quantile'
    add_lesion_mean_intensity = -0.5492 
    intermediate_timestep = 3 # starting from this timesteps. num_inference_steps means the whole pipeline and 1 the last step. 
    mode = "tuning_parameters" # 'train', 'eval' or "tuning_parameters"
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
    config.num_dataloader_workers = 1


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
datasetTrain = DatasetMRI2D(root_dir_img=Path(config.dataset_train_path), root_dir_segm=Path(config.segm_train_path), connected_masks=config.train_only_connected_masks, target_shape=config.t1n_target_shape, transforms=transformations)
datasetEvaluation = DatasetMRI2D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), root_dir_synthesis=Path(config.masks_eval_path), connected_masks=config.eval_only_connected_masks, target_shape=config.t1n_target_shape)
dataset3DEvaluation = DatasetMRI3D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), root_dir_synthesis=Path(config.masks_eval_path), connected_masks=config.eval_only_connected_masks, target_shape=config.t1n_target_shape)


# ### Finding good intensity of lesions

# In[6]:


from pathlib import Path
import nibabel as nib
from tqdm.auto import tqdm 
from custom_modules import DatasetMRI2D, DatasetMRI3D, ScaleDecorator

#skip to speed up
if False:
    #t1w_norm_noskull_list = list(Path("temp/unhealthy_DL+DiReCT_Segmentation").rglob("*T1w_norm_noskull.nii.gz"))
    #lesion_list = list(Path("temp/unhealthy_registered_lesions").rglob("*transformed_lesion.nii.gz"))
    #t1w_norm_noskull_list = list(Path("datasets_full/unhealthy_flair").rglob("*3DFLAIR.nii.gz"))
    #lesion_list = list(Path("datasets_full/unhealthy_flair_masks").rglob("*Consensus.nii.gz"))

    dataset_eval_path="datasets_full/unhealthy_flair"
    masks_eval_path="datasets_full/unhealthy_flair_masks"
    
    dataset3DEvaluation = DatasetMRI3D(root_dir_img=Path(dataset_eval_path), root_dir_masks=Path(masks_eval_path), root_dir_synthesis=Path(masks_eval_path), connected_masks=config.eval_only_connected_masks, target_shape=config.t1n_target_shape)
    
    t1ws = list()
    lesions = list()
    for i in tqdm(range(len(dataset3DEvaluation))):
        t1w = dataset3DEvaluation[i]["gt_image"]
        lesion = dataset3DEvaluation[i]["mask"]
        
        #t1w = t1w.get_fdata()
        #t1w, _, lesion, _, _ = DatasetMRI3D.preprocess(t1w, masks=lesion)
        #lesion = lesion.get_fdata()
        #lesion = DatasetMRI3D._padding(torch.from_numpy(lesion).to(torch.uint8))
        #means.append(t1w[lesion.to(torch.bool)].mean())
        #stds.append(t1w[lesion.to(torch.bool)].std())
        t1ws.append(t1w)
        lesions.append(lesion)
    t1w_big = torch.cat(t1ws)
    lesion_big = torch.cat(lesions)
    
    print("mean lesion intensity: ", t1w_big[lesion_big.to(torch.bool)].mean()) # -0.3572
    print("std lesion intensity: ", t1w_big[lesion_big.to(torch.bool)].std()) # 0.1829
    print("median lesion intensity: ", t1w_big[lesion_big.to(torch.bool)].median()) # 0.1829

 


# In[6]:


import matplotlib.pyplot as plt 
import random 
 

#skip to speed up
if False: 
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


# ### Playground LPIPS

# In[7]:


#skip to speed up
if False:
    from Evaluation2DSynthesis import Evaluation2DSynthesis
    from accelerate import Accelerator
    from DDIMGuidedPipeline import DDIMGuidedPipeline
    
    pipeline = DDIMGuidedPipeline.from_pretrained(config.output_dir) 
    
    eval = Evaluation2DUnconditionalSynthesis(config, pipeline, None, None, Accelerator(), None)
    images_with_lesions, _ = eval._add_coarse_lesions(datasetEvaluation[123]["gt_image"], datasetEvaluation[123])
    clean_image=datasetEvaluation[123]["gt_image"]
    synthesized_images = pipeline(images_with_lesions, 1).images
    synthesized_images_2 = pipeline(images_with_lesions, 3).images
    print("Pips for one timestep: ", eval._calc_lpip(clean_image.unsqueeze(0), synthesized_images))
    print("Pips for three timestep: ", eval._calc_lpip(clean_image.unsqueeze(0), synthesized_images_2))


# In[8]:


#skip to speed up
if False:
    fig, axis = plt.subplots(1,4, figsize=(24,12))
    axis[0].imshow(clean_image.squeeze()/2+0.5)
    axis[0].set_title("original image")
    axis[1].imshow(images_with_lesions.squeeze()/2+0.5)
    axis[1].set_title("coarse lesions")
    axis[2].imshow(synthesized_images.squeeze()/2+0.5)
    axis[2].set_title("Diffusion with one timestep")
    axis[3].imshow(synthesized_images_2.squeeze()/2+0.5)
    axis[3].set_title("Diffusion with three timestep")


# In[9]:


#skip to speed up
if False:
    diff_1=clean_image-synthesized_images
    diff_3=clean_image-synthesized_images_2
    fig, axis = plt.subplots(1,2, figsize=(16,8))
    axis[0].imshow(diff_1.squeeze()/2+0.5)
    axis[0].set_title("Differences with one timestep")
    axis[1].imshow(diff_3.squeeze()/2+0.5)
    axis[1].set_title("Differences with three timestep")


# ### Training

# In[10]:


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


# In[11]:


#setup noise scheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


# In[12]:


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


# In[13]:


from custom_modules import TrainingUnconditional, GuidedRePaintPipeline, Evaluation2DUnconditionalSynthesis, Evaluation3DUnconditionalSynthesis, DDIMGuidedPipeline
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
    "evaluation2D": Evaluation2DUnconditionalSynthesis,
    "evaluation3D": Evaluation3DUnconditionalSynthesis, 
    "pipelineFactory": PipelineFactories.get_ddim_guided_pipeline, 
    "deactivate3Devaluation": config.deactivate3Devaluation} 
trainingSynthesis = TrainingUnconditional(**args)


# In[ ]:


if config.mode == "train":
    trainingSynthesis.train()


# In[ ]:


if config.mode == "eval":
    trainingSynthesis.config.deactivate3Devaluation = False
    pipeline = GuidedRePaintPipeline.from_pretrained(config.output_dir) 
    trainingSynthesis.evaluate(pipeline, deactivate_save_model=True)


# In[ ]:


import os
import csv
from itertools import product

if config.mode == "tuning_parameters":
    trainingSynthesis.config.deactivate3Devaluation = False
    pipeline = DDIMGuidedPipeline.from_pretrained(config.output_dir)
    original_output_dir = config.output_dir 
    timesteps = [30, 40, 45, 48, 50]
    
    for timestep in timesteps: 
        trainingSynthesis.config.intermediate_timestep = timestep 
        trainingSynthesis.config.output_dir = original_output_dir + "/tuning_parameters/timestep_" + str(timestep)
        if trainingSynthesis.accelerator.is_main_process:
            os.makedirs(config.output_dir, exist_ok=True)
            trainingSynthesis.log_meta_logs()
        trainingSynthesis.evaluate(pipeline, deactivate_save_model=True) 

# In[ ]:
