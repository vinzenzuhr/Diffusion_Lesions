#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# In this jupyter notebook we're filling (removing) MS lesions. We're training a unet model with pseudo3D resnet blocks which conditions on a binary mask and the voided image. For training it uses a mixture of random circle masks and lesions masks annotated by doctors.

# ### Configuration

# In[1]:


from dataclasses import dataclass

@dataclass
class Config: 
    mode = "train" # ['train', 'eval']
    debug = False
    output_dir = "lesion-filling-3D-conditioned-circles" 

    #dataset config
    dataset_train_path = "./datasets/filling/dataset_train/imgs"
    segm_train_path = "./datasets/filling/dataset_train/segm"
    masks_train_path = "./datasets/filling/dataset_train/masks"
    dataset_eval_path = "./datasets/filling/dataset_eval/imgs"
    segm_eval_path = "./datasets/filling/dataset_eval/segm"
    masks_eval_path = "./datasets/filling/dataset_eval/masks"  
    target_shape = None # During preprocessing the img gets transformered to this shape (computationally expensive) 
    unet_img_shape = (256,256) # This defines the input layer of the model
    channels = 3 # the number of input channels: 1 for grayscale img
    restrict_train_slices = "segm" # Defines which 2D slices are used from the 3D MRI ['mask', 'segm', or 'unrestricted']
    restrict_eval_slices = "mask" # Defines which 2D slices are used from the 3D MRI ['mask', 'segm', or 'unrestricted']
    restrict_mask_to_wm = False # Restricts lesion masks to white matter based on segmentation
    proportion_training_circular_masks = 1.0 # Defines if random circular masks should be used instead of the provided lesion masks. 
                                             # 1 is 100% random circular masks and 0 is 100% lesion masks.
    uniform_center_circular_masks = True # the center of the circular mask is uniform within a batch
    train_connected_masks = True # The distribution of the masks is extended by splitting the masks into several smaller connected components.  	
    brightness_augmentation = True	# The training data gets augmented with randomly applied ColorJitter. 
    num_dataloader_workers = 8 # how many subprocesses to use for data loading

    # train config 
    num_epochs = 12 
    sorted_slice_sample_size = None # The number of sorted slices within one sample. Defaults to 1.
                                    # This is needed for the pseudo3Dmodels, where the model expects that the slices within one batch
                                    # are next to each other in the 3D volume.
    train_batch_size = 1
    effective_train_batch_size = 32 # The train_batch_size gets recalculated to this batch size based on accumulation_steps and number of GPU's.
	                                # For pseudo3D models the sorted_slice_sample_size gets calculcated to this batch size. 
                                    # The train_batch_size and eval_batch_size should be 1.
    eval_batch_size = 1 
    learning_rate = 1e-4
    lr_warmup_steps = 500
    use_min_snr_loss = False
    snr_gamma = 5 
    gradient_accumulation_steps = 1
    mixed_precision = "fp16" # `no` for float32, `fp16` for automatic mixed precision 

    # evaluation config
    num_inference_steps=50 
    evaluate_2D_epochs = 0.3 # The interval at which to evaluate the model on 2D images. 
    evaluate_3D_epochs = 1000 # The interval at which to evaluate the model on 3D images.  
    evaluate_num_batches = 15 # Number of batches used for evaluation. -1 means all batches. 
    evaluate_num_batches_3d = -1 # Number of batches used for evaluation. -1 means all batches.   
    evaluate_unconditional_img = False # Used for unconditional models to generate some samples without the repaint pipeline. 
    deactivate_2D_evaluation = False
    deactivate_3D_evaluation = True
    img3D_filename = "T1" # Filename to save the processed 3D images 
    eval_loss_timesteps = [20,80,140,200,260,320,380,440,560,620,680,740,800,860,920,980] # List of timesteps to evalute validation loss.
    eval_mask_dilation = 0 # dilation value for masks
	#add_lesion_technique = "other_lesions_99Quantile" # Used for synthesis only. 
                                                      # ['empty', 'mean_intensity', 'other_lesions_1stQuantile', 'other_lesions_mean', 
                                                      # 'other_lesions_median', 'other_lesions_3rdQuantile', 'other_lesions_99Quantile'] 
    #intermediate_timestep = 3 # Used for synthesis only. Diffusion process starts from this timesteps. 
                               # Num_inference_steps means the whole pipeline and 1 the last step. 
    #jump_length = 8 # Used for unconditional lesion filling only. Defines the jump_length from the repaint paper.
    #jump_n_sample = 10 # Used for unconditional lesion filling only. Defines the jump_n_sample from the repaint paper.
    log_csv = False # saves evaluation metrics as csv 
    seed = 0 # used for dataloader sampling and generation of the initial noise to start the diffusion process
config = Config()


# In[2]:


import torch
import numpy as np
import accelerate
accelerate.commands.config.default.write_basic_config(config.mixed_precision)
#if there are problems with ports then add manually "main_process_port: 0" or another number to yaml file


# In[3]:


from pathlib import Path
import json
with open(Path.home() / ".cache/huggingface/accelerate/default_config.yaml") as f:
    data = json.load(f)
    config.num_processes = data["num_processes"]


# In[4]:


config.sorted_slice_sample_size = int((config.effective_train_batch_size / config.gradient_accumulation_steps) / config.num_processes)


# In[5]:


if config.debug:
    config.num_inference_steps = 1
    config.train_batch_size = 1
    config.eval_batch_size = 1 
    config.eval_loss_timesteps = [20]
    config.train_connected_masks = False
    config.eval_connected_masks = False
    config.evaluate_num_batches = 1
    config.dataset_train_path = "./datasets/filling/dataset_eval/imgs"
    config.segm_train_path = "./datasets/filling/dataset_eval/segm"
    config.masks_train_path = "./datasets/filling/dataset_eval/masks"  
    config.sorted_slice_sample_size = 1
    config.num_dataloader_workers = 1


# In[6]:


print(f"Start training with batch size {config.sorted_slice_sample_size}, {config.gradient_accumulation_steps} accumulation steps and {config.num_processes} process(es)")


# ### Datasets

# In[7]:


from custom_modules import DatasetMRI2D, DatasetMRI3D, ScaleDecorator
from pathlib import Path
from torchvision import transforms 
 
transformations = None
if config.brightness_augmentation:
    transformations = transforms.RandomApply([ScaleDecorator(transforms.ColorJitter(brightness=1))], p=0.5)
 
dataset_train = DatasetMRI2D(
    root_dir_img=Path(config.dataset_train_path),
    root_dir_segm=Path(config.segm_train_path), 
    restriction=config.restrict_train_slices,   
    proportion_training_circular_masks=config.proportion_training_circular_masks,
    circle_mask_shape=config.unet_img_shape, 
    uniform_mask_center=config.uniform_center_circular_masks,
    connected_masks=config.train_connected_masks, 
    restrict_mask_to_wm=config.restrict_mask_to_wm, 
    transforms=transformations, 
    sorted_slice_sample_size=config.sorted_slice_sample_size, 
    target_shape =config.target_shape, 
)
dataset_evaluation = DatasetMRI2D(
    root_dir_img=Path(config.dataset_eval_path), 
    root_dir_masks=Path(config.masks_eval_path),  
    restriction=config.restrict_eval_slices, 
    dilation=config.eval_mask_dilation, 
    sorted_slice_sample_size=config.sorted_slice_sample_size, 
    target_shape =config.target_shape, 
)
dataset_3D_evaluation = DatasetMRI3D(
    root_dir_img=Path(config.dataset_eval_path), 
    root_dir_masks=Path(config.masks_eval_path),  
    dilation=config.eval_mask_dilation, 
    target_shape =config.target_shape, 
)


# ### Training environement

# In[8]:


from custom_modules import UNet2DModel

model = UNet2DModel(
    sample_size=config.unet_img_shape,  
    in_channels=config.channels,  
    out_channels=1,  
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "Pseudo3DDownBlock2D",  
        "Pseudo3DDownBlock2D",
        "Pseudo3DDownBlock2D",
        "Pseudo3DDownBlock2D",
        "Pseudo3DAttnDownBlock2D", 
        "Pseudo3DDownBlock2D",
    ),
    up_block_types=(
        "Pseudo3DUpBlock2D", 
        "Pseudo3DAttnUpBlock2D", 
        "Pseudo3DUpBlock2D",
        "Pseudo3DUpBlock2D",
        "Pseudo3DUpBlock2D",
        "Pseudo3DUpBlock2D",
    ),
)

config.model = "Pseudo3DUNet2DModel"


# In[9]:


import torch
from PIL import Image
from diffusers import DDIMScheduler

# setup noise scheduler
noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


# In[10]:


from diffusers.optimization import get_cosine_schedule_with_warmup
import math

# setup lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(math.ceil(len(dataset_train)/config.train_batch_size) * config.num_epochs),  
)
config.lr_scheduler = "cosine_schedule_with_warmup"


# In[11]:


from accelerate import Accelerator 

# setup accelerator for distributed training
accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,  
)


# In[12]:


from torch.utils.tensorboard import SummaryWriter
import os
from custom_modules import Logger

# setup tensorboard
if accelerator.is_main_process:
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True) 
    tb_summary = SummaryWriter(config.output_dir, purge_step=0)
    accelerator.init_trackers("train_example") #maybe delete
    logger = Logger(tb_summary, config.log_csv)
    logger.log_config(config)


# In[13]:


from custom_modules import get_dataloader

train_dataloader = get_dataloader(
    dataset=dataset_train, 
    batch_size=config.train_batch_size, 
    num_workers=config.num_dataloader_workers, 
    random_sampler=True, 
    seed=config.seed, 
    multi_slice=config.sorted_slice_sample_size > 1
)
d2_eval_dataloader = get_dataloader(
    dataset=dataset_evaluation, 
    batch_size=config.eval_batch_size, 
    num_workers=config.num_dataloader_workers, 
    random_sampler=False, 
    seed=config.seed, 
    multi_slice=config.sorted_slice_sample_size > 1
)
d3_eval_dataloader = get_dataloader(
    dataset=dataset_3D_evaluation, 
    batch_size=1, 
    num_workers=config.num_dataloader_workers,
    random_sampler=False, 
    seed=config.seed, 
    multi_slice=False
) 


# In[14]:


model, optimizer, train_dataloader, d2_eval_dataloader, d3_eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, d2_eval_dataloader, d3_eval_dataloader, lr_scheduler
)


# In[15]:


from custom_modules import ModelInputGenerator, Evaluation2DFilling, Evaluation3DFilling 

model_input_generator = ModelInputGenerator(conditional=True, noise_scheduler=noise_scheduler)

args = {
    "eval_dataloader": d2_eval_dataloader, 
    "train_dataloader": train_dataloader,
    "logger": None if not accelerator.is_main_process else logger, 
    "accelerator": accelerator,
    "num_inference_steps": config.num_inference_steps,
    "model_input_generator": model_input_generator,
    "output_dir": config.output_dir,
    "eval_loss_timesteps": config.eval_loss_timesteps, 
    "evaluate_num_batches": config.evaluate_num_batches, 
    "seed": config.seed
}
evaluation2D = Evaluation2DFilling(**args)
args = {
    "dataloader": d3_eval_dataloader, 
    "logger": None if not accelerator.is_main_process else logger, 
    "accelerator": accelerator,
    "output_dir": config.output_dir,
    "filename": config.img3D_filename,
    "num_inference_steps": config.num_inference_steps,
    "eval_batch_size": config.eval_batch_size,
    "sorted_slice_sample_size": config.sorted_slice_sample_size,
    "evaluate_num_batches": config.evaluate_num_batches_3d,
    "seed": config.seed,
}
evaluation3D = Evaluation3DFilling(**args)


# ### Start training

# In[16]:


from custom_modules import Training, RePaintPipeline, Evaluation2DFilling, Evaluation3DFilling 
from custom_modules import PipelineFactories

args = { 
    "accelerator": accelerator,
    "model": model, 
    "noise_scheduler": noise_scheduler, 
    "optimizer": optimizer, 
    "lr_scheduler": lr_scheduler, 
    "train_dataloader": train_dataloader, 
    "d2_eval_dataloader": d2_eval_dataloader, 
    "d3_eval_dataloader": d3_eval_dataloader, 
    "model_input_generator": model_input_generator,
    "evaluation2D": evaluation2D,
    "evaluation3D": evaluation3D,
    "logger": None if not accelerator.is_main_process else logger,
    "pipeline_factory": PipelineFactories.get_ddim_inpaint_pipeline,
    "num_epochs": config.num_epochs, 
    "evaluate_2D_epochs": config.evaluate_2D_epochs,
    "evaluate_3D_epochs": config.evaluate_3D_epochs,
    "min_snr_loss": config.use_min_snr_loss,
    "snr_gamma": config.snr_gamma,
    "evaluate_unconditional_img": config.evaluate_unconditional_img,
    "deactivate_2D_evaluation": config.deactivate_2D_evaluation, 
    "deactivate_3D_evaluation": config.deactivate_3D_evaluation, 
    "evaluation_pipeline_parameters": {},
    "debug": config.debug, 
    }

training3Dlesions = Training(**args)


# In[17]:


if config.mode == "train":
    training3Dlesions.train()


# In[ ]:


if config.mode == "eval":
    training3Dlesions.deactivate_3D_evaluation = False
    pipeline = RePaintPipeline.from_pretrained(config.output_dir) 
    training3Dlesions.evaluate(pipeline, deactivate_save_model=True)


# In[ ]:


print("Finished Training")


# ### Save jupyter notebook as python script for hpc

