#!/usr/bin/env python
# coding: utf-8

# In[1]:


#create config
from dataclasses import dataclass

@dataclass
class Config: 
    mode = "train" # ['train', 'eval']
    debug = False
    output_dir = "lesion-filling-256-cond-mixture" 

    #dataset config
    dataset_train_path = "./datasets/filling/dataset_train/imgs"
    segm_train_path = "./datasets/filling/dataset_train/segm"
    masks_train_path = "./datasets/filling/dataset_train/masks"
    dataset_eval_path = "./datasets/filling/dataset_eval/imgs"
    segm_eval_path = "./datasets/filling/dataset_eval/segm"
    masks_eval_path = "./datasets/filling/dataset_eval/masks" 
    target_shape = None # During preprocessing the img gets transformered to this shape (computationally expensive) 
    unet_img_shape = (256,256) # This defines the input layer of the model
    channels = 1 # Number of channels of input layer of the model (1 for grayscale and 3 for RGB)
    restrict_train_slices = "segm" # Defines which 2D slices are used from the 3D MRI ['mask', 'segm', or 'unrestricted']
    restrict_eval_slices = "mask" # Defines which 2D slices are used from the 3D MRI ['mask', 'segm', or 'unrestricted']
    restrict_mask_to_wm=True # Restricts lesion masks to white matter based on segmentation
    proportion_training_circular_masks = 0.5 # Defines if random circular masks should be used instead of the provided lesion masks. 
                                             # 1 is 100% random circular masks and 0 is 100% lesion masks.
    train_connected_masks=True  # The distribution of the masks is extended by splitting the masks into several smaller connected components.  	
    brightness_augmentation = True	# The training data gets augmented with randomly applied ColorJitter. 
    num_dataloader_workers = 8 # how many subprocesses to use for data loading

    # train config 
    num_epochs = 500 
    sorted_slice_sample_size = 1 # The number of sorted slices within one sample. Defaults to 1.
                                 # This is needed for the pseudo3Dmodels, where the model expects that the slices within one batch
                                 # are next to each other in the 3D volume.
    train_batch_size = None
    effective_train_batch_size=32  # The train_batch_size gets recalculated to this batch size based on accumulation_steps and number of GPU's.
	                                # For pseudo3D models the sorted_slice_sample_size gets calculcated to this batch size. 
                                    # The train_batch_size and eval_batch_size should be 1.
    eval_batch_size = 16 
    learning_rate = 1e-4
    lr_warmup_steps = 500 
    use_min_snr_loss=False
    snr_gamma=5
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision 

    # evaluation config
    num_inference_steps=50
    evaluate_2D_epochs = 10 # The interval at which to evaluate the model on 2D images. 
    evaluate_3D_epochs = 1000  # The interval at which to evaluate the model on 3D images.  
    evaluate_num_batches = 20 # Number of batches used for evaluation. -1 means all batches. 
    evaluate_num_batches_3d = 2 # Number of batches used for evaluation. -1 means all batches. 
    evaluate_unconditional_img = False # Used for unconditional models to generate some samples without the repaint pipeline. 
    deactivate_2D_evaluation = False
    deactivate_3D_evaluation = True
    img3D_filename = "T1" # Filename to save the processed 3D images 
    eval_loss_timesteps=[20,80,140,200,260,320,380,440,560,620,680,740,800,860,920,980] # List of timesteps to evalute validation loss.
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


#setup huggingface accelerate
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


config.train_batch_size = int((config.effective_train_batch_size / config.gradient_accumulation_steps) / config.num_processes)


# In[5]:


if config.debug:
    config.num_inference_steps=1
    config.train_batch_size = 1
    config.eval_batch_size = 1 
    config.eval_loss_timesteps = [20]
    config.train_connected_masks=False
    config.eval_connected_masks=False
    config.evaluate_num_batches=1
    config.dataset_train_path = "./datasets/filling/dataset_eval/imgs"
    config.segm_train_path = "./datasets/filling/dataset_eval/segm"
    config.masks_train_path = "./datasets/filling/dataset_eval/masks"
    config.num_dataloader_workers = 1


# In[6]:


print(f"Start training with batch size {config.train_batch_size}, {config.gradient_accumulation_steps} accumulation steps and {config.num_processes} process(es)")


# In[7]:


from custom_modules import DatasetMRI2D, DatasetMRI3D, ScaleDecorator
from pathlib import Path
from torchvision import transforms 

#add augmentation
transformations = None
if config.brightness_augmentation:
    transformations = transforms.RandomApply([ScaleDecorator(transforms.ColorJitter(brightness=1))], p=0.5)

#create dataset
dataset_train = DatasetMRI2D(root_dir_img=Path(config.dataset_train_path), restriction=config.restrict_train_slices, root_dir_segm=Path(config.segm_train_path), root_dir_masks=Path(config.masks_train_path), connected_masks=config.train_connected_masks, target_shape=config.target_shape, transforms=transformations, proportion_training_circular_masks=config.proportion_training_circular_masks, circle_mask_shape=config.unet_img_shape, restrict_mask_to_wm=config.restrict_mask_to_wm)
dataset_evaluation = DatasetMRI2D(root_dir_img=Path(config.dataset_eval_path), restriction=config.restrict_eval_slices, root_dir_masks=Path(config.masks_eval_path), connected_masks=config.eval_connected_masks, target_shape=config.target_shape, dilation=config.eval_mask_dilation)
dataset_3D_evaluation = DatasetMRI3D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), connected_masks=config.eval_connected_masks, target_shape=config.target_shape, dilation=config.eval_mask_dilation)


# ### Prepare Training

# In[8]:


#create model
from diffusers import UNet2DModel

model = UNet2DModel(
    sample_size=config.unet_img_shape,  # the target image resolution
    in_channels=3, # the number of input channels: 1 for img, 1 for img_voided, 1 for mask
    out_channels=config.channels,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

config.model = "BigUNet2DModel"


# In[9]:


#setup noise scheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
#sample_image = datasetTrain[0]['gt_image'].unsqueeze(0)
#noise = torch.randn(sample_image.shape)
#timesteps = torch.LongTensor([50])
#noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


# In[10]:


# setup lr scheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
import math

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(math.ceil(len(dataset_train)/config.train_batch_size) * config.num_epochs), # num_iterations per epoch * num_epochs
)

config.lr_scheduler = "cosine_schedule_with_warmup"


# In[11]:


from accelerate import Accelerator 

accelerator = Accelerator(
    mixed_precision=config.mixed_precision,
    gradient_accumulation_steps=config.gradient_accumulation_steps,  
)


# In[12]:


from torch.utils.tensorboard import SummaryWriter
import os

if accelerator.is_main_process:
    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True) 
    #setup tensorboard
    tb_summary = SummaryWriter(config.output_dir, purge_step=0)
    accelerator.init_trackers("train_example") #maybe delete


# In[13]:


if accelerator.is_main_process:
    from custom_modules import Logger
    logger = Logger(tb_summary, config.log_csv)
    logger.log_config(config)


# In[14]:


from custom_modules import get_dataloader

train_dataloader = get_dataloader(dataset=dataset_train, batch_size = config.train_batch_size, 
                                  num_workers=config.num_dataloader_workers, random_sampler=True, 
                                  seed=config.seed, multi_slice=config.sorted_slice_sample_size > 1)
d2_eval_dataloader = get_dataloader(dataset=dataset_evaluation, batch_size = config.eval_batch_size, 
                                    num_workers=config.num_dataloader_workers, random_sampler=False, 
                                    seed=config.seed, multi_slice=config.sorted_slice_sample_size > 1)
d3_eval_dataloader = get_dataloader(dataset=dataset_3D_evaluation, batch_size = 1, 
                                    num_workers=config.num_dataloader_workers, random_sampler=False, 
                                    seed=config.seed, multi_slice=False) 


# In[ ]:


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
    "filename": config.3d_img_filename,
    "num_inference_steps": config.num_inference_steps,
    "eval_batch_size": config.eval_batch_size,
    "sorted_slice_sample_size": config.sorted_slice_sample_size,
    "evaluate_num_batches": config.evaluate_num_batches_3d,
    "seed": config.seed,
}
evaluation3D = Evaluation3DFilling(**args)


# In[ ]:


from custom_modules import Training, DDIMInpaintPipeline, Evaluation2DFilling, Evaluation3DFilling
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
trainingMixture = Training(**args)


# In[ ]:


if config.mode == "train":
    trainingMixture.train()


# In[ ]:


if config.mode == "eval":
    trainingMixture.deactivate_3D_evaluation = False
    pipeline = DDIMInpaintPipeline.from_pretrained(config.output_dir) 
    trainingMixture.evaluate(pipeline, deactivate_save_model=True)


# In[ ]:


print("Finished Training")

