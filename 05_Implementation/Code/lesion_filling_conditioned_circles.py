#!/usr/bin/env python
# coding: utf-8

# In[1]:


#create config
from dataclasses import dataclass

@dataclass
class Config: 
    target_shape = None # will transform t1n during preprocessing (computationally expensive) 
    unet_img_shape = (256,256)
    channels = 1
    effective_train_batch_size=32 
    eval_batch_size = 16
    sorted_slice_sample_size = 1
    num_dataloader_workers = 8
    num_epochs = 900 # nochmals einschätzen
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500 #500
    evaluate_epochs = 18 #30
    deactivate2Devaluation = False
    deactivate3Devaluation = True
    evaluate_num_batches = 30 # one batch needs ~15s. 
    evaluate_num_batches_3d = -1  
    evaluate_unconditional_img = False
    evaluate_3D_epochs = 1000  # one 3D evaluation needs ~20min 
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-filling-256-cond-circle"  # the model name locally and on the HF Hub
    dataset_train_path = "./datasets/filling/dataset_train/imgs"
    segm_train_path = "./datasets/filling/dataset_train/segm"
    masks_train_path = "./datasets/filling/dataset_train/masks"
    dataset_eval_path = "./datasets/filling/dataset_eval/imgs"
    segm_eval_path = "./datasets/filling/dataset_eval/segm"
    masks_eval_path = "./datasets/filling/dataset_eval/masks" 
    train_connected_masks=False # No Training with lesion masks
    eval_connected_masks=False 
    num_inference_steps=50
    log_csv = False
    mode = "train" # train / eval
    debug = False
    brightness_augmentation = True
    proportion_training_circular_masks = 1.0
    eval_loss_timesteps=[20,80,140,200,260,320,380,440,560,620,680,740,800,860,920,980]
    restrict_train_slices = "segm"
    restrict_eval_slices = "mask"
    use_min_snr_loss=False
    snr_gamma=5

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    #hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = False
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
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

transformations = None
if config.brightness_augmentation:
    transformations = transforms.RandomApply([ScaleDecorator(transforms.ColorJitter(brightness=1))], p=0.5)

#create dataset
dataset_train = DatasetMRI2D(root_dir_img=Path(config.dataset_train_path), restriction=config.restrict_train_slices, root_dir_segm=Path(config.segm_train_path), connected_masks=config.train_connected_masks, target_shape=config.target_shape, transforms=transformations, proportion_training_circular_masks=config.proportion_training_circular_masks, circle_mask_shape=config.unet_img_shape)
dataset_evaluation = DatasetMRI2D(root_dir_img=Path(config.dataset_eval_path), restriction=config.restrict_eval_slices, root_dir_masks=Path(config.masks_eval_path), connected_masks=config.eval_connected_masks, target_shape=config.target_shape)
dataset_3D_evaluation = DatasetMRI3D(root_dir_img=Path(config.dataset_eval_path), root_dir_masks=Path(config.masks_eval_path), connected_masks=config.eval_connected_masks, target_shape=config.target_shape)


# ### Visualize dataset

# In[ ]:


import matplotlib.pyplot as plt
fig, axis = plt.subplots(1,2,figsize=(16,4))
idx=80
axis[0].imshow((dataset_train[idx]["gt_image"].squeeze()+1)/2)
axis[1].imshow(np.logical_or(dataset_train[idx]["segm"].squeeze()==41, dataset_train[idx]["segm"].squeeze()==2))
fig.show 


# In[ ]:


# Get 6 random sample
random_indices = np.random.randint(0, len(dataset_train) - 1, size=(6)) 

# Plot: t1n segmentations
fig, axis = plt.subplots(2,3,figsize=(16,4))
for i, idx in enumerate(random_indices):
    axis[i//3,i%3].imshow(np.logical_or(dataset_train[idx]["segm"].squeeze()==41, dataset_train[idx]["segm"].squeeze()==2))
    axis[i//3,i%3].set_axis_off()
fig.show()


# In[ ]:


# Plot: t1n images
fig, axis = plt.subplots(2,3,figsize=(16,4))
for i, idx in enumerate(random_indices):
    axis[i//3,i%3].imshow((dataset_train[idx]["gt_image"].squeeze()+1)/2)
    axis[i//3,i%3].set_axis_off()
fig.show()


# ### Playground for random circles

# In[ ]:


# visualize normal distributions of center points
centers=[]
for _ in np.arange(100):
    centers.append(torch.normal(torch.tensor([127.,127.]),torch.tensor(30.)))

plt.imshow((dataset_train[70]["gt_image"].squeeze()+1)/2) 
for center in centers:
    plt.scatter(center[0], center[1])


# In[ ]:


example=torch.zeros((10,256,256)).shape

#create circular mask with random center around the center point of the pictures and a radius between 3 and 40 pixels
center=np.random.normal([127,127],30, size=(example[0],2))
radius=np.random.uniform(low=3,high=40, size=(example[0]))

Y, X = np.ogrid[:256, :256] # gives two vectors, each containing the pixel locations. There's a column vector for the column indices and a row vector for the row indices.
dist_from_center = np.sqrt((X.T - center[:,0])[None,:,:]**2 + (Y-center[:,1])[:,None,:]**2) # creates matrix with euclidean distance to center

mask = dist_from_center <= radius # creates mask for pixels which are inside the radius
mask = 1-mask

plt.imshow(((dataset_train[70]["gt_image"].squeeze()+1)/2)*mask[:,:,4]) 



# ### Prepare Training

# In[ ]:


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


# In[ ]:


#setup noise scheduler
import torch
from PIL import Image
from diffusers import DDIMScheduler

noise_scheduler = DDIMScheduler(num_train_timesteps=1000)
#sample_image = datasetTrain[0]['gt_image'].unsqueeze(0)
#noise = torch.randn(sample_image.shape)
#timesteps = torch.LongTensor([50])
#noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

#tb_summary.add_text("noise_scheduler", "DDIMScheduler(num_train_timesteps=1000)", 0) 

#Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])


config.noise_scheduler = "DDIMScheduler(num_train_timesteps=1000)"


# In[ ]:


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
    "config": config, 
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
    "logger": logger,
    "pipeline_factory": PipelineFactories.get_ddim_inpaint_pipeline,
    "num_epochs": config.num_epochs, 
    "evaluate_2D_epochs": config.evaluate_epochs,
    "evaluate_3D_epochs": config.evaluate_3D_epochs,
    "min_snr_loss": config.use_min_snr_loss,
    "snr_gamma": config.snr_gamma,
    "evaluate_unconditional_img": config.evaluate_unconditional_img,
    "deactivate_2D_evaluation": config.deactivate2Devaluation, 
    "deactivate_3D_evaluation": config.deactivate3Devaluation, 
    "evaluation_pipeline_parameters": {},
    "debug": config.debug, 
    }
trainingCircles = Training(**args) 


# In[ ]:


if config.mode == "train":
    trainingCircles.train()


# In[ ]:


if config.mode == "eval":
    trainingCircles.deactivate_3D_evaluation = False
    pipeline = DDIMInpaintPipeline.from_pretrained(config.output_dir) 
    trainingCircles.evaluate(pipeline, deactivate_save_model=True)


# In[ ]:


print("Finished Training")

