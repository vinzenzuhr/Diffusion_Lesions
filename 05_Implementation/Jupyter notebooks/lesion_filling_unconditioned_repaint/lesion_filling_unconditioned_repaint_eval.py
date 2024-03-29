#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(1, '../custom_modules')


# In[2]:


#### create config
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256  # TODO: the generated image resolution
    channels = 1
    train_batch_size = 4 
    eval_batch_size = 4  
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    evaluate_epochs = 40 # anpassen auf Anzahl epochs
    evaluate_num_batches = 2 # one batch needs ~130s 
    deactivate3Devaluation = False
    evaluate_3D_epochs = 1000  # one 3D evaluation has 77 slices and needs 166min
    save_model_epochs = 300
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-filling-256-repaint"  # the model name locally and on the HF Hub
    dataset_train_path = "./dataset_train/imgs"
    segm_train_path = "./dataset_train/segm"
    masks_train_path = "./dataset_train/masks"
    dataset_eval_path = "./dataset_eval/imgs"
    segm_eval_path = "./dataset_eval/segm"
    masks_eval_path = "./dataset_eval/masks"  
    train_only_connected_masks=False  # No Training with lesion masks
    eval_only_connected_masks=False 
    num_inference_steps=50
    mode = "eval" # train / eval
    debug = False
    jump_length=8
    jump_n_sample=10 
    #uniform_dataset_path = "./uniform_dataset"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    #hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = False
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()


# In[3]:


if config.debug:
    config.num_inference_steps=5
    config.train_batch_size = 1
    config.eval_batch_size = 1 
    config.train_only_connected_masks=False
    config.eval_only_connected_masks=False
    config.evaluate_num_batches=1
    dataset_train_path = "./dataset_eval/imgs"
    segm_train_path = "./dataset_eval/segm"
    masks_train_path = "./dataset_eval/masks"  
    config.jump_length=1
    config.jump_n_sample=1


# In[4]:


#setup huggingface accelerate
import torch
import numpy as np
import accelerate
accelerate.commands.config.default.write_basic_config(config.mixed_precision)


# In[5]:


"""# create dataset
import torch 
from torch.utils.data import Dataset
from torch.nn import functional as F
from pathlib import Path
import nibabel as nib
import numpy as np
from math import floor, ceil

class DatasetMRI(Dataset):
    
    Dataset for Training purposes. 
    Adapted implementation of BraTS 2023 Inpainting Challenge (https://github.com/BraTS-inpainting/2023_challenge).
    
    Contains ground truth t1n images (gt) 
    Args:
        root_dir_img: Path to img files
        root_dir_segm: Path to segmentation maps
        pad_shape: Shape the images will be transformed to

    Raises:
        UserWarning: When your input images are not (256, 256, 160)

    Returns: 
        __getitem__: Returns a dictoinary containing:
            "gt_image": Padded and cropped version of t1n 2D slice
            "t1n_path": Path to the unpadded t1n file for this sample
            "max_v": Maximal value of t1 image (used for normalization) 
    

    def __init__(self, root_dir_img: Path, root_dir_segm: Path, pad_shape=(256,256,256)):
        #Initialize variables
        self.root_dir_img = root_dir_img
        self.root_dir_segm = root_dir_segm
        self.pad_shape = pad_shape 
        self.list_paths_t1n = list(root_dir_img.rglob("*.nii.gz"))
        self.list_paths_segm = list(root_dir_segm.rglob("*.nii.gz"))
        #define offsets between first and last segmented slices and the slices to be used for training
        bottom_offset=60 
        top_offset=20

        #go through all 3D imgs
        idx=0
        self.idx_to_2D_slice = dict()
        for j, path in enumerate(self.list_paths_segm):
            t1n_segm = nib.load(path)
            t1n_3d = t1n_segm.get_fdata()

            #get first slice with segmented content and add offset
            i=0
            while(not t1n_3d[:,i,:].any()):
                i+=1
            bottom=i+bottom_offset

            #get last slice with segmented content and add offset
            i=t1n_3d.shape[1]-1
            while(not t1n_3d[:,i,:].any()):
                i-=1
            top=i-top_offset

            #Add all slices between desired top and bottom slice to dataset
            for i in np.arange(top-bottom):
                self.idx_to_2D_slice[idx]=(self.list_paths_t1n[j],bottom+i)
                idx+=1 

    def __len__(self): 
        return len(self.idx_to_2D_slice.keys()) 

    def preprocess(self, t1n: np.ndarray):
        
        Transforms the images to a more unified format.
        Normalizes to -1,1. Pad and crop to bounding box.
        
        Args:
            t1n (np.ndarray): t1n from t1n file (ground truth).

        Raises:
            UserWarning: When your input images are not (256, 256, 160)

        Returns:
            t1n: The padded and cropped version of t1n.
            t1n_max_v: Maximal value of t1n image (used for normalization).
        

        #Size assertions
        reference_shape = (256,256,160)
        if t1n.shape != reference_shape:
            raise UserWarning(f"Your t1n shape is not {reference_shape}, it is {t1n.shape}")

        #Normalize the image to [0,1]
        t1n[t1n<0] = 0 #Values below 0 are considered to be noise #TODO: Check validity
        t1n_max_v = np.max(t1n)
        t1n /= t1n_max_v

        #pad to bounding box
        size = self.pad_shape # shape of bounding box is (size,size,size) #TODO: Find solution for 2D
        t1n = torch.Tensor(t1n)
        d, w, h = t1n.shape[-3], t1n.shape[-2], t1n.shape[-1]
        d_max, w_max, h_max = size
        d_pad = max((d_max - d) / 2, 0)
        w_pad = max((w_max - w) / 2, 0)
        h_pad = max((h_max - h) / 2, 0)
        padding = (
            int(floor(h_pad)),
            int(ceil(h_pad)),
            int(floor(w_pad)),
            int(ceil(w_pad)),
            int(floor(d_pad)),
            int(ceil(d_pad)),
        )
        t1n = F.pad(t1n, padding, value=0, mode="constant") 

        #map images from [0,1] to [-1,1]
        t1n = (t1n*2) - 1

        return t1n, t1n_max_v

    def __getitem__(self, idx):
        t1n_path = self.idx_to_2D_slice[idx][0]
        slice_idx = self.idx_to_2D_slice[idx][1]
        t1n_img = nib.load(t1n_path)
        t1n = t1n_img.get_fdata()
        
        # preprocess data
        t1n, t1n_max_v = self.preprocess(t1n) # around 0.2s on local machine
        
        # get 2D slice from 3D
        t1n_slice = t1n[:,slice_idx,:] 
        
        # Output data
        sample_dict = {
            "gt_image": t1n_slice.unsqueeze(0),
            "t1n_path": str(t1n_path),  # path to the 3D t1n file for this sample.
            "max_v": t1n_max_v,  # maximal t1n_voided value used for normalization 
        }
        return sample_dict 
"""


# In[6]:


from DatasetMRI2D import DatasetMRI2D
from DatasetMRI3D import DatasetMRI3D
from pathlib import Path

#create dataset
datasetTrain = DatasetMRI2D(Path(config.dataset_train_path), Path(config.segm_train_path), only_connected_masks=config.train_only_connected_masks)
datasetEvaluation = DatasetMRI2D(Path(config.dataset_eval_path), Path(config.segm_eval_path), Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks)
dataset3DEvaluation = DatasetMRI3D(Path(config.dataset_eval_path), Path(config.segm_eval_path), Path(config.masks_eval_path), only_connected_masks=config.eval_only_connected_masks)

"""#create dataset
datasetEvaluation = DatasetMRI(Path(config.dataset_eval_path), Path(config.segm_eval_path), pad_shape=(256, 256, 256)) # TODO: check shape 

print(f"Dataset size: {len(datasetEvaluation)}")
print(f"\tImage shape: {datasetEvaluation[0]['gt_image'].shape}")
print(f"Evaluation Data: {list(datasetEvaluation[0].keys())}") """


# In[7]:


"""from torch.utils.data import DataLoader

# create dataloader function, which is executed inside the training loop (necessary because of huggingface accelerate)
def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)"""


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
from diffusers import RePaintPipeline

config.conditional_data = "None"

args = {"config": config, "model": model, "noise_scheduler": noise_scheduler, "optimizer": optimizer, "lr_scheduler": lr_scheduler, "datasetTrain": datasetTrain, "datasetEvaluation": datasetEvaluation, "dataset3DEvaluation": dataset3DEvaluation, "deactivate3Devaluation": config.deactivate3Devaluation} 
trainingRepaint = TrainingUnconditional(**args)


# In[12]:


if config.mode == "train":
    trainingRepaint.train()


# In[ ]:


if config.mode == "eval":
    pipeline = RePaintPipeline.from_pretrained(config.output_dir) 
    trainingRepaint.evaluate(pipeline)


# In[ ]:


print("Finished Training")


# In[1]:
