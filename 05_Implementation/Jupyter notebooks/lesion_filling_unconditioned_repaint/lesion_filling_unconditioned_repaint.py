#!/usr/bin/env python
# coding: utf-8

# In[1]:


#create config
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256  # TODO: the generated image resolution
    channels = 1
    train_batch_size = 2 # 16
    eval_batch_size = 2  # 2 x 2 GPU = 4 
    num_epochs = 600
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 30
    save_model_epochs = 300
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-filling-256-repaint"  # the model name locally and on the HF Hub
    dataset_eval_path = "./dataset_eval/imgs"
    segm_eval_path = "./dataset_eval/segm"
    pretrained_path = "./pretrained"
    num_gpu=2
    #uniform_dataset_path = "./uniform_dataset"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    #hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = False
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()


# In[2]:


# create dataset
import torch 
from torch.utils.data import Dataset
from torch.nn import functional as F
from pathlib import Path
import nibabel as nib
import numpy as np
from math import floor, ceil

class DatasetMRI(Dataset):
    """
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
    """

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
        """
        Transforms the images to a more unified format.
        Normalizes to -1,1. Pad and crop to bounding box.
        
        Args:
            t1n (np.ndarray): t1n from t1n file (ground truth).

        Raises:
            UserWarning: When your input images are not (256, 256, 160)

        Returns:
            t1n: The padded and cropped version of t1n.
            t1n_max_v: Maximal value of t1n image (used for normalization).
        """

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


# In[3]:


#create dataset
datasetEvaluation = DatasetMRI(Path(config.dataset_eval_path), Path(config.segm_eval_path), pad_shape=(256, 256, 256)) # TODO: check shape 

print(f"Dataset size: {len(datasetEvaluation)}")
print(f"\tImage shape: {datasetEvaluation[0]['gt_image'].shape}")
print(f"Evaluation Data: {list(datasetEvaluation[0].keys())}") 


# In[4]:


from torch.utils.data import DataLoader

# create dataloader function, which is executed inside the training loop (necessary because of huggingface accelerate)
def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


# In[5]:


# load model
from diffusers import DDIMPipeline
from diffusers import UNet2DModel

pipe = DDIMPipeline.from_pretrained(config.pretrained_path)
model=pipe.unet
print("done")


# ### RePaint Inpainting

# In[6]:


def generate_masks(n, device, generator=None):
    #create circular mask with random center around the center point of the pictures and a radius between 3 and 50 pixels
    center=torch.normal(mean=config.image_size/2, std=30, size=(n,2), generator=generator, device=device) # 30 is chosen by inspection
    low=3   
    high=50
    radius=torch.rand(n, generator=generator, device=device)*(high-low)+low # get radius between 3 and 50 from uniform distribution 

    #Test case
    #center=torch.tensor([[0,255],[0,255]]) 
    #radius=torch.tensor([2,2])
    
    Y, X = [torch.arange(config.image_size, device=device)[:,None],torch.arange(config.image_size, device=device)[None,:]] # gives two vectors, each containing the pixel locations. There's a column vector for the column indices and a row vector for the row indices.
    dist_from_center = torch.sqrt((X.T - center[:,0])[None,:,:]**2 + (Y-center[:,1])[:,None,:]**2) # creates matrix with euclidean distance to center
    dist_from_center = dist_from_center.permute(2,0,1) 

    #Test case
    #print(dist_from_center[0,0,0]) #=255
    #print(dist_from_center[0,0,255]) #=360.624
    #print(dist_from_center[0,255,0]) #=0
    #print(dist_from_center[0,255,255]) #=255
    #print(dist_from_center[0,127,127]) #=180.313 
    
    masks = dist_from_center > radius[:,None,None] # creates mask for pixels which are outside the radius. 
    masks = masks[:,None,:,:].int() 
    return masks


# In[7]:


#setup evaluation
from diffusers import DDIMPipeline
from diffusers.utils import make_image_grid
import os
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

def evaluate(config, pipeline, eval_dataloader, accelerator):
    batch = next(iter(eval_dataloader)) #TODO: Anpassen, falls grösseres evaluation set. Evt. anpassen für accelerate.
    clean_images = batch["gt_image"]
    
    generator = torch.cuda.manual_seed(config.seed) if torch.cuda.is_available() else torch.manual_seed(config.seed)
    masks = generate_masks(n=clean_images.shape[0], generator=generator, device=clean_images.device)
    voided_images = clean_images*masks 

    images = pipeline(voided_images, masks, generator=torch.manual_seed(config.seed), output_type=np.array, jump_length=8).images
    images = (torch.from_numpy(images)).to(clean_images.device) 
    

    #collect all images from the different processes/GPUs
    clean_images  = accelerator.gather(clean_images)
    images = accelerator.gather(images)
    voided_images = accelerator.gather(voided_images)

    if accelerator.is_main_process: 

        test_dir = os.path.join(config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True) 

        
        pil_images = [to_pil_image(x) for x in images]
        image_grid = make_image_grid(pil_images, rows=2, cols=2)
        image_grid.save(f"{test_dir}/inpainted_image.png")
        
        pil_voided_images = [to_pil_image(x) for x in voided_images]
        voided_image_grid = make_image_grid(pil_voided_images, rows=2, cols=2)
        voided_image_grid.save(f"{test_dir}/voided_image.png")    

        pil_clean_images = [to_pil_image(x) for x in clean_images]
        clean_image_grid = make_image_grid(pil_clean_images, rows=2, cols=2)
        clean_image_grid.save(f"{test_dir}/clean_image.png")   
        
        print("image saved")


# In[10]:


from diffusers import RePaintPipeline, RePaintScheduler  
from accelerate import Accelerator, PartialState

accelerator = Accelerator()
distributed_state = PartialState()

eval_dataloader = get_dataloader(datasetEvaluation, config.eval_batch_size, shuffle=False)
model.eval()
pipeline = RePaintPipeline(unet=model, scheduler=RePaintScheduler())
pipeline.to(distributed_state.device) 

pipeline, eval_dataloader = accelerator.prepare(
        pipeline, eval_dataloader
    )

evaluate(config, pipeline, eval_dataloader, accelerator)
    
    
    


# In[14]:
