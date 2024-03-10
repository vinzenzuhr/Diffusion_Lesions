#!/usr/bin/env python
# coding: utf-8

# In[121]:


from torch.utils.tensorboard import SummaryWriter
tb_summary = SummaryWriter("./Tensorboards/", purge_step=0)


# In[122]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu") 


# In[123]:


#create config
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256  # TODO: the generated image resolution
    channels = 3 # only used for logging
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 50
    #gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-filling-128"  # the model name locally and on the HF Hub
    dataset_path = "./dataset"
    uniform_dataset_path = "./uniform_dataset"

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    #hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = False
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()


# In[124]:


#log at tensorboard
tb_summary.add_scalar("image_size", config.image_size, 0)
tb_summary.add_scalar("train_batch_size", config.train_batch_size, 0)
tb_summary.add_scalar("eval_batch_size", config.eval_batch_size, 0)
tb_summary.add_scalar("num_epochs", config.num_epochs, 0)
tb_summary.add_scalar("learning_rate", config.learning_rate, 0)
tb_summary.add_scalar("lr_warmup_steps", config.lr_warmup_steps, 0)
tb_summary.add_scalar("save_image_epochs", config.save_image_epochs, 0)
tb_summary.add_text("mixed_precision", config.mixed_precision, 0) 


# In[125]:


#Preprocess with UniRes
"""
from unires.struct import settings
from unires.run import preproc
import os
from pathlib import Path

reconstructed_affine_matrices = dict()

os.makedirs(config.uniform_dataset_path, exist_ok=True)
if (not any(Path(config.uniform_dataset_path).iterdir())):
    data = list(Path(config.dataset_path).rglob("*.nii.gz"))
    [str(x) for x in data]
    s = settings()
    s.common_output = True  # ensures 'standardised' outputs across subjects
    s.dir_out = config.uniform_dataset_path
    for img in data:
        _, mat, uniform_data_pth = preproc(img, sett=s)
        reconstructed_affine_matrices[uniform_data_pth] = mat 
else:
    print("Skipping preprocessing with UniRes, because output_dir is not empty")
"""


# In[133]:


#Preprocess with Nitorch
import os
import nitorch as ni
from nitorch.tools.preproc import atlas_align
from pathlib import Path

reconstructed_affine_matrices = dict()

os.makedirs(config.uniform_dataset_path, exist_ok=True)
if (not any(Path(config.uniform_dataset_path).iterdir())):
    data = list(Path(config.dataset_path).rglob("*.nii.gz"))
    [str(x) for x in data]

    for img in data[0:10]:
        nii_mov = ni.io.map(img)
        dat_mov = nii_mov.fdata(device=device)
        mat_mov = nii_mov.affine.to(device)
    
        dat_mov_atlas, mat, pth, _ = atlas_align([dat_mov, mat_mov], rigid=False, device=device, default_atlas="atlas_t1", write="affine", odir=config.uniform_dataset_path)
        reconstructed_affine_matrices[pth] = mat

    """
    dat_mov = list()
    mat_mov = list()
    for img in data[0:10]:
        nii_mov = ni.io.map(img)
        dat_mov.append(nii_mov.fdata(device=device))
        mat_mov.append(nii_mov.affine.to(device))

    dat_mov_atlas, mat, paths, _ = atlas_align([list(x) for x in zip(dat_mov, mat_mov)], rigid=False, device=device, default_atlas="atlas_t1", write="affine", odir=config.uniform_dataset_path)

    for i, path in enumerate(paths):
        reconstructed_affine_matrices[path] = mat[i]
    """
else:
    print("Skipping preprocessing with Nitorch, because output_dir is not empty")


# In[96]:


