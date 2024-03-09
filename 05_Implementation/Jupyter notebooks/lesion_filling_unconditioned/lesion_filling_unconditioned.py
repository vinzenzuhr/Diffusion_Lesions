#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[101]:


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




# In[109]:


#Preprocess with UniRes
from unires.struct import settings
from unires.run import preproc
import os

reconstructed_affine_matrices = dict()

os.makedirs(config.uniform_dataset_path, exist_ok=True)
if (not any(Path(config.uniform_dataset_path).iterdir())):
    data = list(Path(config.dataset_path).rglob("*.nii.gz"))
    s = settings()
    s.common_output = True  # ensures 'standardised' outputs across subjects
    s.dir_out = config.uniform_dataset_path
    _, mat, uniform_data_pth = preproc(data, sett=s)
    reconstructed_affine_matrices[uniform_data_pth] = mat
    
    #Show 4 examples
    _ = [show_image(uniform_data_pth[i], fig_num=i) for i in range(4)]
else:
    print("Skipping preprocessing with UniRes, because output_dir is not empty")


