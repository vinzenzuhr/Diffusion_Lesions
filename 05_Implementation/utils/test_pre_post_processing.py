import sys
sys.path.insert(1, './custom_modules')
#create config
from dataclasses import dataclass

@dataclass
class TrainingConfig: 
    img_target_shape = (512,512)#(128,224) #(128,512)
    t1n_target_shape = (256,256,256)
    channels = 1 
    train_batch_size = 4
    eval_batch_size = 4
    num_samples_per_batch = 1
    num_epochs = 80 # one epoch needs ~12min (x2 GPU), because their are more training samples due connected_components
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 100 #500
    evaluate_epochs = 5
    deactivate3Devaluation = True
    evaluate_num_batches = -1 # one batch needs ~15s.  
    evaluate_3D_epochs = 1000  # one 3D evaluation needs ~20min
    save_model_epochs = 60
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "lesion-filling-256-cond-lesions"  # the model name locally and on the HF Hub
    dataset_train_path = "./datasets/filling/dataset_train/imgs"
    segm_train_path = "./datasets/filling/dataset_train/segm"
    masks_train_path = "./datasets/filling/dataset_train/masks"
    dataset_eval_path = "./datasets/filling/dataset_eval/imgs"
    segm_eval_path = "./datasets/filling/dataset_eval/segm"
    masks_eval_path = "./datasets/filling/dataset_eval/masks" 
    """
    
    dataset_train_path = "./datasets/synthesis_flair/dataset_train/imgs"
    segm_train_path = "./datasets/synthesis_flair/dataset_train/segm"
    masks_train_path = "./datasets/synthesis_flair/dataset_train/masks"
    dataset_eval_path = "./datasets/synthesis_flair/dataset_eval/imgs"
    segm_eval_path = "./datasets/synthesis_flair/dataset_eval/segm"
    masks_eval_path = "./datasets/synthesis_flair/dataset_eval/masks" 
    """
    train_only_connected_masks=False
    eval_only_connected_masks=False
    num_inference_steps=50
    log_csv = False
    mode = "train" # train / eval
    debug = True
    brightness_augmentation = True

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    #hub_model_id = "<your-username>/<my-awesome-model>"  # the name of the repository to create on the HF Hub
    #hub_private_repo = False
    #overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
config = TrainingConfig()

if config.debug:
    config.num_inference_steps=1
    config.train_batch_size = 1
    config.eval_batch_size = 1 
    config.train_only_connected_masks=False
    config.eval_only_connected_masks=False
    config.evaluate_num_batches=1
    config.dataset_train_path = "./datasets/filling/dataset_eval/imgs"
    config.segm_train_path = "./datasets/filling/dataset_eval/segm"
    config.masks_train_path = "./datasets/filling/dataset_eval/masks"

#setup huggingface accelerate
import torch
import numpy as np
import accelerate
accelerate.commands.config.default.write_basic_config(config.mixed_precision)
#if there are problems with ports then add manually "main_process_port: 0" or another number to yaml file

from DatasetMRI3D import DatasetMRI3D
from pathlib import Path

dataset = DatasetMRI3D(root_dir_img=Path("./datasets/filling/dataset_eval_unhealthy/imgs"), t1n_target_shape=config.t1n_target_shape)    

for batch in dataset:
    print(batch["name"])   
    file = dataset.postprocess(batch["gt_image"][0], *batch["proc_info"], dataset.get_metadata(batch["idx"]))
    dataset.save(file, batch["name"] + ".nii.gz")