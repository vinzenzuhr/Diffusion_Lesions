import sys
sys.path.insert(1, './custom_modules')
from DatasetMRI2D import DatasetMRI2D 
from pathlib import Path 
import os

path_t1n=""
path_mask=""

os.system("find " + path_mask + " -type f -name '*component_matrix.npy' -delete")

#create dataset, which creates all the component_matrices
dataset = DatasetMRI2D(root_dir_img=Path(path_t1n), root_dir_masks=Path(path_mask), only_connected_masks=True, t1n_target_shape=None, transforms=None)
 