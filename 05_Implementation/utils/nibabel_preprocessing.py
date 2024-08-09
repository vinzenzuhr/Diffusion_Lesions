"""This script preprocesses the MRI files

The script preprocesses the MRI files by transforming them to a target shape, a target voxel size and
a uniform orientation. The preprocessed files are saved in a new folder. 

Args:
    path (str): Path to the folder containing the MRI files.
    t1n_target_shape (str): The target shape for the MRI files e.g. "256,256,256".
"""

import os
from pathlib import Path
import sys

import nibabel as nib
import nibabel.processing

path = Path(sys.argv[1])
t1n_target_shape = str(sys.argv[2])  # "256,256,256"

t1n_target_shape = tuple(map(int, t1n_target_shape.split(","))) 
preprocessed_path = path.parent / (path.name+"_preprocessed")
files = path.rglob("*.nii.gz") 
print("Start processing ", len(list(files)), " files")
files = path.rglob("*.nii.gz") 

for file in files:
    print("Processing: ", file, "...")
    t1n = nib.load(file)
    t1n = nibabel.processing.conform(t1n, out_shape=t1n_target_shape, voxel_size=(1.0, 1.0, 1.0), orientation='RAS')   
    folder_name = file.parent
    os.makedirs(preprocessed_path / folder_name, exist_ok=True)
    nib.save(t1n, preprocessed_path / folder_name / file.name)
print("Done!")