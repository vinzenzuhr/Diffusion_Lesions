"""Generates new masks based on the largest connected component from the selected masks.

In the evaluation task a doctor has to find the synthesized lesion in a set of real lesions.
For this task the largest connected components (lesions) are used as set of real lesions. One of these
components is then selected randomly and used as the synthesized lesion.
The output are two masks, one with the selected components and one with the random component used 
to inpaint a synthetic lesion.
"""

import heapq
import os
from pathlib import Path
import random

import nibabel as nib
from scipy.ndimage import label
import torch
from tqdm.auto import tqdm

path = Path("./unhealthy_tysabri_flair_segmentation_lesion_2")
num_components = 10

mask_paths = list(path.rglob("*.nii.gz"))
for mask_path in tqdm(mask_paths):
    # Create a heap of all connected components sorted by size
    mask_nib = nib.load(mask_path)
    mask = torch.from_numpy(mask_nib.get_fdata())
    component_matrix, n = label(mask)
    component_matrix = torch.from_numpy(component_matrix)
    component_sizes = []
    for i in range(1, n + 1): 
        component_sizes.append((-torch.count_nonzero(component_matrix == i), i))
    heapq.heapify(component_sizes)

    # Select the largest connected components and create a new mask.
    # These components are the ones that will be used for the evaluation.
    components = [heapq.heappop(component_sizes)[1] for _ in range(num_components)]
    mapping_labels = torch.tensor([0] * (n+1))  
    for component in components: 
        mapping_labels[component] = 1  
    matrix_selected = mapping_labels[component_matrix]
    metadata = {
        "affine": mask_nib.affine,
        "header": mask_nib.header,
        "extra": mask_nib.extra,
        "file_map": mask_nib.file_map,
        "dtype": mask_nib.get_data_dtype()
    }
    mask = nib.nifti1.Nifti1Image(matrix_selected.numpy(), **metadata)
    path = "tasks/" + mask_path.parent.name + "/"
    os.makedirs(path, exist_ok=True)
    nib.save(mask, path + "selected_masks.nii.gz")
    
    # Create a random mask with one of the selected components
    random_component = components[random.randint(0, num_components - 1)]
    mapping_labels = torch.tensor([0] * (n+1)) 
    mapping_labels[random_component] = 1  
    matrix_random_component = mapping_labels[component_matrix]
    mask = nib.nifti1.Nifti1Image(matrix_random_component.numpy(), **metadata)
    path = "solutions/" + mask_path.parent.name + "/"
    os.makedirs(path, exist_ok=True)
    nib.save(mask, path + "random_component_masks.nii.gz")