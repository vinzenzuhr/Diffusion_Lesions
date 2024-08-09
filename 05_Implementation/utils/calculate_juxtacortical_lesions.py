"""This script calculates for every ROI of dl+direct the list of patients, which have lesions overlapping with it.

The ROI are defined by labels. The labels from 1000-3000 are the cortex labels from dl+direct. 
The labels from 1-100 are other ROI. The results are saved to a csv file. 
Optionally the masks can be further dilated. Assuming a perfect segmentation we would need a dilation=1 
to get the juxtacortical lesions overlapping with the cortex segmentation. 

"""

from pathlib import Path

import nibabel as nib
import torch
from tqdm.auto import tqdm

from custom_modules import DatasetMRI

mask_paths = Path("extended_cases/unhealthy_tysabri_T1_mask")
segm_paths = Path("extended_cases/unhealthy_tysabri_T1_segm_preprocessed")
min_overlapping_area = 64
# All cortex labels from dl+direct
DLDIRECT_LABELS = [1001,  1002,  1003,  1005,  1006,  1007,  1008,  1009,  1010,  1011,  1012,  1013,  1014,  1015,  1016,  1017,  1018,  
                   1019,  1020,  1021,  1022,  1023,  1024,  1025,  1026,  1027,  1028,  1029,  1030,  1031,  1032,  1033,  1034,  1035,  
                   2001,  2002,  2003,  2005,  2006,  2007,  2008,  2009,  2010,  2011,  2012,  2013,  2014,  2015,  2016,  2017,  2018,  
                   2019,  2020,  2021,  2022,  2023,  2024,  2025,  2026,  2027,  2028,  2029,  2030,  2031,  2032,  2033,  2034,  2035,  ]
# Other than cortex labels from dl+direct
#DLDIRECT_LABELS = [2,  101,  102,  10,  11,  12,  13,  17,  18,  26,  28,  41,  112,  113,  49,  50,  51,  52,  53,  54,  58,  60,  16,  
#                   14,  15,  125, 3,  42,  196,  197,  198]

mask_paths = list(mask_paths.rglob("*.nii.gz"))
segm_paths = list(segm_paths.rglob("*.nii.gz"))

label2text = dict()
with open("label_def.csv", 'r') as f: 
    for line in f:
        label, text = line.split(",")
        if label == "ID":
            continue
        label2text[int(label)] = text[:-1]  

juxtacortical_lesions = dict()
for mask_path, segm_path in tqdm(zip(mask_paths, segm_paths)):
    # First we dilate the mask and then calculate the overlapping area of the segmentation
    mask_nib = nib.load(mask_path)
    mask = torch.from_numpy(mask_nib.get_fdata())
    segm_nib = nib.load(segm_path)
    segm = torch.from_numpy(segm_nib.get_fdata())
    mask = DatasetMRI.dilate_mask(mask.to(torch.float32), num_pixels=1, kernel_shape="cross")
    limited_segm = segm * mask
    
    # we go through every cortex label and save the label if there is an lesion overlapping with it
    labels = []
    for label in DLDIRECT_LABELS:
        if torch.count_nonzero(limited_segm == label) > min_overlapping_area:
            labels.append(label)
    for label in labels:
        if label2text[label] in juxtacortical_lesions.keys():
            juxtacortical_lesions[label2text[label]].append(mask_path.parent.name)
        else:
            juxtacortical_lesions[label2text[label]] = [mask_path.parent.name] 

print(juxtacortical_lesions)

# Save the results to a csv file
with open("lesion_cortex.csv", 'w') as f:
    for label in juxtacortical_lesions.keys():
        f.writelines(label + ", " + ", ".join(juxtacortical_lesions[label]) + "\n")