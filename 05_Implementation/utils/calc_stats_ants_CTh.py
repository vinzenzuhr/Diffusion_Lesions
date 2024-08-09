
"""
Calculate statistics from the output of the cortical thickness pipeline from ANTs.

Raises:
    AssertionError: If the number of files in dir_thick is not equal to the number of files in dir_aparc.

"""

import os
from pathlib import Path
import sys
import subprocess
 
dir_thick = Path(sys.argv[1])
dir_aparc = Path(sys.argv[2])
dir_stats_script = Path(sys.argv[3])

assert(len(os.listdir(dir_thick)) == len(os.listdir(dir_aparc)))

patient_names = os.listdir(dir_thick)
for patient_name in patient_names:
    thick_img = dir_thick / patient_name / "CorticalThickness.nii.gz" #"thickness_image.nii.gz"
    segm_img = dir_thick / patient_name / "BrainSegmentation.nii.gz" #"segmentation_image.nii.gz"
    aparc_file = dir_aparc / patient_name / "T1w_norm_seg_transposed.nii.gz"
    subprocess.run(["python", os.path.abspath((dir_stats_script / "extract_stats.py")), thick_img, segm_img, aparc_file, patient_name])