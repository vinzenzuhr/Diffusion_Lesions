""" Calculate cortical thickness using deep learning pipeline from ANTsPyNet

Output is a thickness map and a segmentation image, which are saved in the output directory.
"""

import os
from pathlib import Path
import sys

import ants
import antspynet

dir = Path(sys.argv[1]) 
files = dir.rglob("*.nii.gz")

output_dir = dir.parent / (dir.name + "_CTh")
os.makedirs(output_dir, exist_ok=True)
for file in files:
    t1 = ants.image_read(os.path.abspath(file))
    kk = antspynet.utilities.cortical_thickness(t1, antsxnet_cache_directory=None, verbose=True)
    os.makedirs(output_dir / file.parent.name, exist_ok=True)
    ants.core.ants_image_io.image_write(kk["thickness_image"], os.path.abspath(output_dir / file.parent.name / "thickness_image.nii.gz"))
    ants.core.ants_image_io.image_write(kk["segmentation_image"], os.path.abspath(output_dir / file.parent.name / "segmentation_image.nii.gz"))





