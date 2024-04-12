import numpy as np
import nibabel as nib
import sys
from pathlib import Path

print("Start dldirect transformation")

folder = Path(sys.argv[1]) # "lesion-filling-256-cond-lesions/segmentations_3D"

file_list = list(folder.rglob("*T1w_norm_seg.nii.gz"))

for file in file_list:
    t1n_img = nib.load(file)
    t1n = t1n_img.get_fdata() 
    t1n = np.transpose(t1n)
    t1n = np.flip(t1n, axis=1)
    nifti_img = nib.Nifti1Image(t1n, None)
    #output name is the same as the input name with _transposed added to the end of the name. The filetype should be the same as the input file 
    output_name = file.parent / "T1w_norm_seg_transposed.nii.gz"
    nib.save(nifti_img, output_name)


print("Finished dldirect transformation")