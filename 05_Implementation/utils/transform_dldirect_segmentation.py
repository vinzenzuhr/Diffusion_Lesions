import numpy as np
import nibabel as nib

t1n_img = nib.load("T1w_norm_seg.nii.gz")
t1n = t1n_img.get_fdata() 
t1n = np.transpose(t1n)
t1n = np.flip(t1n, axis=1)
nifti_img = nib.Nifti1Image(t1n, None)
nib.save(nifti_img, "T1w_norm_seg_transposed.nii.gz")