#!/bin/bash

#pip install niftyreg

nifty_path="/opt/conda/lib/python3.11/site-packages/niftyreg/bin" #python import niftyreg; niftyreg.bin_path
float_mri_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/conditioned_dataset_without_preprocessing/*"
float_flair_name="3DFLAIR.nii.gz"
float_segmentation_name="Consensus.nii.gz"
ref_mri_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/OASIS/imgs/*" 
ref_file_name="T1.nii.gz"
output_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/transformed_lesions"
temp_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/transformed_lesions"

mkdir -p $output_path
mkdir -p $temp_path

Affine registration  
"${nifty_path}/reg_aladin" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_flair_name -aff $temp_path/temp_aff_transform.txt -res $temp_path/temp_img.nii.gz
#Non linear registration with affine initialization

#Auf Ubelix mit FLAIR und T1:
hd-bet -i INPUT_FOLDER -o OUTPUT_FOLDER


#Mit skull stripped FLAIR und T1
"${nifty_path}/reg_f3d" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_flair_name -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz" -cpp $temp_path"/temp_non_linear_transform.nii"
#Transformation based on non linear registration. Inter=0 for nearest neighbor interpolation
"${nifty_path}/reg_resample" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_segmentation_name -trans $temp_path"/temp_non_linear_transform.nii" -res $output_path"/${ref_folder##*/}/${float_folder##*/}/transformed_segmentation.nii.gz" -inter 0



rm $temp_path"/temp_img.nii.gz"
rm $temp_path"/temp_non_linear_transform.nii"
rm $temp_path"/temp_aff_transform.txt"