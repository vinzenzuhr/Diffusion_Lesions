#!/bin/bash

#pip install niftyreg

nifty_path="/opt/conda/lib/python3.11/site-packages/niftyreg/bin"
float_mri_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/conditioned_dataset/*"
ref_mri_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/OASIS/imgs/*" 
output_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/transformed_lesions"
temp_path="/home/jovyan/workdir/Master_Thesis/Diffusion_Lesions/temp_04_Datasets/transformed_lesions"

mkdir -p $output_path
mkdir -p $temp_path

#Register and transform FLAIR segmentation to T1
#for float_folder in $float_mri_path"/*"
#do
	#Affine registration
#	$nifty_path"/reg_aladin" -ref $float_folder"/T1_preprocessed.nii.gz" -flo $float_folder"/FLAIR_preprocessed.nii.gz" -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz"
	#Non linear registration with affine initialization 
#	$nifty_path"/reg_f3d" -ref $float_folder"/T1_preprocessed.nii.gz" -flo $float_folder"/FLAIR_preprocessed.nii.gz" -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz" -cpp $temp_path"/temp_non_linear_transform.nii"
	#Transformation based on non linear registration. Inter=0 for nearest neighbor interpolation
#	$nifty_path"/reg_resample" -ref $float_folder"/T1_preprocessed.nii.gz" -flo $float_folder"/Consensus.nii.gz" -trans $temp_path"/temp_non_linear_transform.nii" -res $float_folder"/Consensus_T1registered.nii.gz" -inter 0
#done
#$nifty_path"/reg_aladin" -ref $ref_folder"/T1.nii.gz" -flo $float_folder"/T1_preprocessed.nii.gz" -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz"
#$nifty_path"/reg_f3d" -ref $ref_folder"/T1.nii.gz" -flo $float_folder"/T1_preprocessed.nii.gz" -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz" -cpp $temp_path"/temp_non_linear_transform.nii"
#$nifty_path"/reg_resample" -ref $ref_folder"/T1.nii.gz" -flo $float_folder"/Consensus_T1registered.nii.gz" -trans $temp_path"/temp_non_linear_transform.nii" -res $output_path"/${ref_folder##*/}/${float_folder##*/}/transformed_segmentation.nii.gz" -inter 0
	
#Register and transform segmentations between different patients
for ref_folder in $ref_mri_path
do
	for float_folder in $float_mri_path
	do  
		mkdir -p $output_path"/${ref_folder##*/}/${float_folder##*/}/"
		#Affine registration 
		#$nifty_path"/reg_aladin" -ref $ref_folder"/T1.nii.gz" -flo $float_folder"/FLAIR_preprocessed.nii.gz" -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz"
		"${nifty_path}/reg_aladin" -ref $ref_folder/T1.nii.gz -flo $float_folder/FLAIR_preprocessed.nii.gz -aff $temp_path/temp_aff_transform.txt -res $temp_path/temp_img.nii.gz
		#Non linear registration with affine initialization 
		"${nifty_path}/reg_f3d" -ref $ref_folder"/T1.nii.gz" -flo $float_folder"/FLAIR_preprocessed.nii.gz" -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz" -cpp $temp_path"/temp_non_linear_transform.nii"
		#Transformation based on non linear registration. Inter=0 for nearest neighbor interpolation
		"${nifty_path}/reg_resample" -ref $ref_folder"/T1.nii.gz" -flo $float_folder"/Consensus.nii.gz" -trans $temp_path"/temp_non_linear_transform.nii" -res $output_path"/${ref_folder##*/}/${float_folder##*/}/transformed_segmentation.nii.gz" -inter 0
	done
done

rm $temp_path"/temp_img.nii.gz"
rm $temp_path"/temp_non_linear_transform.nii"
rm $temp_path"/temp_aff_transform.txt"