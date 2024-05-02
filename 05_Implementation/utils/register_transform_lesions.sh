
#!/bin/bash

#pip install niftyreg

nifty_path="/storage/homefs/vu16r049/miniconda3/lib/python3.9/site-packages/niftyreg/bin" #python import niftyreg; niftyreg.bin_path
float_mri_path="/storage/homefs/vu16r049/tasks/test_nibabel_preprocessed/unhealthy_T1_DL+DiReCT_Segmentation"
float_flair_name="T1w_norm.nii.gz"
#float_noskull_name="3DFLAIR_noskull.nii.gz"
float_segmentation_name="T1w_norm_seg.nii.gz"
ref_mri_path="/storage/homefs/vu16r049/tasks/test_nibabel_preprocessed/imgs/*" 
ref_file_name="3DFLAIR.nii.gz"
output_path="/storage/homefs/vu16r049/tasks/test_nibabel_preprocessed/segm"
temp_path="/storage/homefs/vu16r049/tasks/test_nibabel_preprocessed/segm"

mkdir -p $output_path
mkdir -p $temp_path

#Register and transform segmentations between different patients
for ref_folder in $ref_mri_path
do
#	hd-bet -i $ref_folder/$float_flair_name -o $ref_folder/$float_noskull_name
	mkdir -p $output_path"/${ref_folder##*/}/"
	#Affine registration  
	"${nifty_path}/reg_aladin" -ref $ref_folder/$ref_file_name -flo $float_mri_path/${ref_folder##*/}/$float_flair_name -aff $temp_path/temp_aff_transform.txt -res $temp_path/temp_img.nii.gz
	#Non linear registration with affine initialization
	"${nifty_path}/reg_f3d" -ref $ref_folder/$ref_file_name -flo $float_mri_path/${ref_folder##*/}/$float_flair_name -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz" -cpp $temp_path"/temp_non_linear_transform.nii"
	#Transformation based on non linear registration. Inter=0 for nearest neighbor interpolation
	"${nifty_path}/reg_resample" -ref $ref_folder/$ref_file_name -flo $float_mri_path/${ref_folder##*/}/$float_segmentation_name -trans $temp_path"/temp_non_linear_transform.nii" -res $output_path"/${ref_folder##*/}/transformed_lesion.nii.gz" -inter 0

done

#Register multiple floating objects to one reference
#for ref_folder in $ref_mri_path
#do
#	for float_folder in $float_mri_path
#	do  
#		mkdir -p $output_path"/${ref_folder##*/}/${float_folder##*/}/"
#		#Affine registration  
#		"${nifty_path}/reg_aladin" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_flair_name -aff $temp_path/temp_aff_transform.txt -res $temp_path/temp_img.nii.gz
#		#Non linear registration with affine initialization
#		"${nifty_path}/reg_f3d" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_flair_name -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz" -cpp $temp_path"/temp_non_linear_transform.nii"
#		#Transformation based on non linear registration. Inter=0 for nearest neighbor interpolation
#		"${nifty_path}/reg_resample" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_segmentation_name -trans $temp_path"/temp_non_linear_transform.nii" -res $output_path"/${ref_folder##*/}/${float_folder##*/}/$float_segmentation_name" -inter 0		
#	done
#done

#If there are multiple floating objects to be transformed for one reference
#for ref_folder in $ref_mri_path
#do
#	for float_folder in $float_mri_path
#	do  
#		mkdir -p $output_path"/${ref_folder##*/}/${float_folder##*/}/"
#		#Affine registration  
#		"${nifty_path}/reg_aladin" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_flair_name -aff $temp_path/temp_aff_transform.txt -res $temp_path/temp_img.nii.gz
#		#Non linear registration with affine initialization
#		"${nifty_path}/reg_f3d" -ref $ref_folder/$ref_file_name -flo $float_folder/$float_flair_name -aff $temp_path"/temp_aff_transform.txt" -res $temp_path"/temp_img.nii.gz" -cpp $temp_path"/temp_non_linear_transform.nii"
#		for segmentation_file in $float_folder/ManualSegmentation*.nii.gz
#		do
#			#Transformation based on non linear registration. Inter=0 for nearest neighbor interpolation
#			"${nifty_path}/reg_resample" -ref $ref_folder/$ref_file_name -flo $segmentation_file -trans $temp_path"/temp_non_linear_transform.nii" -res $output_path"/${ref_folder##*/}/${float_folder##*/}/${segmentation_file##*/}" -inter 0
#		done
#	done
#done

rm $temp_path"/temp_img.nii.gz"
rm $temp_path"/temp_non_linear_transform.nii"
rm $temp_path"/temp_aff_transform.txt"

