#!/bin/bash

#Calculate cortical thickness using the ANTS pipeline
#Output is a thickness map and a segmentation image, which are saved in the output directory.


t1_dir="temp_unhealthy_tysabri_T1/*"
template_dir="ANTS_atropos_template"
output_path="temp_unhealthy_tysabri_T1_CTh_old"

mkdir -p $output_path

for t1_folder in $t1_dir
do
    echo processing $t1_folder...
    mkdir -p $output_path/${t1_folder##*/}/
    antsCorticalThickness.sh -d 3 -a $t1_folder/T1.nii.gz -o $output_path/${t1_folder##*/}/ -e $template_dir/T_template0.nii.gz -t $template_dir/T_template0_BrainCerebellum.nii.gz -m $template_dir/T_template0_BrainCerebellumProbabilityMask.nii.gz -f $template_dir/T_template0_BrainCerebellumExtractionMask.nii.gz -p $template_dir/Priors2/priors%d.nii.gz
done