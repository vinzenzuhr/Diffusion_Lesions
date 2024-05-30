from pathlib import Path
import shutil
import os

input_dir=Path("inpaint_challenge")
output_dir=Path("datasets_challenge/filling/dataset_train")

imgs = input_dir.rglob("*BraTS-GLI-*-t1n.nii.gz")
masks = input_dir.rglob("*BraTS-GLI-*-mask-healthy.nii.gz")

for img in imgs:
    print("Processing img ", img)
    folder = img.parent.name
    file = img.name
    output = output_dir / "imgs" / folder
    os.makedirs(output, exist_ok=True)
    shutil.copyfile(img, output / file)

for mask in masks:
    print("Processing mask ", mask)
    folder = mask.parent.name
    file = mask.name
    output = output_dir / "masks" / folder
    os.makedirs(output, exist_ok=True)
    shutil.copyfile(mask, output / file)