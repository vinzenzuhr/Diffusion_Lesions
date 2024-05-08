import sys
from pathlib import Path
import nibabel as nib
import torch
from torchmetrics import Dice
from torch.utils.tensorboard import SummaryWriter
 
lesion_segm = list(Path(sys.argv[1]).rglob("*lesions2.nii.gz"))
lesion_segm = dict(zip([x.parent.stem for x in lesion_segm],lesion_segm))
gt_lesion_segm = list(Path(sys.argv[2]).rglob("*lesions2.nii.gz"))
gt_lesion_segm = dict(zip([x.parent.stem for x in gt_lesion_segm],gt_lesion_segm))
wm_segm = list(Path(sys.argv[3]).rglob("*.nii.gz"))
wm_segm = dict(zip([x.parent.stem for x in wm_segm],wm_segm)) 
tensorboard_output_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else None

# only keep segmentations which are in lesion_segm
gt_lesion_segm = {key: gt_lesion_segm[key] for key in gt_lesion_segm if key in lesion_segm.keys()}
wm_segm = {key: wm_segm[key] for key in wm_segm if key in lesion_segm.keys()}
assert len(lesion_segm.keys()) == len(gt_lesion_segm.keys()) == len(wm_segm.keys()), f"Length has to be equal. Got {len(lesion_segm.keys())}, {len(gt_lesion_segm.keys())}, {len(wm_segm.keys())}"

# create lesions and gt_lesions tensors
lesions = []
gt_lesions = []
for key in lesion_segm.keys():
    lesion = torch.from_numpy(nib.load(lesion_segm[key])).to(torch.int)
    gt_lesion = torch.from_numpy(nib.load(gt_lesion_segm[key])).to(torch.int)
    wm = torch.from_numpy(nib.load(wm_segm[key])).to(torch.int)
    binary_wm = torch.logical_or(wm==41, wm==2)
    gt_lesion = binary_wm * gt_lesion
 
    lesions.append(lesion)
    gt_lesions.append(gt_lesion)
lesions = torch.cat(lesions)
gt_lesions = torch.cat(gt_lesions)

# calculate and log dice score
dice = Dice(ignore_index=0, average="micro")
score = dice(lesions, gt_lesions)
print("dice: ", score)
if tensorboard_output_dir:
    tb_summary = SummaryWriter(tensorboard_output_dir, purge_step=0)
    tb_summary.add_scalar("dice", score, 0)