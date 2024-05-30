import torch
from skimage.metrics import structural_similarity
import os
import PIL 
from diffusers.utils import make_image_grid
from pathlib import Path
 
def calc_metrics(images1, images2, masks):
    batch_size = images1.shape[0]

    metrics = dict()
    metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", "psnr_full", "psnr_out", "psnr_in"]
    for metric in metric_list:
        metrics[metric] = 0 

    for idx in range(batch_size):

        # skip images with no mask
        if (masks[idx].sum() == 0):
            batch_size -= 1
            continue

        # SSIM metric
        _, sim_mat = structural_similarity(
            images1[idx].detach().cpu().numpy(),
            images2[idx].detach().cpu().numpy(),
            channel_axis=0,
            data_range=2,
            full=True
        )
        sim_mat = torch.from_numpy(sim_mat).to(images1.device)
        metrics["ssim_full"] += sim_mat.mean() 
        ssim_mat_out = sim_mat*(1-masks[idx])
        metrics["ssim_out"] += ssim_mat_out.sum() / (1-masks[idx]).sum()
        ssim_mat_in = sim_mat*masks[idx]
        metrics["ssim_in"] += ssim_mat_in.sum() / masks[idx].sum()

        # MSE metric 
        se_mat = (images1[idx] - images2[idx])**2
        mse_full = se_mat.mean()
        metrics["mse_full"] += mse_full 
        se_mat_out = se_mat*(1-masks[idx])
        mse_out = se_mat_out.sum() / (1-masks[idx]).sum()
        metrics["mse_out"] += mse_out
        se_mat_in = se_mat*masks[idx]
        mse_in = se_mat_in.sum() / masks[idx].sum()
        metrics["mse_in"] += mse_in 

        # PSNR metric
        metrics["psnr_full"] += 10*torch.log10(4/mse_full) 
        metrics["psnr_out"] += 10*torch.log10(4/mse_out)
        metrics["psnr_in"] += 10*torch.log10(4/mse_in) 
    
    for metric in metric_list:
        metrics[metric] /= batch_size
        
    return metrics

def log_metrics(tb_summary, global_step, metrics, config):
    """
    Log metrics to tensorboard and print them to console

    Args:
        tb_summary: tensorboard SummaryWriter
        global_step: int
        metrics: dict with metric name as key and metric value as value
    """

    for key, value in metrics.items():
        print(f"{key}: {value}")
        tb_summary.add_scalar(key, value, global_step)
    print("global_step: ", global_step)

    if config.log_csv:
        with open(os.path.join(config.output_dir, "metrics.csv"), "a") as f:
            for key, value in metrics.items():
                f.write(f"{key}:{value},")
            f.write(f"global_step:{global_step}")
            f.write("\n")

def get_lesion_technique(add_lesion_technique, image_lesions, lesion_intensity = None):
    if add_lesion_technique == "mean_intensity":
        return lesion_intensity
    elif add_lesion_technique == "other_lesions_1stQuantile":
        # use first quantile of lesion intensity as new lesion intensity
        lesion_intensity = image_lesions.quantile(0.25)
        print("1st quantile lesion intensity: ", lesion_intensity)
    elif add_lesion_technique == "other_lesions_mean":
        # use mean of lesion intensity as new lesion intensity
        lesion_intensity = image_lesions.mean()
        print("mean lesion intensity: ", lesion_intensity)
    elif add_lesion_technique == "other_lesions_median":
        # use mean of lesion intensity as new lesion intensity
        lesion_intensity = image_lesions.median()
        print("median lesion intensity: ", lesion_intensity)
    elif add_lesion_technique == "other_lesions_3rdQuantile":
        # use 3rd quantile of lesion intensity as new lesion intensity
        lesion_intensity = image_lesions.quantile(0.75)
        print("3rd quantile lesion intensity: ", lesion_intensity)
    elif add_lesion_technique == "other_lesions_99Quantile":
        lesion_intensity = image_lesions.quantile(0.99)
        print("0.99 quantile lesion intensity: ", lesion_intensity)
    else:
        raise ValueError("add_lesion_technique unknown")
    
    return lesion_intensity

def save_image(images: list[list[PIL.Image]], titles: list[str], path: Path, global_step: int, img_shape: tuple[int, int]):
    os.makedirs(path, exist_ok=True)
    for image_list, title in zip(images, titles):             
        if len(image_list) > 4:
            ValueError("Number of images in list must be less than 4")
        missing_num = 4-len(image_list)
        for _ in range(missing_num):
            image_list.append(PIL.Image.new("L", img_shape, 0))
        image_grid = make_image_grid(image_list, rows=2, cols=2)
        image_grid.save(f"{path}/{title}_{global_step:07d}.png")
    print("image saved")