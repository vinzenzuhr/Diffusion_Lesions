import os

from skimage.metrics import structural_similarity
import torch 
 
def calc_metrics(gt_images: torch.tensor, gen_images: torch.tensor, masks: torch.tensor) -> dict:
    """
    Calculate evaluation metrics for a batch of generated images.

    Args:
        gt_images (torch.tensor): Ground truth images.
        gen_images (torch.tensor): Generated images.
        masks (torch.tensor): The masks indicating the regions of interest.

    Returns:
        dict: A dictionary containing the calculated metrics.

    """
    batch_size = gt_images.shape[0]

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
            gt_images[idx].detach().cpu().numpy(),
            gen_images[idx].detach().cpu().numpy(),
            channel_axis=0,
            data_range=2,
            full=True
        )
        sim_mat = torch.from_numpy(sim_mat).to(gt_images.device)
        metrics["ssim_full"] += sim_mat.mean() 
        ssim_mat_out = sim_mat * (1-masks[idx])
        metrics["ssim_out"] += ssim_mat_out.sum() / (1-masks[idx]).sum()
        ssim_mat_in = sim_mat * masks[idx]
        metrics["ssim_in"] += ssim_mat_in.sum() / masks[idx].sum()

        # MSE metric 
        se_mat = (gt_images[idx] - gen_images[idx])**2
        mse_full = se_mat.mean()
        metrics["mse_full"] += mse_full 
        se_mat_out = se_mat * (1-masks[idx])
        mse_out = se_mat_out.sum() / (1-masks[idx]).sum()
        metrics["mse_out"] += mse_out
        se_mat_in = se_mat * masks[idx]
        mse_in = se_mat_in.sum() / masks[idx].sum()
        metrics["mse_in"] += mse_in 

        # PSNR metric
        metrics["psnr_full"] += 10 * torch.log10(4 / mse_full) 
        metrics["psnr_out"] += 10 * torch.log10(4 / mse_out)
        metrics["psnr_in"] += 10 * torch.log10(4 / mse_in) 
    
    for metric in metric_list:
        metrics[metric] /= batch_size
        
    return metrics

def get_lesion_intensity(add_lesion_technique: str, image_lesions: torch.tensor):
    """
    Get the intensity value for the lesions based on the specified technique.

    Parameters:
    - add_lesion_technique (str): The technique to determine the lesion intensity.
        "other_lesions_1stQuantile": Use the first quantile of the lesion intensity.
        "other_lesions_mean": Use the mean of the lesion intensity.
        "other_lesions_median": Use the median of the lesion intensity.
        "other_lesions_3rdQuantile": Use the 3rd quantile of the lesion intensity.
        "other_lesions_99Quantile": Use the 99th quantile of the lesion intensity.
        "empty": Use 0 as the lesion intensity.
    - image_lesions (torch.tensor): The tensor containing the lesion images.

    Returns:
    - lesion_intensity (float): The determined lesion intensity based on the specified technique.

    Raises:
    - ValueError: If the add_lesion_technique is unknown.
    """

    if add_lesion_technique == "other_lesions_1stQuantile": 
        lesion_intensity = image_lesions.quantile(0.25) 
    elif add_lesion_technique == "other_lesions_mean": 
        lesion_intensity = image_lesions.mean() 
    elif add_lesion_technique == "other_lesions_median": 
        lesion_intensity = image_lesions.median() 
    elif add_lesion_technique == "other_lesions_3rdQuantile": 
        lesion_intensity = image_lesions.quantile(0.75) 
    elif add_lesion_technique == "other_lesions_99Quantile":
        lesion_intensity = image_lesions.quantile(0.99) 
    elif add_lesion_technique == "empty":
        lesion_intensity = 0
    else:
        raise ValueError("add_lesion_technique unknown")
    
    return lesion_intensity