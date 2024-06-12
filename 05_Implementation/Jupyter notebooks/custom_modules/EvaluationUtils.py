import os
from pathlib import Path

from diffusers.utils import make_image_grid
import PIL 
from skimage.metrics import structural_similarity
import torch
from torch.utils.tensorboard import SummaryWriter 
 
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


def log_metrics(tb_summary: SummaryWriter, global_step: int, metrics: dict, config):
    """
    Logs the metrics to TensorBoard and optionally to a CSV file.

    Args:
        tb_summary (SummaryWriter): The TensorBoard summary writer.
        global_step (int): The global step value.
        metrics (dict): A dictionary containing the metrics to be logged.
        config: The configuration object.
    """
    for key, value in metrics.items(): 
        tb_summary.add_scalar(key, value, global_step) 

    if config.log_csv:
        with open(os.path.join(config.output_dir, "metrics.csv"), "a") as f:
            for key, value in metrics.items():
                f.write(f"{key}:{value},")
            f.write(f"global_step:{global_step}")
            f.write("\n")


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


def save_image(images: list[list[PIL.Image]], titles: list[str], path: Path, global_step: int, img_shape: tuple[int, int]):
    """
    Save a grid of images to the specified path.

    Args:
        images (list[list[PIL.Image]]): A list of lists of PIL.Image objects representing the images to be saved.
        titles (list[str]): A list of titles for each image grid.
        path (Path): The path where the images will be saved.
        global_step (int): The global step used for naming the saved images.
        img_shape (tuple[int, int]): The shape of the images in the grid.

    Raises:
        ValueError: If the number of images in the list is greater than 16.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
    for image_list, title in zip(images, titles):             
        if len(image_list) > 16:
            raise ValueError("Number of images in list must be less than 16")
        missing_num = 16 - len(image_list)
        for _ in range(missing_num):
            image_list.append(PIL.Image.new("L", img_shape, 0))
        image_grid = make_image_grid(image_list, rows=4, cols=4)
        image_grid.save(f"{path}/{title}_{global_step:07d}.png")
    print("image saved")