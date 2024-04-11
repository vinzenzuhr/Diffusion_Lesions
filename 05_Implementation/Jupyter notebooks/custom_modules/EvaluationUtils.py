import torch
from skimage.metrics import structural_similarity
import os
 
def calc_metrics(images1, images2, masks):
    batch_size = images1.shape[0]

    metrics = dict()
    metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", "psnr_full", "psnr_out", "psnr_in"]
    for metric in metric_list:
        metrics[metric] = 0 

    for idx in range(batch_size):

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
        metrics["ssim_out"] += ssim_mat_out.sum() / torch.count_nonzero(ssim_mat_out)
        ssim_mat_in = sim_mat*masks[idx]
        metrics["ssim_in"] += ssim_mat_in.sum() / torch.count_nonzero(ssim_mat_in)

        # MSE metric 
        se_mat = (images1[idx] - images2[idx])**2
        mse_full = se_mat.mean()
        metrics["mse_full"] += mse_full 
        se_mat_out = se_mat*(1-masks[idx])
        mse_out = se_mat_out.sum() / torch.count_nonzero(se_mat_out)
        metrics["mse_out"] += mse_out
        se_mat_in = se_mat*masks[idx]
        mse_in = se_mat_in.sum() / torch.count_nonzero(se_mat_in) 
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