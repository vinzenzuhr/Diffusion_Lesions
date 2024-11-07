from abc import ABC, abstractmethod 
import os

from accelerate import Accelerator 
import nibabel as nib
from diffusers import DiffusionPipeline
import torch 
from torch.utils.data import DataLoader 
from tqdm.auto import tqdm

from custom_modules import EvaluationUtils, DatasetMRI, Logger

class Evaluation3D(ABC):
    """
    Abstract class for evaluating the performance of a 2D diffusion pipeline with 3D images.

    For evaluation the 3D images are sliced in 2D images and processed with the 2D pipeline. 
    The processed 2D images are then combined to a 3D image and metrics are calculated.

    Args:
        dataloader (DataLoader): DataLoader object for loading 3D evaluation data.
        logger (Logger): The logger object for logging.
        accelerator (Accelerator): The accelerator object for distributed training.
        output_dir (str): The output directory for saving results.
        filename (str): The filename for saving the processed nifti images.
        evaluate_num_batches (int, optional): The number of batches to evaluate. Defaults to -1 (all batches). 
    """

    def __init__(self, dataloader: DataLoader, logger: Logger, accelerator: Accelerator, output_dir: str, 
                 filename: str, evaluate_num_batches: int = -1):
        self.dataloader = dataloader
        self.logger = logger
        self.accelerator = accelerator
        self.output_dir = output_dir
        self.filename = filename
        self.evaluate_num_batches = evaluate_num_batches

        # create folder for segmentation algorithm afterwards
        segmentation_dir = os.path.join(self.output_dir, "segmentations_3D")
        os.makedirs(segmentation_dir, exist_ok=True)

    def _save_image(self, final_3d_image: nib.nifti1.Nifti1Image, save_dir: str, filename: str): 
        """
        Save the final 3D nifti image to a specified directory.

        Args:
            final_3d_images (nib.nifti1.Nifti1Image): Final 3D images to be saved.
            save_dir (str): Directory path to save the images.
            filename (str): Filename of the image.
        """
        nib.save(final_3d_image, f"{save_dir}/{filename}.nii.gz")

    @abstractmethod
    def _start_pipeline(self, clean_images: torch.tensor, masks: torch.tensor, parameters: dict,
                        ) -> tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Abstract method to start the pipeline for 3D evaluation.

        Args:
            clean_images (torch.tensor): The clean images tensor.
            masks (torch.tensor): The masks tensor.
            parameters (dict): The dictionary of parameters.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]: The processed images, clean images, slice indices, and masks.
        """
        pass

    def evaluate(self, pipeline: DiffusionPipeline, global_step: int, parameters: dict = {}):
        """
        Evaluate the diffusion pipeline and calculate metrics for 3D images.

        Args:
            pipeline (DiffusionPipeline): The diffusion pipeline object.
            global_step (int): The global step for logging.
            parameters (dict, optional): The dictionary of parameters for the evaluation.
        """
        # initialize metrics
        if self.accelerator.is_local_main_process:
            metrics = dict()
            metric_list = ["ssim_full", "ssim_out", "ssim_in", "mse_full", "mse_out", "mse_in", "psnr_full", "psnr_out", "psnr_in"]
            for metric in metric_list:
                metrics[metric] = 0
            num_iterations = 0
        self.progress_bar = tqdm(total=len(self.dataloader), disable=not self.accelerator.is_local_main_process)
        self.progress_bar.set_description(f"Evaluation 3D")

        print("Start 3D evaluation")
        for n_iter, batch in enumerate(self.dataloader):
            for sample_idx in torch.arange(batch["gt_image"].shape[0]):

                idx = batch["idx"][sample_idx]
                name = batch["name"][sample_idx]
                proc_info = batch["proc_info"][sample_idx]

                images, clean_images, slice_indices, masks = self._start_pipeline(pipeline, batch, sample_idx, parameters)

                # overwrite the original 3D image with the modified 2D slices
                final_3d_images = torch.clone(clean_images.detach())
                final_3d_images[:, :, slice_indices, :] = images

                # calculate metrics
                if self.accelerator.is_local_main_process:
                    new_metrics = EvaluationUtils.calc_metrics(clean_images, final_3d_images, masks)
                    for key, value in new_metrics.items():
                        metrics[key] += value
                    num_iterations += 1

                # postprocess and save image as nifti file
                final_3d_images = DatasetMRI.postprocess(final_3d_images.squeeze(), *proc_info, 
                                                         self.dataloader.dataset.get_metadata(int(idx)))
                save_dir = os.path.join(self.output_dir, f"samples_3D/{name}")
                os.makedirs(save_dir, exist_ok=True)
                self._save_image(final_3d_images, save_dir, self.filename)

            self.progress_bar.update(1)

            if (self.evaluate_num_batches != -1) and (n_iter >= self.evaluate_num_batches - 1):
                break

        # log metrics
        if self.accelerator.is_local_main_process:
            # calculate mean of metrics
            for key, value in metrics.items():
                metrics[key] /= num_iterations

            # rename metrics to 3D metrics
            dim3_metrics = dict()
            for key, value in metrics.items():
                dim3_metrics[f"{key}_3D"] = value

            self.logger.log_eval_metrics(global_step, dim3_metrics, self.output_dir)
        print("3D evaluation finished")