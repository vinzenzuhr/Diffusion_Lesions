from Evaluation2D import Evaluation2D
import torch
import numpy as np

class Evaluation2DFilling(Evaluation2D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator, train_env):
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator, train_env)  

    def _start_pipeline(self, clean_images, masks, segmentation=None, parameters={}):
        #segmentation is not needed
        voided_images = clean_images*(1-masks)

        inpainted_images = self.pipeline(
            voided_images,
            masks,
            generator=torch.cuda.manual_seed_all(self.config.seed),
            output_type=np.array,
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images
        inpainted_images = torch.from_numpy(inpainted_images).to(clean_images.device) 
        return inpainted_images
