from Evaluation3D import Evaluation3D
import torch
import numpy as np

class Evaluation3DFilling(Evaluation3D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator):
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator)

    def _start_pipeline(self, clean_images, masks, parameters={}):
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