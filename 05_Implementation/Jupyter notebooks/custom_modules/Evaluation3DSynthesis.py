from Evaluation3D import Evaluation3D
import torch
import numpy as np

class Evaluation3DSynthesis(Evaluation3D):
    def __init__(self, config, pipeline, dataloader, tb_summary, accelerator):
        super().__init__(config, pipeline, dataloader, tb_summary, accelerator)

    def _add_lesions(self, clean_images, masks):
        #add lesions
        images_with_lesions = clean_images.clone()
        images_with_lesions[masks] = -1
        return images_with_lesions

    def _start_pipeline(self, clean_images, masks, parameters):
        #add coarse lesions
        images_with_lesions = self._add_lesions(clean_images, masks)

        #run it through network
        synthesized_images = self.pipeline(
            images_with_lesions,
            masks,
            generator=torch.cuda.manual_seed_all(self.config.seed),
            output_type=np.array,
            num_inference_steps = self.config.num_inference_steps,
            **parameters
        ).images
        synthesized_images = torch.from_numpy(synthesized_images).to(clean_images.device) 
        return synthesized_images