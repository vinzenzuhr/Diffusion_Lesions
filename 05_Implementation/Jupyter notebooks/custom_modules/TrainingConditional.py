#import custom modules 
from custom_modules import Training

# import other modules 
import torch

class TrainingConditional(Training):
    def __init__(self, config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory, multi_sample=False):
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory, multi_sample)

    def _get_training_input(self, batch, generator=None, timesteps=None):
        clean_images = batch["gt_image"]
 
        masks = batch["mask"]

        noisy_images, noise, timesteps = self._get_noisy_images(clean_images, generator, timesteps)

        #create voided img
        voided_images = clean_images*(1-masks)

        # concatenate noisy_images, voided_images and mask
        input=torch.cat((noisy_images, voided_images, masks), dim=1)

        return input, noise, timesteps

    def evaluate(self, pipeline=None, deactivate_save_model=False):
        self.model.eval()
        # Create pipeline if not given
        if pipeline is None:
            pipeline = self.pipelineFactory(self.accelerator.unwrap_model(self.model), self.noise_scheduler) 
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device)
        
        # Evaluate 2D images
        if not self.config.deactivate2Devaluation and ((self.epoch) % self.config.evaluate_epochs == 0 or self.epoch == self.config.num_epochs - 1): 
            self.evaluation2D.evaluate(
                pipeline, 
                self.global_step, 
                self._get_training_input, 
                parameters = {},
                deactivate_save_model=deactivate_save_model)

        # Evaluate 3D images composed of 2D slices
        if (not self.config.deactivate3Devaluation and ((self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1)): 
            self.evaluation3D.evaluate(pipeline, self.global_step, 
                parameters = {})



 