#import custom modules
from DDIMInpaintPipeline import DDIMInpaintPipeline
from Evaluation2D import Evaluation2D
from Evaluation3D import Evaluation3D

# import other modules
from Training import Training
import torch

class TrainingConditional(Training):
    def __init__(self, config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, trainingCircularMasks):
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation)
        self.trainingCircularMasks = trainingCircularMasks

    def _get_training_input(self, batch):
        clean_images = batch["gt_image"]

        if self.trainingCircularMasks:
            masks = self._get_random_masks(clean_images.shape[0])
            masks = masks.to(clean_images.device)
        else:
            masks = batch["mask"]

        noisy_images, noise, timesteps = self._get_noisy_images(clean_images)

        #create voided img
        voided_images = clean_images*masks

        # concatenate noisy_images, voided_images and mask
        input=torch.cat((noisy_images, voided_images, masks), dim=1)

        return input, noise, timesteps

    def evaluate(self, pipeline=None):
        # Create pipeline if not given
        self.model.eval()
        if pipeline is None:
            pipeline = DDIMInpaintPipeline(unet=self.accelerator.unwrap_model(self.model), scheduler=self.noise_scheduler)
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device)
        
        # Evaluate 2D images
        if (self.epoch) % self.config.evaluate_epochs == 0 or self.epoch == self.config.num_epochs - 1: 
            eval = Evaluation2D(self.config, pipeline, self.d2_eval_dataloader, None if not self.accelerator.is_main_process else self.tb_summary, self.accelerator)
            eval.evaluate(self.epoch, self.global_step)

        # Evaluate 3D images composed of 2D slices
        if (self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1: 
            eval = Evaluation3D(self.config, pipeline, self.d3_eval_dataloader)  
            eval.evaluate(self.epoch)
    
        # Save model
        if self.accelerator.is_main_process:
            if pipeline is not None and ((self.epoch) % self.config.save_model_epochs == 0 or self.epoch == self.config.num_epochs - 1): 
                pipeline.save_pretrained(self.config.output_dir)



 