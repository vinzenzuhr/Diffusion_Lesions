#import custom modules
from Evaluation2D import Evaluation2D
from Evaluation3D import Evaluation3D

# import other modules
from diffusers import RePaintPipeline, RePaintScheduler  
from Training import Training 

class TrainingUnconditional(Training):
    def __init__(self, config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, deactivate3Devaluation = True):
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation)
        self.deactivate3Devaluation = deactivate3Devaluation

    def _get_training_input(self, batch):
        clean_images = batch["gt_image"]

        noisy_images, noise, timesteps = self._get_noisy_images(clean_images)
        
        return noisy_images, noise, timesteps

    def evaluate(self, pipeline=None):
        # Create pipeline if not given
        self.model.eval()
        if pipeline is None:
            pipeline = RePaintPipeline(unet=self.accelerator.unwrap_model(self.model), scheduler=RePaintScheduler())
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device)

        # Evaluate 2D images
        if (self.epoch) % self.config.evaluate_epochs == 0 or self.epoch == self.config.num_epochs - 1: 
            eval = Evaluation2D(
                self.config, 
                pipeline, 
                self.d2_eval_dataloader, 
                None if not self.accelerator.is_main_process else self.tb_summary, 
                self.accelerator)
            eval.evaluate(
                self.epoch, 
                self.global_step, 
                parameters = {
                    "jump_length": self.config.jump_length,
                    "jump_n_sample": self.config.jump_n_sample,
                })

        
        # Evaluate 3D images composed of 2D slices
        if (not self.deactivate3Devaluation and ((self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1)): 
            eval = Evaluation3D(
                self.config, 
                pipeline, 
                self.d3_eval_dataloader)  
            eval.evaluate(
                self.epoch, 
                parameters = {
                    "jump_length": self.config.jump_length,
                    "jump_n_sample": self.config.jump_n_sample,
                })
        

        # Save model
        if self.accelerator.is_main_process:
            if pipeline is not None and ((self.epoch) % self.config.save_model_epochs == 0 or self.epoch == self.config.num_epochs - 1): 
                pipeline.save_pretrained(self.config.output_dir)
 