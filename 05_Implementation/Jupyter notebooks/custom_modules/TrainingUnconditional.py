from custom_modules import Training, EvaluationUtils
from diffusers import DDIMPipeline  
import torch  
import os

class TrainingUnconditional(Training):
    def __init__(self, config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory, multi_sample=False, deactivate3Devaluation = True, evaluation_pipeline_parameters = {}):
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory, multi_sample)
        self.deactivate3Devaluation = deactivate3Devaluation
        self.evaluation_pipeline_parameters = evaluation_pipeline_parameters 

    def _get_training_input(self, batch, generator=None, timesteps=None):
        clean_images = batch["gt_image"]

        noisy_images, noise, timesteps = self._get_noisy_images(clean_images, generator, timesteps)
        
        return noisy_images, noise, timesteps
    
    def _save_unconditional_img(self, pipeline):
        unconditional_pipeline = DDIMPipeline(
            unet=pipeline.unet, 
            scheduler=self.noise_scheduler)
        unconditional_pipeline = self.accelerator.prepare(unconditional_pipeline)
        #unconditional_pipeline.to(self.accelerator.device)     
        images = unconditional_pipeline(
            batch_size = self.config.eval_batch_size,
            generator=torch.Generator(device=self.accelerator.device).manual_seed(self.config.seed), 
            output_type = "pil",
            num_inference_steps = self.config.num_inference_steps
        )[0] 
        EvaluationUtils.save_image([images],["unconditional_images"], os.path.join(self.config.output_dir, "samples_2D"), self.global_step, self.config.unet_img_shape)

    def evaluate(self, pipeline=None, deactivate_save_model=False): 
        self.model.eval()
        # Create pipeline if not given
        if pipeline is None:
            pipeline = self.pipelineFactory(self.accelerator.unwrap_model(self.model), self.noise_scheduler) 
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device) 
        
        # Evaluate 2D images
        if (self.epoch) % self.config.evaluate_epochs == 0 or self.epoch == self.config.num_epochs - 1: 
            self.evaluation2D.evaluate(
                pipeline, 
                self.global_step, 
                self._get_training_input,
                parameters = self.evaluation_pipeline_parameters,
                deactivate_save_model=deactivate_save_model)
            self._save_unconditional_img(pipeline)
        
        # Evaluate 3D images composed of 2D slices
        if (not self.config.deactivate3Devaluation and ((self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1)): 
            self.evaluation3D.evaluate(
                pipeline, 
                self.global_step, 
                parameters = self.evaluation_pipeline_parameters )
        
 