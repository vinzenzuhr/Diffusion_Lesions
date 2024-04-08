from diffusers import RePaintScheduler  
from Training import Training 
from RePaintPipeline import RePaintPipeline

class TrainingUnconditional(Training):
    def __init__(self, config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory, deactivate3Devaluation = True, evaluation_pipeline_parameters = {}):
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory)
        self.deactivate3Devaluation = deactivate3Devaluation
        self.evaluation_pipeline_parameters = evaluation_pipeline_parameters 

    def _get_training_input(self, batch, generator=None):
        clean_images = batch["gt_image"]

        noisy_images, noise, timesteps = self._get_noisy_images(clean_images, generator)
        
        return noisy_images, noise, timesteps

    def evaluate(self, pipeline=None):
        # Create pipeline if not given
        self.model.eval()
        if pipeline is None:
            pipeline = self.pipelineFactory(self.accelerator.unwrap_model(self.model), self.noise_scheduler) 
        pipeline = self.accelerator.prepare(pipeline)
        pipeline.to(self.accelerator.device) 
        
        # Evaluate 2D images
        if (self.epoch) % self.config.evaluate_epochs == 0 or self.epoch == self.config.num_epochs - 1: 
            eval = self.evaluation2D(
                self.config, 
                pipeline, 
                self.d2_eval_dataloader, 
                None if not self.accelerator.is_main_process else self.tb_summary, 
                self.accelerator,
                self)
            eval.evaluate(
                self.global_step, 
                parameters = self.evaluation_pipeline_parameters)

        
        # Evaluate 3D images composed of 2D slices
        if (not self.deactivate3Devaluation and ((self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1)): 
            eval = self.evaluation3D(
                self.config, 
                pipeline, 
                self.d3_eval_dataloader, 
                None if not self.accelerator.is_main_process else self.tb_summary, 
                self.accelerator)  
            eval.evaluate(
                self.global_step, 
                parameters = self.evaluation_pipeline_parameters)
        

        # Save model
        if self.accelerator.is_main_process:
            if pipeline is not None and ((self.epoch) % self.config.save_model_epochs == 0 or self.epoch == self.config.num_epochs - 1): 
                pipeline.save_pretrained(self.config.output_dir)
 