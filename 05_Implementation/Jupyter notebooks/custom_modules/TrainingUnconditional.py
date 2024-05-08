from custom_modules import Training  

class TrainingUnconditional(Training):
    def __init__(self, config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory, multi_sample=False, deactivate3Devaluation = True, evaluation_pipeline_parameters = {}):
        super().__init__(config, model, noise_scheduler, optimizer, lr_scheduler, datasetTrain, datasetEvaluation, dataset3DEvaluation, evaluation2D, evaluation3D, pipelineFactory, multi_sample)
        self.deactivate3Devaluation = deactivate3Devaluation
        self.evaluation_pipeline_parameters = evaluation_pipeline_parameters 

    def _get_training_input(self, batch, generator=None):
        clean_images = batch["gt_image"]

        noisy_images, noise, timesteps = self._get_noisy_images(clean_images, generator)
        
        return noisy_images, noise, timesteps

    def evaluate(self, pipeline=None, deactivate_save_model=False):
        # Create pipeline if not given
        self.model.eval()
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
                parameters = self.evaluation_pipeline_parameters)
        
        # Evaluate 3D images composed of 2D slices
        if (not self.config.deactivate3Devaluation and ((self.epoch) % self.config.evaluate_3D_epochs == 0 or self.epoch == self.config.num_epochs - 1)): 
            self.evaluation3D.evaluate(
                pipeline, 
                self.global_step, 
                parameters = self.evaluation_pipeline_parameters)
 