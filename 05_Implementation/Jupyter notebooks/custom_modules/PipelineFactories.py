from RePaintPipeline import RePaintPipeline
from DDIMInpaintPipeline import DDIMInpaintPipeline
from DDIMGuidedPipeline import DDIMGuidedPipeline
from diffusers import RePaintScheduler

def get_ddim_guided_pipeline(model, noise_scheduler):
    return DDIMGuidedPipeline(unet=model, scheduler=noise_scheduler)

def get_ddim_inpaint_pipeline(model, noise_scheduler):
    return DDIMInpaintPipeline(unet=model, scheduler=noise_scheduler)

def get_repaint_pipeline(model, noise_scheduler):
    return RePaintPipeline(unet=model, scheduler=RePaintScheduler())