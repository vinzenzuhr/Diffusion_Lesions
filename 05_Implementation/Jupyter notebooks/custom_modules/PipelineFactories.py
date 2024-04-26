from RePaintPipeline import RePaintPipeline
from DDIMInpaintPipeline import DDIMInpaintPipeline
from DDIMGuidedPipeline import DDIMGuidedPipeline
from diffusers import RePaintScheduler
from GuidedRePaintPipeline import GuidedRePaintPipeline

def get_ddim_guided_pipeline(model, noise_scheduler):
    return DDIMGuidedPipeline(unet=model, scheduler=noise_scheduler)

def get_ddim_inpaint_pipeline(model, noise_scheduler):
    return DDIMInpaintPipeline(unet=model, scheduler=noise_scheduler)

def get_repaint_pipeline(model, noise_scheduler):
    return RePaintPipeline(unet=model, scheduler=RePaintScheduler())

def get_guided_repaint_pipeline(model, noise_scheduler):
    return GuidedRePaintPipeline(unet=model, scheduler=RePaintScheduler(), ddim_scheduler=noise_scheduler)