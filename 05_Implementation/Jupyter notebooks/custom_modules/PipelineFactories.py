from custom_modules import RePaintPipeline
from custom_modules import DDIMInpaintPipeline
from custom_modules import DDIMGuidedPipeline
from custom_modules import GuidedRePaintPipeline

from diffusers import RePaintScheduler

def get_ddim_guided_pipeline(model, noise_scheduler):
    return DDIMGuidedPipeline(unet=model, scheduler=noise_scheduler)

def get_ddim_inpaint_pipeline(model, noise_scheduler):
    return DDIMInpaintPipeline(unet=model, scheduler=noise_scheduler)

def get_repaint_pipeline(model, noise_scheduler):
    return RePaintPipeline(unet=model, scheduler=RePaintScheduler())

def get_guided_repaint_pipeline(model, noise_scheduler):
    return GuidedRePaintPipeline(unet=model, scheduler=RePaintScheduler(), ddim_scheduler=noise_scheduler)