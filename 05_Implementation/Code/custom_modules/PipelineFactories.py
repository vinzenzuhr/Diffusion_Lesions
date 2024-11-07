from custom_modules import RePaintPipeline
from custom_modules import DDIMInpaintPipeline
from custom_modules import GuidedPipelineUnconditional
from custom_modules import GuidedPipelineConditional
from custom_modules import GuidedRePaintPipeline

from diffusers import RePaintScheduler

def get_guided_unconditional_pipeline(model, noise_scheduler):
    return GuidedPipelineUnconditional(unet=model, scheduler=noise_scheduler)


def get_guided_conditional_pipeline(model, noise_scheduler):
    return GuidedPipelineConditional(unet=model, scheduler=noise_scheduler)


def get_ddim_inpaint_pipeline(model, noise_scheduler):
    return DDIMInpaintPipeline(unet=model, scheduler=noise_scheduler)


def get_repaint_pipeline(model, noise_scheduler):
    return RePaintPipeline(unet=model, scheduler=RePaintScheduler())


def get_guided_repaint_pipeline(model, noise_scheduler):
    return GuidedRePaintPipeline(unet=model, scheduler=RePaintScheduler(), ddim_scheduler=noise_scheduler)