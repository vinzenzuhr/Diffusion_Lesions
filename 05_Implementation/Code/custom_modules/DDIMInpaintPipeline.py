"""
A large part of the code originally comes from the official 
Huggingface Diffusers implementation 'huggingface/diffusers/src/diffusers/pipelines/ddim/pipeline_ddim.py' received from Github
"""

from typing import List, Optional, Tuple, Union

from diffusers import DiffusionPipeline, DDIMScheduler, ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
import torch

class DDIMInpaintPipeline(DiffusionPipeline):
    r"""
    Pipeline for image inpainting.

    Args:
        unet (UNet2DModel):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler (SchedulerMixin):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler = DDIMScheduler.from_config(scheduler.config)

        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        voided_imgs: torch.tensor,
        masks: torch.tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None, 
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            voided_imgs ('torch.tensor'): 
                Images (1 channel) which should be inpainted.
            masks ('torch.tensor'): 
                Binary masks which includes the area to inpaint.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers. A value of `0` corresponds to
                DDIM and `1` corresponds to DDPM.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                If `True` or `False`, see documentation for [`DDIMScheduler.step`]. If `None`, nothing is passed
                downstream to the scheduler (use `None` for schedulers which don't support this argument). 
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # Sample gaussian noise to begin loop. The second dimension represenenting the number of 
        # channels is subtracted by 2 to account for the mask and the image to be inpainted.
        batch_size = voided_imgs.shape[0]
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels-2,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels-2, *self.unet.config.sample_size)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        image = randn_tensor(image_shape, generator=generator, device=self._execution_device, dtype=self.unet.dtype)

        # Inputs to the unet model are the gaussian noise images, the voided images and the binary masks
        input=torch.cat((image, voided_imgs, masks), dim=1)

        for t in self.scheduler.timesteps:
            # 1. predict noise model_output
            model_output = self.unet(input, t).sample

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            ).prev_sample

            #3. Concatenate image with voided images and masks
            input=torch.cat((image, voided_imgs, masks), dim=1)

        image = image.clamp(-1, 1)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)