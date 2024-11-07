from typing import Union

import torch
from diffusers import DDIMScheduler, DDPMScheduler

class ModelInputGenerator:
    """
    Generates model inputs, depending on whether the model is conditional or unconditional.

    Args:
        conditional (bool, optional): Whether the model is conditional or unconditional. Defaults to False.
        noise_scheduler (Union[DDIMScheduler, DDPMScheduler]): The noise scheduler used for adding noise to the images.
    """

    def __init__(self, conditional: bool, noise_scheduler: Union[DDIMScheduler, DDPMScheduler]): 
        self.conditional = conditional   
        self.noise_scheduler = noise_scheduler

    def get_model_input(self, batch: torch.tensor, generator: torch.Generator = None, 
                            timesteps: torch.tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the training input for the model, depending on whether it is conditional or unconditional.

        The input for the conditional model consists of the noisy images and the concatenation of the voided images and masks to be conditioned.
        The input for the unconditional model consists of the noisy images only.

        Args:
            batch (torch.tensor): The input batch of data.
            generator (torch.Generator, optional): The random number generator. Defaults to None.
            timesteps (torch.tensor, optional): Predefined timesteps for diffusing each image. 
                Defaults to None.

        Returns:
            tuple[torch.tensor, torch.tensor, torch.tensor] A tuple containing the input, noise, and timesteps.

        """
        clean_images = batch["gt_image"] 
        noisy_images, noise, timesteps = self.get_noisy_images(clean_images, generator, timesteps)
        if self.conditional:
            masks = batch["mask"] 
            voided_images = clean_images*(1-masks)
            input=torch.cat((noisy_images, voided_images, masks), dim=1)
        else:
            input = noisy_images
        return input, noise, timesteps

    def get_noisy_images(self, clean_images: torch.tensor, generator: torch.Generator = None, 
                          timesteps: torch.tensor = None) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Generates noisy images by adding random gaussian noise to the clean images. 

        Args:
            clean_images (torch.tensor): The clean images to add noise to.
            generator (torch.Generator, optional): The random number generator. Defaults to None.
            timesteps (torch.tensor, optional): Predefined timesteps for diffusing each image. 
                Defaults to None.

        Returns:
            tuple[torch.tensor, torch.tensor, torch.tensor]: A tuple containing the noisy images, 
                the used noise, and the timesteps.
        """
        
        # Sample noise to add to the images 
        noise = torch.randn(clean_images.shape, device=clean_images.device, generator=generator)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        if timesteps is None:
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64, generator=generator
            )
        assert timesteps.shape[0] == bs 

        # Add noise to the voided images according to the noise magnitude at each timestep (forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps) 
        
        return noisy_images, noise, timesteps