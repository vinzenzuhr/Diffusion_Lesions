from collections.abc import Callable 

import torch

class ScaleDecorator(Callable):
    """
    A decorator class that scales the input image tensor, then performs a given transform and rescales the image afterwards.

    Args:
        transform (torch.nn.Module): The transform module to be applied to the image tensor. 
    """
    
    def __init__(self, transform: torch.nn.Module):
        self.transform = transform
    
    def __call__(self, img: torch.tensor) -> torch.tensor:
        return self.transform(img/2+0.5)*2-1
