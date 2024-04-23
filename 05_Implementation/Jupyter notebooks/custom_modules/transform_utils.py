from collections.abc import Callable 

class ScaleDecorator(Callable):
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, img):
        return self.transform(img/2+0.5)*2-1
