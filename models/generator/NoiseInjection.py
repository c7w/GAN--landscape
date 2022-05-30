import IPython
import jittor as jt
import jittor.nn as nn
from models.utils.utils import start_grad, stop_grad, weights_init_normal

class NoiseInjection(jt.Module):
    def __init__(self, original, channel):
        self.noise_weight = jt.zeros((1, channel, 1, 1))
        self.original = original

    def execute(self, image):
        image = self.original(image)
        noise = jt.randn_like(image)
        return image + self.noise_weight * noise
