import jittor as jt
import jittor.nn as nn
from models.utils.utils import start_grad, stop_grad, weights_init_normal

class PseudoJudger(nn.Module):

    def __init__(self, in_channels=3):
        super(PseudoJudger, self).__init__()

        # TODO: implement the segmentor
        # Implement credit to: https://github.com/Jittor/segmentation-jittor
        self.segmentor = nn.Sequential(
            nn
        )

        # TODO: implement FID calculator
        # TODO: implement beauty score calculator
        # What is a beauty score? I do not know how to translate it into English :(

        # Init Parameters
        for m in self.modules():
            weights_init_normal(m)


    def execute(self, input):
        return self.model(input)