import IPython
import jittor as jt
import jittor.nn as nn
from models.utils.utils import start_grad, stop_grad, weights_init_normal
from .NoiseInjection import NoiseInjection


class UNetGeneratorNoise(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetGeneratorNoise, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlockNoise(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlockNoise(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlockNoise(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockNoise(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlockNoise(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlockNoise(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def execute(self, input):
        # print(input.shape)
        return self.model(input)

class UnetSkipConnectionBlockNoise(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlockNoise, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc



        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [NoiseInjection(downconv, inner_nc)]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, NoiseInjection(downconv, inner_nc)]
            up = [uprelu, NoiseInjection(upconv, outer_nc), upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, NoiseInjection(downconv, inner_nc), downnorm]
            up = [uprelu, NoiseInjection(upconv, outer_nc), upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def execute(self, x):
        # print(x.shape)
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return jt.contrib.concat([x, self.model(x)], 1)

        # new line