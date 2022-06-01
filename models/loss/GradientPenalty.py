import numpy as np
import jittor as jt

def DiscriminatorLossGradientPenalty(**kwargs):
    """Calculates the gradient penalty loss for WGAN GP"""

    fake_samples = kwargs['fake_img']
    real_samples = kwargs['real_img']
    D = kwargs['discriminator']

    alpha = jt.array(np.random.random((real_samples.shape[0], 1, 1, 1)).astype('float32'))
    interpolates = ((alpha * real_samples) + ((1 - alpha) * fake_samples))
    d_interpolates = D(interpolates)
    gradients = jt.grad(d_interpolates, interpolates)
    gradients = gradients.reshape((gradients.shape[0], (- 1)))
    gp =((jt.sqrt((gradients.sqr()).sum(1))-1).sqr()).mean()
    return gp
