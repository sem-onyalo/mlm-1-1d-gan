'''
1D GAN.

Ref: https://machinelearningmastery.com/generative_adversarial_networks/
'''

from generator import createGenerator
from discriminator import createDiscriminator
from gan import createGan, train

if __name__ == '__main__':
    latentDim = 5
    discriminator = createDiscriminator()
    generator = createGenerator(latentDim)
    gan = createGan(discriminator, generator)
    train(discriminator, generator, gan, latentDim)
