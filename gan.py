from numpy import ones
from data import generateFakeSamples, generateLatentPoints, getRealSamples
from generator import createGenerator
from discriminator import createDiscriminator
from keras.models import Sequential
from matplotlib import pyplot

def createGan(discriminator: Sequential, generator: Sequential):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def evaluatePerformance(epoch, discriminator: Sequential, generator: Sequential, latentDim, n=100):
    xReal, yReal = getRealSamples(n)
    _, accReal = discriminator.evaluate(xReal, yReal, verbose=0)

    xFake, yFake = generateFakeSamples(generator, latentDim, n)
    _, accFake = discriminator.evaluate(xFake, yFake, verbose=0)

    print(epoch, accReal, accFake)

    pyplot.scatter(xReal[:, 0], xReal[:, 1], color='red')
    pyplot.scatter(xFake[:, 0], xFake[:, 1], color='blue')
    filename = 'eval/generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

def train(discriminator:Sequential, generator:Sequential, gan:Sequential, latentDim, numEpochs=10000, numBatch=128, evalFreq=2000):
    halfBatch = int(numBatch / 2)
    for i in range(numEpochs):
        xReal, yReal = getRealSamples(halfBatch)
        discriminator.train_on_batch(xReal, yReal)

        xFake, yFake = generateFakeSamples(generator, latentDim, halfBatch)
        discriminator.train_on_batch(xFake, yFake)
    
        xGan = generateLatentPoints(latentDim, halfBatch)
        yGan = ones((halfBatch, 1))
        gan.train_on_batch(xGan, yGan)

        if (i+1) % evalFreq == 0:
            evaluatePerformance(i, discriminator, generator, latentDim)

if __name__ == '__main__':
    discriminator = createDiscriminator()
    generator = createGenerator()
    gan = createGan(discriminator, generator)
    gan.summary()