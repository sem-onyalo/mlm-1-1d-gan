from keras.models import Sequential
from numpy.random import rand
from numpy.random import randn
from numpy import hstack
from numpy import zeros
from numpy import ones

def getRealSamples(n):
    # generate inputs in [-0.5, 0.5]
    X1 = rand(n) - 0.5
    # generate outputs X^2
    X2 = X1 * X1
    # stack arrays
    X1 = X1.reshape(n, 1)
    X2 = X2.reshape(n, 1)
    X = hstack((X1, X2))
    # generate class labels
    y = ones((n, 1))
    return X, y

def generateLatentPoints(latentDim, numSamples):
    xInput = randn(latentDim * numSamples)
    xInput = xInput.reshape((numSamples, latentDim))
    return xInput

def generateFakeSamples(generator: Sequential, latentDim, n):
    xInput = generateLatentPoints(latentDim, n)
    # predict outputs
    X = generator.predict(xInput)
    # create class labels
    y = zeros((n, 1))
    return X, y