from keras.models import Sequential
from keras.layers import Dense

def createGenerator(n_inputs=100, n_outputs=2):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(n_outputs, activation='linear'))
    return model

if __name__ == '__main__':
    generator = createGenerator()
    generator.summary()