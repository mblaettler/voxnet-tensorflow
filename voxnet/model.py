from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, BatchNormalization, LeakyReLU, MaxPooling3D, Flatten


def get_model(input_size, num_classes):
    model = Sequential()

    model.add(Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding="same", input_shape=input_size))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha=0.1))

    model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding="same"))
    model.add(BatchNormalization(axis=1))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dense(num_classes), activation="softmax")

    return model
