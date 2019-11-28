from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, BatchNormalization, LeakyReLU, MaxPooling3D, Flatten, Dropout


def get_model(input_size, num_classes):
    model = Sequential()

    model.add(Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding="valid", input_shape=input_size))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Conv3D(32, (3, 3, 3), strides=(1, 1, 1), padding="valid"))
    model.add(LeakyReLU(alpha=0.1))

    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="softmax"))

    return model
