import shapenet10
from voxnet.model import get_model
from tensorflow.keras.utils import to_categorical
import os
import numpy as np


BATCH_SIZE = 128

def train():
    if os.path.isfile("train_data.npz"):
        with np.load("train_data.npz") as archive:
            data = archive["data"]
            labels = archive["labels"]
    else:
        data, labels = shapenet10.get_training_data()
        labels = to_categorical(labels)
        np.savez("train_data", data=data, labels=labels)

    model = get_model((shapenet10.SIZE_X, shapenet10.SIZE_Y, shapenet10.SIZE_Z, 1), 40)

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting training...")
    model.fit(data, labels, epochs=100, batch_size=BATCH_SIZE)


if __name__ == "__main__":
    train()
