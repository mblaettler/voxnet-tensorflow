import shapenet10
from voxnet.model import get_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
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

    if os.path.isfile("test_data.npz"):
        with np.load("test_data.npz") as archive:
            data_test = archive["data"]
            labels_test = archive["labels"]
    else:
        data_test, labels_test = shapenet10.get_test_data()
        labels_test = to_categorical(labels_test)
        np.savez("test_data", data=data_test, labels=labels_test)

    save_checkpoint = ModelCheckpoint("checkpoints", monitor="val_loss", save_best_only=True)

    model = get_model((shapenet10.SIZE_X, shapenet10.SIZE_Y, shapenet10.SIZE_Z, 1), 40)

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting training...")
    log = model.fit(data, labels, epochs=50, batch_size=BATCH_SIZE, validation_data=(data_test, labels_test),
                    verbose=1, callbacks=[save_checkpoint])

    plt.figure(1)
    plt.plot(log.history['loss'], label='Training')
    plt.plot(log.history['val_loss'], label='Testing')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    train()
