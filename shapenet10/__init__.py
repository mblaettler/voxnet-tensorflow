import os
import numpy as np


DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

labels = {
    "airplane": 0,
    "bathtub": 1,
    "bed": 2,
    "bench": 3,
    "bookshelf": 4,
    "bottle": 5,
    "bowl": 6,
    "car": 7,
    "chair": 8,
    "cone": 9,
    "cup": 10,
    "curtain": 11,
    "desk": 12,
    "door": 13,
    "dresser": 14,
    "flower_pot": 15,
    "glass_box": 16,
    "guitar": 17,
    "keyboard": 18,
    "lamp": 19,
    "laptop": 20,
    "mantel": 21,
    "monitor": 22,
    "night_stand": 23,
    "person": 24,
    "piano": 25,
    "plant": 26,
    "radio": 27,
    "range_hood": 28,
    "sink": 29,
    "sofa": 30,
    "stairs": 31,
    "stool": 32,
    "table": 33,
    "tent": 34,
    "toilet": 35,
    "tv_stand": 36,
    "vase": 37,
    "wardrobe": 38,
    "xbox": 39
}


SIZE_X = 30
SIZE_Y = 30
SIZE_Z = 30

NUM_CLASSES = 40


def __get_data(path):
    data = []
    lbl = []

    for root, _, files in os.walk(path):
        for f in files:
            sample_path = os.path.join(root, f)
            sample = np.load(sample_path)
            sample = sample.reshape((SIZE_X, SIZE_Y, SIZE_Z, 1))

            data.append(sample)
            lbl.append(labels[sample_path.split(os.path.sep)[-2]])

    return data, lbl


def get_training_data():
    train_path = os.path.join(DATA_PATH, "train")
    return __get_data(train_path)


def get_test_data():
    train_path = os.path.join(DATA_PATH, "test")
    return __get_data(train_path)
