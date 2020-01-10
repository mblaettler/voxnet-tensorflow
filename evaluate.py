import SVHDProvider as dataset
from voxnet.model import get_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import os
import numpy as np

BATCH_SIZE = 128

data_type = "FPS"   # or TIC


def shuffle_data(data, lbl):
    idx = np.arange(len(lbl))
    np.random.shuffle(idx)
    return data[idx, ...], lbl[idx]


def evaluate():
    if os.path.isfile(f"{data_type}_test_data.npz"):
        with np.load(f"{data_type}_test_data.npz") as archive:
            data_test = archive["data"]
            labels_test = archive["labels"]
            sample_names = archive["sample_names"]
    else:
        data_test, labels_test, sample_names = dataset.get_test_data(data_type)
        data_test = np.stack(data_test)  # convert to numpy array
        labels_test = to_categorical(labels_test)
        np.savez(f"{data_type}_test_data", data=data_test, labels=labels_test, sample_names=sample_names)

    model = load_model(os.path.join("checkpoints", data_type, "model.h5"))

    data_test = np.reshape(data_test,
                           (data_test.shape[0], dataset.SIZE_X, dataset.SIZE_Y, dataset.SIZE_Z, 1)).astype(np.float32)
    predicted_lbls = model.predict(data_test)

    eval_log = ""
    for i in range(len(sample_names)):
        sample = sample_names[i]
        lbl = np.argmax(labels_test[i, ...])
        predicted = np.argmax(predicted_lbls[i, ...])
        eval_log += f"{sample}, {predicted}, {lbl}\n"

    if not os.path.isdir("eval"):
        os.makedirs("eval")

    with open(os.path.join("eval", f"eval-{data_type}.csv"), "w") as eval_file:
        eval_file.write(eval_log)


if __name__ == "__main__":
    evaluate()
