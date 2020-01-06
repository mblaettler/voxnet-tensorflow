import os
import numpy as np
import pandas as pd


DATA_DIR = "G:\\Projects\\MBlaettler_VM1\\DATA\\SICKDataCleanedASC_Voxelized"
META_DIR = "G:\\Projects\\MBlaettler_VM1\\DATA\\SICKMetadata"

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(MODULE_PATH, "data", "SVHD")


labels = {
    "ArticTruck": 0,
    "ArticTruckDumptor": 1,
    "ArticTruckLowLoaded": 2,
    "ArticTruckTanker": 3,
    "Bike": 4,
    "Bus": 5,
    "CamperVan": 6,
    "Car": 7,
    "CarWithTrailer": 8,
    "Truck": 9,
    "TruckCarTransporterLoaded": 10,
    "TruckDumptor": 11,
    "TruckLowLoaded": 12,
    "TruckTanker": 13,
    "TruckWithTrailer": 14,
    "Van": 15,
    "VanDelivery": 16,
    "VanPickup": 17,
    "VanPickupWithTrailer": 18,
    "VanWithTrailer": 19,
    "Phantom": 20,
    "ArticVan": 21,
    "TruckCarTransporterEmpty": 22,
    "VanDeliveryWithTrailer": 23
}


SIZE_X = 16
SIZE_Y = 20
SIZE_Z = 88


def num_classes(data_type):
    if data_type == "TIC":
        return 21
    else:  # assume FPS
        return 24


def __get_data(data_info):
    metadata_cache = {}
    data = []
    lbl = []

    with open(data_info, "r") as data_info_file:
        files = [line.strip() for line in data_info_file]

    for f in files:
        filepath = os.path.join(MODULE_PATH, f)
        with open(filepath, "r") as data_file:
            data_files = [line.strip() for line in data_file]

        for df in data_files:
            campaign_name, vehicle_name = df.split("/")

            df_path = os.path.join(DATA_DIR, df).replace(".vehicle", ".npy")
            df_data = np.load(df_path)
            df_data = df_data.reshape((SIZE_X, SIZE_Y, SIZE_Z, 1))

            metadata_filename = os.path.join(META_DIR, f"{campaign_name}.csv")

            if campaign_name not in metadata_cache:
                metadata_cache[campaign_name] = pd.read_csv(metadata_filename,
                                                            sep=",", header=None, index_col="fileName",
                                                            names=["fileName", "label", "labelId", "shape",
                                                                   "shapeAttributes(1)", "shapeAttributes(2)"])

            class_lbl = labels[metadata_cache[campaign_name].loc[vehicle_name]["label"]]

            lbl.append(class_lbl)
            data.append(df_data)

    return data, lbl


def get_training_data(data_type):
    train_path = os.path.join(DATA_PATH, data_type, "train_files.txt")
    return __get_data(train_path)


def get_test_data(data_type):
    test_path = os.path.join(DATA_PATH, data_type, "test_files.txt")
    return __get_data(test_path)
