import logging
import numpy as np
import os
from zipfile import ZipFile
import scipy.io

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s| %(message)s')

script_dir = os.path.dirname(os.path.realpath(__file__))

data_path = os.path.join(script_dir, "3DShapeNets")
if not os.path.exists(data_path):
    zip_file = os.path.join(script_dir, "3DShapeNetsCode.zip")
    if not os.path.isfile(zip_file):
        print("Please download \"3DShapeNetsCode.zip\" and place it in the project root")
        exit(-1)

    with ZipFile(zip_file, "r") as zip_data:
        zip_data.extractall(script_dir)

volumetric_path = os.path.join(data_path, "volumetric_data")

logging.info('Loading .mat files')
mat_files = []

for r, _, f in os.walk(volumetric_path):
    for file in f:
        if file.endswith(".mat") and not file.endswith("feature.mat"):
            mat_files.append(os.path.join(r, file))

data_dir = os.path.join(script_dir, "data")
for file in mat_files:
    mat_file = scipy.io.loadmat(file)
    voxel = mat_file["instance"]
    lbl = file.split(os.path.sep)[-4]
    set_type = file.split(os.path.sep)[-2]

    npz_path = os.path.join(data_dir, set_type, lbl)
    if not os.path.isdir(npz_path):
        os.makedirs(npz_path)

    npz_name = os.path.basename(file).replace(".mat", "")
    np.save(os.path.join(npz_path, npz_name), voxel)
