import argparse
import csv
import json
import os
import re

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


from .utils.prepare_variant_data import PrepareVariantData
from .utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", help="Path to the JSON config file", default="config.json"
)
args = parser.parse_args()

with open(args.config) as json_file:
    parameters = json.load(json_file)

data_root = "data"


# Folder to save the data (it will be created if it does not exist)
FOLDER = parameters["FOLDER"]

os.makedirs("{}/ready/{}".format(data_root, FOLDER), exist_ok=True)


VARIANT_FILE = "{}/{}".format(data_root, parameters["VARIANT_FILE"])

GENESYMBOL_TO_HGNC_DATA_FILE = "{}/{}".format(
    data_root, parameters["GENESYMBOL_TO_HGNC_DATA_FILE"]
)

DATA_TYPE = parameters["DATA_TYPE"]

VERTICES_DIC = "{}/ready/{}/{}".format(
    data_root, FOLDER, parameters["VERTICES_DIC"]
)

# Prepare variant data
prepare_variant_data = PrepareVariantData(VARIANT_FILE)
prepare_variant_data.save_variant(
    GENESYMBOL_TO_HGNC_DATA_FILE,
    VERTICES_DIC,
    DATA_TYPE
    )

