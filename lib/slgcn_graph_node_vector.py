import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 

from .utils.models_container import ModelsContainer
from .utils.utils import load_graph_from_tsv

import joblib

import glob
import statistics
from .models.model_graph import DistMultPred, GraphNet


    
def main(parameters):
    if torch.cuda.is_available() and parameters["CUDA"]:
        device = torch.device("cuda")
        num_workers = parameters["num_workers"]
    else:
        device = torch.device("cpu")
        num_workers = 0
    FOLDER = parameters["FOLDER"]

    SAVING_ROOT_PATH = "{}".format(
        parameters["SAVING_ROOT_PATH"]
    )    
    
    graph_filename = "data/ready/{}/graph.tsv".format(FOLDER)
    print("[LOAD]", graph_filename)
    adjs, nodes = load_graph_from_tsv(graph_filename)
    #node_num = len(nodes) + 1
    node_num = len(nodes)
    adjs = [torch.tensor(adj).to(device) for adj in adjs]
    adj_num = len(adjs)
    print("#nodes:", node_num)
    print("#types of edges:", adj_num)
    


    graph_model_args = {
        "node_num": node_num,
        "adj_num": adj_num,
        "num_node_feature": parameters["NUM_NODE_FEATURE"],
    }


    models_container = ModelsContainer(
        SAVING_ROOT_PATH,
        parameters["MODEL_KEYWORD_NAME"],
        parameters["RANDOM_STATE"],
        graph_model_args,
        device,
        parameters["SAVING_ROOT_PATH"]
    )

    node_path = "{}/graph_node_vector.size_{}".format(
                parameters["SAVING_ROOT_PATH"], parameters["NUM_NODE_FEATURE"]
            )
    print(node_path)
    graph_model = models_container[0]
    out = graph_model(adjs)
    torch.save(out, node_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to the JSON config file", default="config.json"
    )
    args = parser.parse_args()
    parameters={
        "NUM_NODE_FEATURE": 32,
        "SAVING_ROOT_PATH": "model",
        "MODEL_KEYWORD_NAME": "default_model",
        "RANDOM_STATE": 100,
        "LOSS_CLIP": 0.01,
        "num_workers": 16,
        "LOSS": "Shr",
        "REDUCTION": "sum",
        "CUDA": False,
        "ENABLE_PROGRESSBAR":False,
    }
    with open(args.config) as json_file:
        parameters.update(json.load(json_file))
    main(parameters)
