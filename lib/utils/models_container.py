import os

import torch
from ..models.model_graph import GraphNet

from .utils import compare_distributions, plot_gt_vs_pred
from lib.models.model_graph import GraphNet
from torch import device
from typing import Dict, Tuple, Type, Union


class ModelsContainer(object):
    def __init__(
        self,
        saving_root: str,
        model_name: str,
        random_state: int,
        graph_model_args: Dict[str, int],
        device: device,
        saving_root_path: str,
    ) -> None:
        self.results = {}
        if saving_root[-1] == "/":
            saving_root = saving_root[:-1]
        self.saving_root = saving_root
        self.model_name = model_name
        self.random_state = random_state
        self.graph_model_args = graph_model_args
        self.device = device
        self.saving_root_path = saving_root_path

    def __len__(self):
        return self.number_of_models

    def __getitem__(
        self, idx: int
    ):
        path_to_graph = "{}/graph_{}_{}_{}.pt".format(
            self.saving_root, self.model_name, self.random_state, idx
        )
        graph_model = self.create_model()
        graph_model.to(self.device)

        return graph_model


    def create_model(
        self,
    ):

        base_model_path = "{}/graph.model.size_{}".format(
                self.saving_root_path, self.graph_model_args["num_node_feature"]
            )    
    
        print(base_model_path)
        print(self.graph_model_args)
        graph_model = GraphNet(**self.graph_model_args)
        graph_model.load_state_dict(torch.load(base_model_path))
        return graph_model
