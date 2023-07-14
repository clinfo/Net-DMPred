#!/bin/bash

source ~/.bashrc
python -m scripts.prepare_graph
python -m scripts.prepare_variant
python -m lib.graph_train
python -m lib.graph_node_vector
python -m lib.driver_train -d 32
