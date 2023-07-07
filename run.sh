#!/bin/bash

source ~/.bashrc
python -m scripts.prepare_graph
python -m scripts.prepare_variant
python -m lib.slgcn_graph_train
python -m lib.slgcn_graph_node_vector
python -m lib.slgcn_sample_train -d 32
