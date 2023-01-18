#!/bin/bash
#$ -S /bin/sh
#$ -cwd
#$ -j y
#$ -V
#$ -q cpu.q

source ~/.bashrc

python -m scripts.prepare_graph
python -m scripts.prepare_variant
python -m lib.slgcn_graph_train
python -m lib.slgcn_graph_node_vector
python -m lib.slgcn_sample_train -d 32
echo "Done..."

