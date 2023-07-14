import json
import os
import argparse
from .engine.run_randomforest import model_train


if __name__=="__main__":

    ## Parse Arguments ##
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--graph_dimention', required=True, help="graph_node_dimention")
    args = p.parse_args()

    graph_dimention = args.graph_dimention
    model_train(graph_dimention)

