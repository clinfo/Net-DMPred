import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from numpy import float64, ndarray
from torch import Tensor
from typing import Any, Dict, Iterator, List, Tuple


def load_graph_from_tsv(filename: str) -> Tuple[List[ndarray], List[int]]:
    graphs = {}
    nodes = set()
    for line in open(filename):
        arr = line.strip().split("\t")
        eid = arr[1]  # Types of graphs
        if eid not in graphs:
            graphs[eid] = []
        graphs[eid].append([int(arr[0]), int(arr[2])])
        nodes.add(int(arr[0]))
        nodes.add(int(arr[2]))
    adjs = []  # All the edges in each graph type
    for k, edges in sorted(graphs.items()):  # k stands for the type of graph
        adj = np.transpose(np.array(list(sorted(edges))))
        adjs.append(adj)
    return adjs, list(nodes)


def load_sample_feature_from_tsv(filename):
    sample_idx = []
    feature = []
    for line in open(filename):
        arr = line.strip().split("\t")
        sid = int(arr[0])
        f = list(map(float, arr[1:]))
        sample_idx.append(sid)
        feature.append(f)
    return np.array(sample_idx), np.array(feature)


def load_sample_node_feature_from_tsv(filename):
    sample_node_idx = []
    feature = []
    for line in open(filename):
        arr = line.strip().split("\t")
        sid = int(arr[0])
        nid = int(arr[1])
        f = list(map(float, arr[2:]))
        sample_node_idx.append((sid, nid))
        feature.append(f)
    return np.array(sample_node_idx), np.array(feature)


def load_label_from_tsv(filename):
    labels = []
    for line in open(filename):
        arr = line.strip().split("\t")
        labels.append(int(arr[1]))
    return np.array(labels)


def plot_grad_flow(named_parameters: Iterator[Any], saving_root: str) -> None:
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    plt.clf()
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (
            (p.requires_grad)
            and ("bias" not in n)
            and ("bn" not in n)
            and ("se" not in n)
            and ("fpn.en._fc.weight" not in n)
        ):
            if p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
            else:
                layers.append(n)
                ave_grads.append(0)
                max_grads.append(0)
                print(n, ": nograd")
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    # plt.ylim(bottom = -0.001, top=2.5) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient"],
    )
    # plt.show()
    #plt.savefig("{}/grads.png".format(saving_root))


def plot_gt_vs_pred(gt, pred, title, path="./gt_vs_pred.png", cut_off=0.5):
    min_v = -8.5
    max_v = 11
    plt.clf()
    
    g = plt.scatter(gt, pred, s=0.5, color="darkblue")
    
    #g.axes.axvspan(min_v, cut_off, alpha=0.15, color="red", label="Real True")
    #g.axes.axhspan(min_v, cut_off, alpha=0.15, color="blue", label="Pred True")
    #g.axes.legend(loc="upper left")
    #g.axes.plot([0, 1], [0, 1], transform=g.axes.transAxes, ls="--", c=".3")
    #g.axes.set_xlabel("True Values ")
    #g.axes.set_ylabel("Predictions ")
    #g.axes.set_title(title)

    #g.axes.set_xlim(min_v, max_v)
    #g.axes.set_ylim(min_v, max_v)
    #g.figure.savefig(path)


class Converter(object):
    def __init__(
        self, reference_scale: Dict[str, float64], cut_off: int = 0.5
    ) -> None:
        self.reference_scale = reference_scale
        self.cut_off = cut_off

    def sc_to_IC50(self, values):
        rescaled_lnIC50 = (
            values * self.reference_scale["std"] + self.reference_scale["mean"]
        )
        iC50 = torch.exp(rescaled_lnIC50)

        return iC50

    def sc_to_lnIC50(self, values: Tensor) -> Tensor:
        rescaled_lnIC50 = (
            values * self.reference_scale["std"] + self.reference_scale["mean"]
        )

        return rescaled_lnIC50

    def sc_to_deactivated(self, values):
        #rescaled_lnIC50 = (
        #    values * self.reference_scale["std"] + self.reference_scale["mean"]
        #)
        #deactivated = rescaled_lnIC50.lt(self.cut_off)
        #return deactivated
        
        #values = values.lt(self.cut_off)
        values = values.gt(self.cut_off)
        return values

class LogCosh(object):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, pred, true):
        loss = torch.log(torch.cosh(pred - true))

        if self.reduction == "mean":
            op = torch.mean
        elif self.reduction == "sum":
            op = torch.sum
        else:
            return loss
        return op(loss)


def compare_distributions(gt, pred, title, path="./distributions.png"):
    plt.clf()
    fig, ax = plt.subplots()
    ax.set_title(title)
    sns.histplot(gt, ax=ax, stat="density", color="red", label="GT")
    sns.histplot(pred, ax=ax, stat="density", color="skyblue", label="Pred")
    plt.legend()
    #plt.savefig(path)
