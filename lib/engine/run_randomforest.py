import os
from scipy import stats
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import statistics

from sklearn.metrics import roc_auc_score
#from sklearn.metrics import roc_curve
#from sklearn.metrics import precision_recall_fscore_support
#from sklearn.metrics import confusion_matrix




def load_graph(graph_dimention):
    base_model_path = "model/graph_node_vector.size_" + str(graph_dimention)
    embed = torch.load(base_model_path)
    embed = embed.detach().numpy()
    df_graph = pd.DataFrame(embed)
    df_graph = df_graph.reset_index().rename(columns={"index":"node_num"})
    embed_mean = pd.Series(df_graph.loc[:,0:].mean())
    embed_mean = embed_mean.append(pd.Series([-1],index=["node_num"]))
    df_graph = df_graph.append(embed_mean,ignore_index=True)
    return df_graph


def train_dataset(feature_dic, df_graph):
    df_train = pd.read_csv("data/ready/full_features/train/labels.tsv",header=None,sep="\t",names=["num","label"])
    df_node_feature = pd.read_csv("data/ready/full_features/train/node_features.tsv",sep="\t",header=None)
    df_node_feature = df_node_feature.rename(columns=feature_dic)
    df_node_feature = df_node_feature.rename(columns={0:"num",1:"node_num"})
    df_train = pd.merge(df_train,df_node_feature,on="num",how="left")
    df_train = pd.merge(df_train,df_graph,on="node_num",how="left")
    return df_train

def training_model(df_train):
    n_splits = 5
    random_state = 0
    shuffle = True
        
    param_grid = {"max_depth":[1,3,5,7,None],
                  "min_samples_leaf":[1,5,10],
                  "n_estimators":[100,150,300,500,700,1000],
                  "max_features":["sqrt","log2"]}

    stratification_labels = df_train[["label"]]
    kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)    
    result_list = []
    for train_index, test_index in kfold.split(sample_features, stratification_labels):
        train_features = sample_features.query("index in @train_index")
        test_features = sample_features.query("index in @test_index")
        train_label = stratification_labels.query("index in @train_index")
        test_label = stratification_labels.query("index in @test_index")

        X_train = train_features.values
        y_train = train_label["label"].values
        X_test = test_features.values
        y_test = test_label["label"].values


        forest_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=random_state),
                                  param_grid = param_grid,
                                  scoring="roc_auc",
                                  cv=3)

        forest_grid.fit(X_train, y_train)

        forest_grid_best = forest_grid.best_estimator_

        print(forest_grid.best_params_)
        best_model = forest_grid.best_estimator_
        print(best_model)
        ROC_AUC = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])
        print(ROC_AUC)

        result = [ROC_AUC,forest_grid.best_params_]
        result_list.append(result)
        print()
    return result_list

def model_train(graph_dimention):
    with open("data/raw/train/train.tsv") as f:
        firstline = f.readline().rstrip()
        firstline = firstline.split("\t")
    feature = [s for s in firstline if s not in ["Hugo_Symbol","HGVSp_Short","y","HGNC_ID","cancer_type"]]
    print(len(feature))
    feature_dic = {n+2:v for n,v in enumerate(feature)}
    df_graph = load_graph(graph_dimention)
    df_train = train_dataset(feature_dic, df_graph)    
    result_list = training_model(df_train)
    cv_auc_result = [result_list[n][0] for n in range(len(result_list))]
    print("cross_validation_mean: ", round(statistics.mean(cv_auc_result),3))
    print("cross_validation_std: ",round(statistics.pstdev(cv_auc_result),3))
