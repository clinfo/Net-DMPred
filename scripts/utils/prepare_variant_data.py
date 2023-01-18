import os
import csv
import math
import re
from typing import Dict, Tuple, List

import pandas as pd
from pandas.core.frame import DataFrame

def convert_to_hgnc_ids(variant: DataFrame, hgnc: DataFrame) -> DataFrame:
    dic_hgnc = hgnc.reset_index(drop=True).to_dict("records")
    dic_hgnc = {element["Symbol"]: element["HGNC_ID"] for element in dic_hgnc}
    variant["HGNC_ID"] = variant["Hugo_Symbol"].map(dic_hgnc)
    variant.loc[variant["HGNC_ID"].isnull(), "HGNC_ID"] = variant["Hugo_Symbol"]
    return variant

def convert_to_vertices_num(variant: DataFrame, vertices_num: DataFrame) -> DataFrame:
    print(vertices_num.head())
    dic_vertices_num = vertices_num.reset_index(drop=True).to_dict("records")
    dic_vertices_num = {element["HGNC_ID"]: element["num"] for element in dic_vertices_num}
    variant["vertices_num"] = variant["HGNC_ID"].map(dic_vertices_num)
    variant["vertices_num"] = variant["vertices_num"].fillna(-1)
    return variant

def prepare_variant_name(variant: DataFrame) -> DataFrame:
    variant["HGNC_HGVSp"] = variant["HGNC_ID"] + "_" + variant["HGVSp_Short"]
#    variant["aaref"] = variant["HGVSp_Short"].str[2]
#    variant["aaalt"] = variant["HGVSp_Short"].str[-1]
#    variant["aapos"] = variant["HGVSp_Short"].str[3:-1]
    return variant

def create_labels_relations_dic(variant: DataFrame) ->  Tuple[Dict[str, int], Dict[str, int]]:
    unique_variant = set(variant["HGNC_HGVSp"].unique())
    return {v: n for n, v in enumerate(unique_variant)}

def save_dics(dic: Dict[str, int], output_file: str) -> None:
    pd.DataFrame.from_dict(dic, orient="index").to_csv(
        output_file, sep="\t", header=False, quoting=csv.QUOTE_NONE
    )

def label(variant: DataFrame) -> DataFrame:
    variant["y"] = variant["y"].replace({"driver":1,"passenger":0})
    variant = variant.sort_values(by="variant_num")
    return variant[["variant_num","y"]].drop_duplicates()


def node_features(variant: DataFrame) -> DataFrame:
    feature_col = list(variant.columns)
    feature_col.remove("Hugo_Symbol")
    feature_col.remove("HGVSp_Short")
    feature_col.remove("y")
    feature_col.remove("HGNC_ID")
    feature_col.remove("cancer_type")
    feature_col.remove("vertices_num")
    feature_col.remove("variant_num")
    feature_col.remove("HGNC_HGVSp")
    print("feature_len", len(feature_col))
    base_col = ["variant_num","vertices_num"]
    use_cols = base_col + feature_col
    
    score = variant[use_cols].drop_duplicates()

    
    score_sort = score.sort_values(by="variant_num")
    score_sort["variant_num"] = score_sort["variant_num"].astype(int) 
    score_sort["vertices_num"] = score_sort["vertices_num"].astype(int)    
    return score_sort    
    

class PrepareVariantData(object):
    def __init__(self, variant_data_file: str) -> None:
        print("Loading Variant Data...")
        with open(variant_data_file) as csv_file:
            reader = csv.reader(csv_file, delimiter="\t")
            header = next(reader)          
            self.variant_data = pd.DataFrame(list(reader), columns=header)
        print(self.variant_data.head())
            
    def save_variant(self,genesymbol_to_hgnc:str, vertices_dic_location: str, data_type: str) -> None:
        print("Prepare variant data...")
        
        genesymbol_to_hgnc = pd.read_csv(genesymbol_to_hgnc, sep="\t")

        hgnc_to_verticesnum = pd.read_csv(vertices_dic_location, sep="\t", names=["HGNC_ID","num"])
        self.variant_data = convert_to_vertices_num(self.variant_data, hgnc_to_verticesnum)

        self.variant_data = prepare_variant_name(self.variant_data)

        self.variant_dic = create_labels_relations_dic(self.variant_data)
        self.variant_data["variant_num"] = self.variant_data["HGNC_HGVSp"].map(self.variant_dic)

        output_variant_location = "data/raw/" + data_type + "/prepare_variant.tsv"
        self.variant_data.to_csv(output_variant_location,sep="\t",index=False,quoting=csv.QUOTE_NONE)

        output_label_filepath = "data/ready/full_features/" + data_type + "/labels.tsv"
        label(self.variant_data).to_csv(output_label_filepath, sep="\t",index=False,header=False,quoting=csv.QUOTE_NONE)


        output_node_features_filepath = "data/ready/full_features/" + data_type + "/node_features.tsv"
        node_features(self.variant_data).to_csv(output_node_features_filepath, sep="\t",index=False,header=False,quoting=csv.QUOTE_NONE)

        output_prepare_sample_features_filepath = "data/ready/full_features/" + data_type + "/sample_features.tsv"


        variant_dic_location = "data/ready/full_features/" + data_type + "/variant_dic.tsv"
        save_dics(self.variant_dic, variant_dic_location)
