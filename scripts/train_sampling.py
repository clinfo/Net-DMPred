import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from collections import Counter


def variant(df):
    df["variant"] = df["HGNC_ID"] + "_" + df["HGVSp_Short"]
    return df

#random_state = 1,2,3
random_state = 1

# driver:passenger = 1:4
weight = 4

df_train_target = pd.read_csv("train_full.tsv",sep="\t")
df_train_target_driver = df_train_target[df_train_target["y"] == 1]
df_train_target_passenger = df_train_target[df_train_target["y"] == 0]
df_train_target_driver_count = pd.DataFrame(df_train_target_driver["HGNC_ID"].value_counts()).rename(columns={"HGNC_ID":"driver_count"})


# driver and passenger mutations were sampled randomly for each gene up to the median of the driver mutations per gene 
driver_med = df_train_target_driver_count["driver_count"].median()
print(driver_med)
# driver_med : 21

df_train_target_driver_count[df_train_target_driver_count > driver_med] = driver_med

df_target_driver_sample_all = pd.DataFrame()
for HGNC_ID in df_train_target_driver_count.index:
    num = df_train_target_driver_count.loc[HGNC_ID,"driver_count"]
    df_target_driver_sample = df_train_target_driver[df_train_target_driver["HGNC_ID"] == HGNC_ID].sample(n=num,random_state=random_state)
    df_target_driver_sample_all = pd.concat([df_target_driver_sample_all,df_target_driver_sample])

df_train_target_passenger_count = pd.DataFrame(df_train_target_passenger["HGNC_ID"].value_counts()).rename(columns={"HGNC_ID":"passenger_count"})
df_train_target_passenger_count[df_train_target_passenger_count > driver_med] = driver_med
df_target_passenger_sample_all = pd.DataFrame()
for HGNC_ID in df_train_target_passenger_count.index:
    num = df_train_target_passenger_count.loc[HGNC_ID,"passenger_count"]
    df_target_passenger_sample = df_train_target_passenger[df_train_target_passenger["HGNC_ID"] == HGNC_ID].sample(n=num,random_state=random_state)
    df_target_passenger_sample_all = pd.concat([df_target_passenger_sample_all,df_target_passenger_sample])


df_driver = df_target_driver_sample_all.copy()
df_passenger = df_target_passenger_sample_all.copy()

driver_gene = list(df_driver["HGNC_ID"].unique())

#passenger mutations occurred in genes with driver mutations
df_passenger_in_driver_gene = df_passenger.query("HGNC_ID in @driver_gene")

#passenger mutations of genes that did not have driver mutations
df_passenger_in_passenger_gene = df_passenger.query("HGNC_ID not in @driver_gene")

driver_variant = df_driver.shape[0]

#passenger mutations were randomly sampled in order that four times as many as driver mutations
passenger_variant = weight * driver_variant
passenger_variant_in_driver_gene = df_passenger_in_driver_gene.shape[0]
passenger_variant_in_passenger_gene = passenger_variant - passenger_variant_in_driver_gene
df_passenger_in_passenger_gene_sampling = df_passenger_in_passenger_gene.sample(n = passenger_variant_in_passenger_gene, random_state=random_state)

df_driver_use = df_driver.copy()
df_passenger_use = pd.concat([df_passenger_in_driver_gene,df_passenger_in_passenger_gene_sampling])
df_train = pd.concat([df_driver_use,df_passenger_use]).drop_duplicates()
df_train.to_csv("train.tsv", sep="\t", index=None)
