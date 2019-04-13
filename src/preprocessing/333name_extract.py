import sys
sys.path.append("../../")

import os
import pandas as pd
import argparse
from myparser.gpr_parser import *

root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

parser = argparse.ArgumentParser(description='gpr')
parser = add_path_args(parser)
parser = add_feature_args(parser)

args = parser.parse_args()

target_path = root_dir + args.nameextract333

if not os.path.exists(target_path):
    os.makedirs(target_path)

gender_dict = {'his': 1, 'him': 1, 'he': 1, 'her': 0, 'she': 0, 'hers': 0}

df_train = pd.concat([
    pd.read_csv("../../input/gap-test.tsv", delimiter="\t"),
    pd.read_csv("../../input/gap-validation.tsv", delimiter="\t")
], axis=0).reset_index(drop=True)

df_test = pd.read_csv("../../input/gap-development.tsv", delimiter="\t")

df_train = df_train[["ID", "Pronoun", "A", "A-coref", "B", "B-coref"]]

name_gendar_list = []

for idx in range(len(df_train)):
    row = df_train.loc[idx]

    if row["A-coref"]:
        name_gendar_list.append([row["A"].lower(), gender_dict[row["Pronoun"].lower()]])
    elif row["B-coref"]:
        name_gendar_list.append([row["B"].lower(), gender_dict[row["Pronoun"].lower()]])
    else:
        pass

gender_df = pd.DataFrame(name_gendar_list, columns=["name", "gender"])

gender_df = gender_df.drop_duplicates(subset=["name", "gender"])

gender_df.to_csv(target_path + "/train_names.csv", index=None, sep="\t")
