import sys
sys.path.append("../../")

import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse
from tqdm import tqdm
from myparser.gpr_parser import *
from src.utils.helper import distance_features

root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

parser = argparse.ArgumentParser(description='gpr')
parser = add_path_args(parser)
parser = add_feature_args(parser)

args = parser.parse_args()

target_path = root_dir + args.dataprocess111


# 将target 变成 lb_code
def extract_target(df):
    df["Neither"] = 0
    df.loc[~(df['A-coref'] | df['B-coref']), "Neither"] = 1
    df["target"] = 0
    df.loc[df['B-coref'] == 1, "target"] = 1
    df.loc[df["Neither"] == 1, "target"] = 2
    print(df.target.value_counts())
    return df


def extract_title(df):

    df["title"] = df["URL"].apply(lambda x: x.split("/")[-1])
    return df


def extract_dist_features(df):
    # index = df.index
    # columns = ["D_PA", "D_PB", "IN_URL"]
    # dist_df = pd.DataFrame(index=index, columns=columns)

    for i in tqdm(range(len(df))):
        text = df.loc[i, 'Text']
        P_offset = df.loc[i, 'Pronoun-offset']
        A_offset = df.loc[i, 'A-offset']
        B_offset = df.loc[i, 'B-offset']
        P, A, B = df.loc[i, 'Pronoun'], df.loc[i, 'A'], df.loc[i, 'B']
        URL = df.loc[i, 'URL']

        D_PA, D_PB, A_IN_URL, B_IN_URL = distance_features(P, A, B, P_offset, A_offset, B_offset, text, URL)
        df.loc[i, "D_PA"] = D_PA
        df.loc[i, "D_PB"] = D_PB
        df.loc[i, "A_IN_URL"] = A_IN_URL
        df.loc[i, "B_IN_URL"] = B_IN_URL

    return df


df_train = pd.concat([
    pd.read_csv("../../input/gap-test.tsv", delimiter="\t"),
    pd.read_csv("../../input/gap-validation.tsv", delimiter="\t")
], axis=0).reset_index(drop=True)

df_test = pd.read_csv("../../input/gap-development.tsv", delimiter="\t")

df_train = extract_title(df_train)
df_test = extract_title(df_test)

df_train = extract_target(df_train)
df_test = extract_target(df_test)

df_train = extract_dist_features(df_train)
df_test = extract_dist_features(df_test)

sample_sub = pd.read_csv("../../input/sample_submission_stage_1.csv")
assert sample_sub.shape[0] == df_test.shape[0]

skf = StratifiedKFold(n_splits=args.folds_num, random_state=args.seed, shuffle=True)

# 清空原来目录并建文件夹
import shutil
import os
if os.path.exists(target_path):
    shutil.rmtree(target_path)
os.mkdir(target_path)
for fold_ in range(args.folds_num):
    os.mkdir(target_path + "/fold%s"%fold_)


for fold_, (train_index, valid_index) in enumerate(skf.split(df_train, df_train["target"])):
    train_df = df_train.iloc[train_index]
    valid_df = df_train.iloc[valid_index]

    train_df.to_pickle("%s/fold%s/train.pkl" %(target_path, fold_))
    train_df.to_csv("%s/fold%s/train.csv" %(target_path, fold_), index=None, sep="\t")
    valid_df.to_pickle("%s/fold%s/valid.pkl" %(target_path, fold_))
    valid_df.to_csv("%s/fold%s/valid.csv" %(target_path, fold_), index=None, sep="\t")

df_train.to_pickle(target_path + "/train.pkl")
df_train.to_csv(target_path + "/train.csv", index=None, sep="\t")
df_test.to_pickle(target_path + "/test.pkl")
df_test.to_csv(target_path + "/test.csv", index=None, sep="\t")


