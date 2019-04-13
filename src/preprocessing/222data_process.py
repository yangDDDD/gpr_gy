import sys
sys.path.append("../..")

import os
import pickle as pkl
import pandas as pd
import argparse
from src.dataset.gpr_dataset import GAPDataset, collate_examples
from torch.utils.data import DataLoader
from tqdm import tqdm

from myparser.gpr_parser import *

root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

parser = argparse.ArgumentParser(description='gpr')
parser = add_path_args(parser)
parser = add_feature_args(parser)
parser = add_learning_args(parser)
parser = add_res_args(parser)

args = parser.parse_args()

read_path = root_dir + args.dataprocess111
target_path = root_dir + args.dataprocess222

with open("../../tokenizer/tokenizer.pkl", "rb") as fr:
    tokenizer = pkl.load(fr)


#name_df = pd.read_csv(root_dir + args.nameextract333 + "/train_names.csv", sep="\t")

test_df = pd.read_pickle("%s/test.pkl"%(read_path))

test_ds = GAPDataset(test_df, tokenizer)

test_loader = DataLoader(
    test_ds,
    collate_fn = collate_examples,
    batch_size=args.test_batch_size,
    num_workers=args.num_workers,
    pin_memory=True,
    shuffle=False
)

# 清空原来目录并建文件夹
import shutil
import os
if os.path.exists(target_path):
    shutil.rmtree(target_path)
os.mkdir(target_path)
for fold_ in range(args.folds_num):
    os.mkdir(target_path + "/fold%s"%fold_)

with open(target_path + "/test_loader.pkl", "wb") as fw:
    pkl.dump(test_loader, fw)

print("test done")

for fold_ in tqdm(range(args.folds_num)):
    print(fold_)
    train_df = pd.read_pickle(read_path + "/fold%s" % fold_ + "/train.pkl")
    valid_df = pd.read_pickle(read_path + "/fold%s" % fold_ + "/valid.pkl")

    train_ds = GAPDataset(train_df, tokenizer, aug=True)
    val_ds = GAPDataset(valid_df, tokenizer)

    train_loader = DataLoader(
        train_ds,
        collate_fn=collate_examples,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        val_ds,
        collate_fn=collate_examples,
        batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    with open("%s/fold%s/train_loader.pkl" % (target_path, fold_), "wb") as fw:
        pkl.dump(train_loader, fw)

    with open("%s/fold%s/valid_loader.pkl" % (target_path, fold_), "wb") as fw:
        pkl.dump(valid_loader, fw)




