import os
import argparse
import pickle as pkl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import torch
import pandas as pd
from tqdm import tqdm

from myparser.gpr_parser import *
from src.model import GAPModel

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

parser = argparse.ArgumentParser(description='gpr')
parser = add_path_args(parser)
parser = add_feature_args(parser)
parser = add_learning_args(parser)
parser = add_res_args(parser)

args = parser.parse_args()

BERT_MODEL = 'bert-large-uncased'


def predict_batch(model, input_tensors):
    model.eval()
    tmp = model(*input_tensors)
    return tmp

def predict(model, loader, *, return_y=False):
    model.eval()
    outputs, y_global = [], []
    with torch.set_grad_enabled(False):
        for *input_tensors, y_local in tqdm(loader):
            input_tensors = [x.to("cuda:0") for x in input_tensors]
            outputs.append(predict_batch(model, input_tensors).cpu())
            y_global.append(y_local.cpu())
        outputs = torch.cat(outputs, dim=0)
        y_global = torch.cat(y_global, dim=0)
    if return_y:
        return outputs, y_global
    return outputs

df_path = root_dir + args.dataprocess111
loader_path = root_dir + args.dataprocess222

skf = StratifiedKFold(n_splits=args.folds_num, random_state=args.seed, shuffle=True)

df_train = pd.read_pickle(df_path+"/train.pkl")

df_test = pd.read_pickle(df_path + "/test.pkl")

with open("%s/test_loader.pkl"%(loader_path), "rb") as fr:
    test_loader = pkl.load(fr)

oof = np.zeros((df_train.shape[0], 3))
sub = np.zeros((df_test.shape[0], 3))
for fold_, (train_index, valid_index) in enumerate(skf.split(df_train, df_train["target"])):
#
    print("=" * 20)
    print("Fold " + str(fold_))
    print("=" * 20)

    model_path = root_dir + "/checkpoints/fold_%s/best.pth"%fold_

    fold_model = GAPModel(BERT_MODEL, torch.device("cuda:0"))
    fold_model.load_state_dict(torch.load(model_path))

    with open("%s/fold%s/valid_loader.pkl"%(loader_path, fold_), "rb") as fr:
        valid_loader = pkl.load(fr)

    oof_temp = torch.softmax(predict(fold_model, valid_loader), -1).clamp(1e-4, 1 - 1e-4).cpu().numpy()

    test_temp = torch.softmax(predict(fold_model, test_loader), -1).clamp(1e-4, 1 - 1e-4).cpu().numpy()

    oof[valid_index, :] = oof_temp

    sub += test_temp / args.folds_num * 5
    break

# print("valid_loss: ", log_loss(df_train["target"], oof))

print("test_loss: ", log_loss(df_test["target"], sub))

