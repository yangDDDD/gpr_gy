import sys
sys.path.append("..")

import argparse
import json
import pickle as pkl
import gc
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from src.utils.helper import *
from src.utils.cycliclr import CyclicLR
from src.utils.tools import set_seed
from src.utils.bot import *

from myparser.gpr_parser import *
from src.model import GAPModel
from src.utils.logger import Logger
from src.gapbot import GAPBot
import logging


root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

parser = argparse.ArgumentParser(description='gpr')
parser = add_path_args(parser)
parser = add_feature_args(parser)
parser = add_learning_args(parser)
parser = add_res_args(parser)
parser.add_argument("--version", default="0406_v1", help="version")

#parser = parser.add_argument("--version")
args = parser.parse_args()

set_seed(args.seed)

BERT_MODEL = 'bert-large-uncased'
DEVICE = "cuda:0"

import shutil
import os
shutil.rmtree("../checkpoints")
os.mkdir("../checkpoints")

# config
with open(args.allconfig_path, 'w') as fw:
    json.dump(vars(args), fw, indent=4)

df_path = root_dir + args.dataprocess111
loader_path = root_dir + args.dataprocess222

with open("%s/test_loader.pkl" % loader_path, "rb") as fr:
    test_loader = pkl.load(fr)

with open("%s/test.pkl"%df_path, "rb") as fr:
    test_df = pkl.load(fr)

test_preds, val_ys, val_losses = [], [], []

# sub = np.zeros((df_test.shape[0], 3))
df_train = pd.read_pickle(df_path+"/train.pkl")
df_train.reset_index(inplace=True, drop=True)

skf = StratifiedKFold(n_splits=args.folds_num, random_state=args.seed, shuffle=True)

# 构建log
logger = Logger("GPR", log_dir=root_dir + args.log_dir, level=logging.INFO, use_tensorboard=False, echo=True)
logger.info(args)

oof = np.zeros((df_train.shape[0], 3))

bert_cache = {}

for fold_, (train_index, valid_index) in enumerate(skf.split(df_train, df_train["target"])):

    logger.info("=" * 20)
    logger.info("Fold " + str(fold_))
    logger.info("=" * 20)

    with open("%s/fold%s/train_loader.pkl"%(loader_path, fold_), "rb") as fr:
        train_loader = pkl.load(fr)

    with open("%s/fold%s/valid_loader.pkl"%(loader_path, fold_), "rb") as fr:
        valid_loader = pkl.load(fr)

    with open("%s/fold%s/valid.pkl"%(df_path, fold_), "rb") as fr:
        valid_df = pkl.load(fr)

    print("loader done")

    model = GAPModel(BERT_MODEL, torch.device("cuda:0"), use_layer=args.use_layers,
                     linear_hidden_size=args.linear_hidden_size, token_dist_ratio=args.token_dist_ratio, bert_cache=bert_cache)

    set_trainable(model.bert, False)
    set_trainable(model.head, True)
    steps_per_epoch = len(train_loader)
    n_steps = steps_per_epoch * 100

    ######################Cycling learning rate########################
    step_size = steps_per_epoch * args.cycle_halfT
    base_lr, max_lr = args.base_lr, args.max_lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=max_lr)

    scheduler = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr,
                         step_size=step_size, mode=args.lr_cyclemode,
                         gamma=0.99994)
    ###################################################################

    bot = GAPBot(model, train_loader, valid_loader, optimizer=optimizer,
                 logger=logger, avg_window=40, checkpoint_dir=root_dir + "/checkpoints/fold_%s"%fold_)

    gc.collect()

    bot.train(
        n_steps,
        log_interval=steps_per_epoch // 2,
        snapshot_interval=steps_per_epoch, early_stopping_cnt=args.early_stopping_rounds,
        scheduler=scheduler
    )
    # # Load the best checkpoint
    bot.load_model(bot.best_performers[0][1])
    # bot.remove_checkpoints(keep=0)

    oof_temp = torch.softmax(bot.predict(valid_loader), -1).clamp(1e-4, 1 - 1e-4).cpu().numpy()
    oof[valid_index, :] = oof_temp
    val_ys.append(valid_df.target.astype("uint8").values)
    val_losses.append(log_loss(val_ys[-1], oof_temp))
    bot.logger.info("Confirm val loss: %.4f", val_losses[-1])
    test_preds.append(torch.softmax(bot.predict(test_loader), -1).clamp(1e-4, 1 - 1e-4).cpu().numpy())

    logger.info("\n" * 4)

    bert_cache = model.bert_cache

print(val_losses)

final_test_preds = np.mean(test_preds, axis=0)
print(final_test_preds.shape)

score_valid = log_loss(df_train.target, oof)

score_test = log_loss(test_df.target, final_test_preds)

logger.info("valid_all: "+ str(score_valid))
logger.info("test score: "+ str(score_test))

# Create submission file
df_oof = pd.DataFrame(oof, columns=["A", "B", "NEITHER"])
df_oof["ID"] = df_train.ID
df_oof.to_csv("../oof/gy_oof_%s_%s_%s.csv"%(args.version, score_valid, score_test), index=False)


df_sub = pd.DataFrame(final_test_preds, columns=["A", "B", "NEITHER"])
df_sub["ID"] = test_df.ID
df_sub.to_csv("../sub/gy_submission_%s_%s_%s.csv"%(args.version, score_valid, score_test), index=False)
df_sub.head()
