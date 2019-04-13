import os
import pickle as pkl
import argparse
import pandas as pd
from pytorch_pretrained_bert import BertTokenizer

from myparser.gpr_parser import *

root_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))

parser = argparse.ArgumentParser(description='gpr')
parser = add_path_args(parser)
parser = add_feature_args(parser)
parser = add_learning_args(parser)
parser = add_res_args(parser)

args = parser.parse_args()

BERT_MODEL = 'bert-large-uncased'
CASED = False
tokenizer = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=CASED,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[MALENAME]", "[FEMALENAME]", "[NAME]", "[A]", "[B]"),
#    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
)

tokenizer.vocab["[A]"] = 1
tokenizer.vocab["[B]"] = 2
tokenizer.vocab["[MALENAME]"] = 3
tokenizer.vocab["[FEMALENAME"] = 4
tokenizer.vocab["[NAME]"] = 5

with open(root_dir + "/tokenizer/tokenizer.pkl", "wb") as fw:
    pkl.dump(tokenizer, fw)

# read_path = root_dir + args.dataprocess111
#
# df_train = pd.read_pickle(read_path + "/train.pkl")
# df_test = pd.read_pickle(read_path + "/test.pkl")
# df = pd.concat([df_train, df_test]).reset_index(drop=True)

# for idx in range(len(df)):
#     text = df.loc[idx, "Text"]
#     text = text.lower()
#     print(text)
#     result = tokenizer.tokenize(text)
#     print(result)

# with open(root_dir + args.dataprocess222 + "/fold0/train_loader.pkl", "rb") as fr:
#     train_loader = pkl.load(fr)
#
# with open(root_dir + args.dataprocess222 + "/fold0/valid_loader.pkl", "rb") as fr:
#     valid_loader = pkl.load(fr)
#
# with open(root_dir + args.dataprocess222 + "/test_loader.pkl", "rb") as fr:
#     test_loader = pkl.load(fr)
#
# print(len(train_loader))
# for tokens, offset, target in valid_loader:
#     print(tokens.size())