import sys
sys.path.append("../../")

import os
import argparse

import numpy as np
import pandas as pd

import torch
import random as rn

from torch.utils.data import Dataset

import spacy
nlp = spacy.load('en_core_web_lg')

from myparser.gpr_parser import *
from src.utils.tools import set_seed

root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

parser = argparse.ArgumentParser(description='gpr')
parser = add_feature_args(parser)
args = parser.parse_args()



set_seed(args.seed)

class GAPDataset(Dataset):
    """Custom GAP Dataset class"""
    def __init__(self, df, tokenizer, labeled=True, aug=False):
        self.labeled = labeled
        self.aug = aug
        if labeled:
            self.y = df.target.values.astype("uint8")

        self.offsets, self.tokens, self.dist, self.in_url = [], [], [], []
        for _, row in df.iterrows():

            tokens, offsets = row_tokenize(row, tokenizer)

            self.offsets.append(offsets)
            self.tokens.append(tokenizer.convert_tokens_to_ids(
                ["[CLS]"] + tokens))
            self.dist.append([row["D_PA"], row["D_PB"]])
            self.in_url.append([row["A_IN_URL"], row["B_IN_URL"]])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        if self.aug:
            if rn.random() < 0.5:
                A_start = self.offsets[idx][0]
                A_end = self.offsets[idx][1]
                for i in range(A_start, A_end + 1):
                    tokens[i] = 1
            if rn.random() < 0.5:
                B_start = self.offsets[idx][2]
                B_end = self.offsets[idx][3]
                for i in range(B_start, B_end + 1):
                    tokens[i] = 2
        else:
            pass

        if self.labeled:
            return tokens, self.offsets[idx], self.dist[idx], self.in_url[idx], self.y[idx]
        return self.tokens[idx], self.offsets[idx], self.dist[idx], self.in_url[idx], None


def row_tokenize(row, tokenizer):
    break_points = sorted(
        [
            ("A", row["A-offset"], row["A"]),
            ("B", row["B-offset"], row["B"]),
            ("P", row["Pronoun-offset"], row["Pronoun"]),
        ], key=lambda x: x[1]
    )

    # if aug:
    #     shift_offset = 0
    #     current_pos = 0
    #     for idx, (name, offset, text) in enumerate(break_points):
    #         if name == "A":
    #             if rn.random() < 0.5 and text.lower() in name_df["name"].values:
    #                 gender = name_df.loc[name_df["name"] == text.lower(), "gender"].values[0]
    #                 gender_df = name_df[name_df["gender"] == gender]
    #                 replace_name = gender_df.sample(n=1, random_state=args.seed)["name"].values[0]
    #                 origin_len = len(text.lower())
    #                 replace_len = len(replace_name)
    #                 token_offset = offset + shift_offset
    #                 row["Text"] = row["Text"][current_pos: token_offset] + \
    #                                                 replace_name + row["Text"][token_offset + origin_len:]
    #                 shift_offset += (replace_len - origin_len)
    #                 break_points[idx] = ("A", token_offset, replace_name)
    #
    #             else:
    #                 token_offset = offset + shift_offset
    #                 break_points[idx] = ("A", token_offset, text)
    #
    #         if name == "B":
    #             if rn.random() < 0.5 and text.lower() in name_df["name"].values:
    #                 gender = name_df.loc[name_df["name"] == text.lower(), "gender"].values[0]
    #                 gender_df = name_df[name_df["gender"] == gender]
    #                 replace_name = gender_df.sample(n=1, random_state=args.seed)["name"].values[0]
    #                 origin_len = len(text.lower())
    #                 replace_len = len(replace_name)
    #                 token_offset = offset + shift_offset
    #                 row["Text"] = row["Text"][current_pos: token_offset] + \
    #                               replace_name + row["Text"][token_offset + origin_len:]
    #                 break_points[idx] = ("B", token_offset, replace_name)
    #                 shift_offset += (replace_len - origin_len)
    #             else:
    #                 token_offset = offset + shift_offset
    #                 break_points[idx] = ("B", token_offset, text)
    #
    #         if name == "P":
    #             token_offset = offset + shift_offset
    #             break_points[idx] = ("P", token_offset, text)

    tokens, spans, current_pos = [], {}, 0
    # print(row["Text"])
    # print("#################")
    for name, offset, text in break_points:
        context = row["Text"][current_pos:offset].lower().replace("#", "*")
        doc = nlp(context)
        for sent in doc.sents:
            if sent.text[-1] in [".", "?", "!", ".''"]:
                tokens.extend(tokenizer.tokenize(sent.text + " [SEP]"))
            else:
                tokens.extend(tokenizer.tokenize(sent.text))

        # Make sure we do not get it wrong
        # assert row["Text"][offset:offset+len(text)] == text
        # Tokenize the target
        # TODO token的分化有什么问题?     P的目的token
        tmp = row["Text"][offset:offset+len(text)].lower().replace("#", "*")
        tmp_tokens = tokenizer.tokenize(tmp)
        spans[name] = [len(tokens), len(tokens) + len(tmp_tokens) - 1] # inclusive
        tokens.extend(tmp_tokens)
        current_pos = offset + len(text)

    tail_context = row["Text"][current_pos:].lower().replace("#", "*")
    doc = nlp(tail_context)
    for sent in doc.sents:
        tokens.extend(tokenizer.tokenize(sent.text + " [SEP]"))

    urls = tokenizer.tokenize(row["title"].lower())
    spans["url"] = [len(tokens), len(tokens) + len(urls) - 1]
    tokens.extend(urls)

    assert spans["P"][0] == spans["P"][1]
    return tokens, (spans["A"] + spans["B"] + [spans["P"][0]] + spans["url"])


def collate_examples(batch, truncate_len=490):
    """Batch preparation.

    1. Pad the sequences
    2. Transform the target.
    """
    transposed = list(zip(*batch))
    max_len = min(
        max((len(x) for x in transposed[0])),
        truncate_len
    )
    tokens = np.zeros((len(batch), max_len), dtype=np.int64)
    for i, row in enumerate(transposed[0]):
        row = np.array(row[:truncate_len])
        tokens[i, :len(row)] = row
    token_tensor = torch.from_numpy(tokens)
    # Offsets
    offsets = torch.stack([
        torch.LongTensor(x) for x in transposed[1]
    ], dim=0) + 1  # Account for the [CLS] token
    # dist
    dists = torch.stack([
        torch.LongTensor(x) for x in transposed[2]
    ], dim=0)
    # in_url
    in_urls = torch.stack([
        torch.FloatTensor(x) for x in transposed[3]
    ], dim=0)
    # Labels
    if len(transposed) == 4:
        return token_tensor, offsets, dists, in_urls,  None
    labels = torch.LongTensor(transposed[4])
    return token_tensor, offsets, dists, in_urls, labels
