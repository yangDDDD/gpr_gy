import torch.nn as nn
import spacy
import re
nlp = spacy.load('en_core_web_lg')

# Adapted from fast.ai library
def children(m):
    return m if isinstance(m, (list, tuple)) else list(m.children())


def set_trainable_attr(m, b):
    m.trainable = b
    for p in m.parameters():
        p.requires_grad = b


def apply_leaf(m, f):
    c = children(m)
    if isinstance(m, nn.Module):
        f(m)
    if len(c) > 0:
        for l in c:
            apply_leaf(l, f)


def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m, b))


def bs(lens, target):
    low, high = 0, len(lens) - 1

    while low < high:
        mid = low + int((high - low) / 2)

        if target > lens[mid]:
            low = mid + 1
        elif target < lens[mid]:
            high = mid
        else:
            return mid + 1

    return low


def bin_distance(dist):
    buckets = [-8, -4, -2, -1, 1, 2, 3, 4, 5, 8, 16, 32, 64]
    low, high = 0, len(buckets)
    while low < high:
        mid = low + int((high - low) / 2)
        if dist > buckets[mid]:
            low = mid + 1
        elif dist < buckets[mid]:
            high = mid
        else:
            return mid

    return low


def distance_features(P, A, B, char_offsetP, char_offsetA, char_offsetB, text, URL):
    doc = nlp(text)

    lens = [token.idx for token in doc]
    mention_offsetP = bs(lens, char_offsetP) - 1
    mention_offsetA = bs(lens, char_offsetA) - 1
    mention_offsetB = bs(lens, char_offsetB) - 1

    mention_distA = mention_offsetP - mention_offsetA
    mention_distB = mention_offsetP - mention_offsetB

    splited_A = A.split()[0].replace("*", "")
    splited_B = B.split()[0].replace("*", "")
    contains_a = 0
    contains_b = 0

    if re.search(splited_A[0], str(URL)):
        contains_a = 1
    elif re.search(splited_B[0], str(URL)):
        contains_b = 1
    else:
        pass

    dist_binA = bin_distance(mention_distA)
    dist_binB = bin_distance(mention_distB)
    output = dist_binA, dist_binB, contains_a, contains_b

    return output
