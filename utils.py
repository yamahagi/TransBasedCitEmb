import os
import json
import re
import pandas as pd
import settings
import collections
from collections import defaultdict


def build_label_vocab(data_dir, task_type='re'):
    label_vocab = {}
    with open(os.path.join(data_dir, 'train.json'), 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    if task_type == 're':
        for ins in data:
            label = ins['label']
            if label not in label_vocab:
                label_vocab[label] = len(label_vocab)
    elif task_type == 'typing':
        for ins in data:
            labels = ins['labels']
            for label in labels:
                if label not in label_vocab:
                    label_vocab[label] = len(label_vocab)
    else:
        raise RuntimeError('wrong task_type')
    print('# of labels: {}'.format(len(label_vocab)))
    return label_vocab


def build_temp_ent_vocab(path):
    ent_vocab = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
    files = ['train.json', 'dev.json', 'test.json']
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        for ins in data:
            for node in ins['nodes']:
                if isinstance(node, str) and node.startswith('Q'):
                    if node not in ent_vocab:
                        ent_vocab[node] = len(ent_vocab)
    print('# of entities occurred in train/dev/test files: {}'.format(len(ent_vocab)))
    return ent_vocab

def build_temp_cite_vocab(path):
    ent_vocab = {'<unk>': 0, '<pad>': 1, '<mask>': 2}
    file = "citationsunion.txt"
    cur = 3
    for line in file:
        l = re.sub("\n","",line).split("\t")
        if l[0] not in ent_vocab:
            ent_vocab[l[0]] = cur
            cur += 1
        if l[1] not in ent_vocab:
            ent_vocab[l[1]] = cur
            cur += 1
    return ent_vocab

def build_ent_vocab(path,dataset="AASC"):
    ent_vocab = {"UNKNOWN":0,"MASK":1}
    if dataset == "AASC":
        df = pd.read_csv(path,quotechar="'")
    else:
        df = pd.read_csv(path)
    entitylist = list(set(list(df["source_id"].values)+list(df["target_id"].values)))
    entitylist.sort()
    for i,entity in enumerate(entitylist):
        ent_vocab[entity] = i+2
    return ent_vocab

#train dataで何回出てきたかによって正解率はどれくらい変わるのかを検証
def count_times(args,ent_vocab):
    #train dataからsource idごとに何回出てきたかを覚える
    source_times_dict = defaultdict(int)
    if args.dataset == "AASC":
        df = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
    else:
        df = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"train.csv"))
    true_labels_train = list(df["source_id"])
    true_labels_counter = collections.Counter(true_labels_train)
    for paper_id in true_labels_counter:
        source_times_dict[ent_vocab[paper_id]] += true_labels_counter[paper_id]
    return source_times_dict
