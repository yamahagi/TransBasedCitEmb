import os
import torch
import json
from torch.utils.data import Dataset
from transformers import RobertaTokenizer,BertTokenizer,BertModel
import re
import pandas as pd
import csv
from utils import build_label_vocab, build_temp_ent_vocab,build_ent_vocab
import numpy as np
import random
import settings

#frozen scibert context embeddingsを作成
def make_context_embeddings(df,path_emb):
    if os.path.exists(path_emb):
        contexts = np.load(path_emb)
        return contexts
    contexts = []
    model = BertModel.from_pretrained(settings.pretrained_scibert_path)
    tokenizer = BertTokenizer.from_pretrained(settings.pretrained_scibert_path)
    with torch.no_grad():
        for i,(left_citated_text,right_citated_text) in enumerate(zip(df["left_citated_text"],df["right_citated_text"])):
            left_citation_tokenized = tokenizer.tokenize(left_citated_text)
            right_citation_tokenized = tokenizer.tokenize(right_citated_text)
            input_tokens = tokenizer.convert_tokens_to_ids(left_citation_tokenized)+[tokenzier.sep_token_id]+tokenizer.convert_tokens_to_ids(right_citation_tokenized)
            position_citation_mark = len(left_citation_tokenized)
            tokens_tensor = torch.tensor([input_tokens])
            outputs = model(tokens_tensor)
            emb = np.array(outputs[0][position_citation_mark].cpu())
            contexts.append(emb)
        np.save(path_emb[:-4],contexts)
    return contexts

#scibert embeddingsを取り出すためのmatrixを作る
#dict1: key: citing id value: second dict
#second dict: key: cited id: value: the context embeddings from citing id to cited id
def makecitationmatrix_PeerRead(path,path_emb,ent_vocab):
    dict_scibert = {}
    df = pd.read_csv(path)
    textemb = make_context_embeddings(df,path_emb)
    target_ids = df["target_id"]
    source_ids = df["source_id"]
    for i,(target_id,source_id) in enumerate(zip(target_ids,source_ids)):
        emb = textemb[i]
        if ent_vocab[target_id] not in dict_scibert:
            dict_scibert[ent_vocab[target_id]] = {ent_vocab[source_id]:emb}
        else:
            dict_scibert[ent_vocab[target_id]][ent_vocab[source_id]] = emb
    return dict_scibert

#scibert embeddingsを取り出すためのmatrixを作る
#dict1: key: citing id value: second dict
#second dict: key: cited id: value: the context embeddings from citing id to cited id
def makecitationmatrix_AASC(path,path_emb,ent_vocab):
    dict_scibert = {}
    df = pd.read_csv(path,quotechar="'")
    textemb = make_context_embeddings(df,path_emb)
    target_ids = df["target_id"]
    source_ids = df["source_id"]
    for i,(target_id,source_id) in enumerate(zip(target_ids,source_ids)):
        emb = textemb[i]
        if ent_vocab[target_id] not in dict_scibert:
            dict_scibert[ent_vocab[target_id]] = {ent_vocab[source_id]:emb}
        else:
            dict_scibert[ent_vocab[target_id]][ent_vocab[source_id]] = emb
    return dict_scibert

class PeerReadDataSet(Dataset):
    def __init__(self, path, ent_vocab, MAX_LEN, matrix,mode="train"):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.matrix = matrix
        self.MAX_LEN = MAX_LEN
        self.data = []
        df = pd.read_csv(path)
        target_ids = df["target_id"]
        source_ids = df["source_id"]
        for i,(target_id,source_id) in enumerate(zip(target_ids,source_ids)):
            if mode == "train":
                self.data.append({
                    'target_id':ent_vocab[target_id],
                    'source_id':ent_vocab[source_id],
                    'MASK_position':0
                })
            self.data.append({
                'target_id':ent_vocab[target_id],
                'source_id':ent_vocab[source_id],
                'MASK_position':2
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class AASCDataSet(Dataset):
    def __init__(self, path, ent_vocab, MAX_LEN, matrix,mode="train"):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.matrix = matrix
        self.MAX_LEN = MAX_LEN
        self.data = []
        df = pd.read_csv(path,quotechar="'")
        target_ids = df["target_id"]
        source_ids = df["source_id"]
        for i,(target_id,source_id) in enumerate(zip(target_ids,source_ids)):
            if mode == "train":
                self.data.append({
                    'target_id':ent_vocab[target_id],
                    'source_id':ent_vocab[source_id],
                    'MASK_position':0
                })
            self.data.append({
                'target_id':ent_vocab[target_id],
                'source_id':ent_vocab[source_id],
                'MASK_position':2
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class Collate_fn():
    def __init__(self,MAX_LEN):
        self.max_words = MAX_LEN
    def collate_fn(self, batch):
        input_keys = ['target_ids','source_ids',"position_ids","contexts","token_type_ids","attention_mask","mask_positions"]
        target_keys = ["target_ids","source_ids"]
        max_words = self.MAX_LEN
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}

        for sample in batch:
            batch_x["position_ids"].append([0,1,2])
            batch_x["token_type_ids"].append([1,0,1])
            batch_x["target_ids"].append(sample["target_id"])
            batch_x["source_ids"].append(sample["source_id"])
            batch_x["contexts"].append(self.matrix[sample["target_id"]][sample["source_id"]])
            batch_x["mask_positions"].append(sample["MASK_position"])
            adj = torch.ones(3,3,dtype=torch.int)
            batch_x["attention_mask"].append(adj)
            batch_y["target_ids"].append(sample["target_id"])
            batch_y["source_ids"].append(sample["source_id"])
        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)


#入力: directory
def load_PeerRead_graph_data(path,frequency,MAX_LEN):
    def extract_by_frequency(path_train, path_test,frequency):
        dftrain = pd.read_csv(path_train)
        dftest = pd.read_csv(path_test)
        source_cut_train = dftrain[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        source_cut_test = dftest[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        ftrain_fre = open(path_train[:-4]+"_frequency"+str(frequency)+".csv","w")
        ftest_fre = open(path_test[:-4]+"_frequency"+str(frequency)+".csv","w")
        wtrain = csv.writer(ftrain_fre)
        wtest = csv.writer(ftest_fre)
        wtrain.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        wtest.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        source_train_keys = source_cut_train.source_id.value_counts().keys()
        source_test_keys = source_cut_test.source_id.value_counts().keys()
        dic1 = {}
        train_counts = source_cut_train.source_id.value_counts()
        test_counts = source_cut_test.source_id.value_counts()
        for key in source_train_keys:
            dic1[key] = train_counts[key]
        for key in source_test_keys:
            if key in dic1:
                dic1[key] += test_counts[key]
            else:
                dic1[key] = test_counts[key]
        frequencylist = []
        for key in dic1:
            if dic1[key] >= frequency:
                frequencylist.append(key)
        dftrain = dftrain.loc[dftrain["source_id"].isin(frequencylist)]
        dftest = dftest.loc[dftest["source_id"].isin(frequencylist)]
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"],dftrain["source_id"]):
            wtrain.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftrain_fre.close()
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftest["target_id"],dftest["left_citated_text"],dftest["right_citated_text"],dftest["source_id"]):
            wtest.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftest_fre.close()
        entitylist = list(set(list(dftrain["source_id"].values) + list(dftrain["target_id"].values) + list(dftest["source_id"].values) + list(dftest["target_id"].values)))
        entitylist.sort()
        entvocab = {"UNKNOWN":0,"MASK":1}
        for i,entity in enumerate(entitylist):
            entvocab[entity] = i+2
        return path_train[:-4]+"_frequency"+str(frequency)+".csv",path_test[:-4]+"_frequency"+str(frequency)+".csv",entvocab
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    path_emb_train = os.path.join(path,"scibert_PeerReadtrain.npy")
    path_emb_test = os.path.join(path,"scibert_PeerReadtest.npy")
    entvocab = build_ent_vocab(path_train,dataset="PeerRead")
    matrix_train = makecitationmatrix_PeerRead(path_train,path_emb_train,entvocab)
    matrix_test = makecitationmatrix_PeerRead(path_test,path_emb_test,entvocab)
    path_train_frequency5,path_test_frequency5,entvocab_frequency5 = extract_by_frequency(path_train,path_test,frequency)
    dataset_train = PeerReadDataSet(path_train,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_train)
    dataset_test = PeerReadDataSet(path_test,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_test,mode="test")
    dataset_train_frequency5 = PeerReadDataSet(path_train_frequency5,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_train)
    dataset_test_frequency5 = PeerReadDataSet(path_test_frequency5,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_test,mode="test")
    return dataset_train,dataset_test_frequency5,entvocab

#入力: directory
def load_AASC_graph_data(path,frequency,MAX_LEN):
    def extract_by_frequency(path_train, path_test,frequency):
        dftrain = pd.read_csv(path_train,quotechar="'")
        dftest = pd.read_csv(path_test,quotechar="'")
        source_cut_train = dftrain[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        source_cut_test = dftest[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        ftrain_fre = open(path_train[:-4]+"_frequency"+str(frequency)+".csv","w")
        ftest_fre = open(path_test[:-4]+"_frequency"+str(frequency)+".csv","w")
        wtrain = csv.writer(ftrain_fre,quotechar="'")
        wtest = csv.writer(ftest_fre,quotechar="'")
        wtrain.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        wtest.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        source_train_keys = source_cut_train.source_id.value_counts().keys()
        source_test_keys = source_cut_test.source_id.value_counts().keys()
        dic1 = {}
        train_counts = source_cut_train.source_id.value_counts()
        test_counts = source_cut_test.source_id.value_counts()
        for key in source_train_keys:
            dic1[key] = train_counts[key]
        for key in source_test_keys:
            if key in dic1:
                dic1[key] += test_counts[key]
            else:
                dic1[key] = test_counts[key]
        frequencylist = []
        for key in dic1:
            if dic1[key] >= frequency:
                frequencylist.append(key)
        dftrain = dftrain.loc[dftrain["source_id"].isin(frequencylist)]
        dftest = dftest.loc[dftest["source_id"].isin(frequencylist)]
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"],dftrain["source_id"]):
            wtrain.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftrain_fre.close()
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftest["target_id"],dftest["left_citated_text"],dftest["right_citated_text"],dftest["source_id"]):
            wtest.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftest_fre.close()
        entitylist = list(set(list(dftrain["source_id"].values) + list(dftrain["target_id"].values) + list(dftest["source_id"].values) + list(dftest["target_id"].values)))
        entitylist.sort()
        entvocab = {"UNKNOWN":0,"MASK":1}
        for i,entity in enumerate(entitylist):
            entvocab[entity] = i+2
        return path_train[:-4]+"_frequency"+str(frequency)+".csv",path_test[:-4]+"_frequency"+str(frequency)+".csv",entvocab
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    path_emb_train = os.path.join(path,"scibert_AASCtrain.npy")
    path_emb_test = os.path.join(path,"scibert_AASCtest.npy")
    entvocab = build_ent_vocab(path_train)
    matrix_train = makecitationmatrix_AASC(path_train,path_emb_train,entvocab)
    matrix_test = makecitationmatrix_AASC(path_test,path_emb_test,entvocab)
    path_train_frequency5,path_test_frequency5,entvocab_frequency5 = extract_by_frequency(path_train,path_test,frequency)
    dataset_train = AASCDataSet(path_train,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_train)
    dataset_test = AASCDataSet(path_test,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_test,mode="test")
    dataset_train_frequency5 = AASCDataSet(path_train_frequency5,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_train)
    dataset_test_frequency5 = AASCDataSet(path_test_frequency5,ent_vocab=entvocab,MAX_LEN=MAX_LEN,matrix=matrix_test,mode="test")
    return dataset_train,dataset_test_frequency5,entvocab


if __name__ == "__main__":
    path = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/citationcontexts.txt"
    entitypath = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/entity.txt"
    fe = open(entitypath)
    ent_vocab = {"UNKNOWN":0,"MASK":1}
    for i,line in enumerate(fe):
        ent_vocab[line.rstrip("\n")] = i+2
