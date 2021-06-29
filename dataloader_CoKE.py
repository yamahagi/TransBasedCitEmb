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

def make_matrix(ent_vocab):
    path = settings.citation_recommendation_dir
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    path_emb_train = os.path.join(path,"scibert_AASCtrain.npy")
    path_emb_test = os.path.join(path,"scibert_AASCtest.npy")
    matrix_train = makecitationmatrix_AASC(path_train,path_emb_train,ent_vocab)
    matrix_test = makecitationmatrix_AASC(path_test,path_emb_test,ent_vocab)
    matrix = matrix_train
    for target_id in matrix_test:
        if target_id in matrix:
            for source_id in matrix_test[target_id]:
                matrix[target_id][source_id] = matrix_test[target_id][source_id]
        else:
            matrix[target_id] = matrix_test[target_id]
    return matrix

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


#入力: directory
def load_PeerRead_graph_data(args):
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
    path = settings.citation_recommendation_dir
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    path_emb_train = os.path.join(path,"scibert_PeerReadtrain.npy")
    path_emb_test = os.path.join(path,"scibert_PeerReadtest.npy")
    entvocab = build_ent_vocab(path_train,dataset="PeerRead")
    matrix_train = makecitationmatrix_PeerRead(path_train,path_emb_train,entvocab)
    matrix_test = makecitationmatrix_PeerRead(path_test,path_emb_test,entvocab)
    path_train_frequency5,path_test_frequency5,entvocab_frequency5 = extract_by_frequency(path_train,path_test,args.frequency)
    dataset_train = PeerReadDataSet(path_train,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_train)
    dataset_test = PeerReadDataSet(path_test,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_test,mode="test")
    dataset_train_frequency5 = PeerReadDataSet(path_train_frequency5,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_train)
    dataset_test_frequency5 = PeerReadDataSet(path_test_frequency5,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_test,mode="test")
    return dataset_train,dataset_test_frequency5,entvocab

#入力: directory
def load_AASC_graph_data(args):
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
    path = settings.citation_recommendation_dir
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    path_emb_train = os.path.join(path,"scibert_AASCtrain.npy")
    path_emb_test = os.path.join(path,"scibert_AASCtest.npy")
    entvocab = build_ent_vocab(path_train)
    matrix_train = makecitationmatrix_AASC(path_train,path_emb_train,entvocab)
    matrix_test = makecitationmatrix_AASC(path_test,path_emb_test,entvocab)
    path_train_frequency5,path_test_frequency5,entvocab_frequency5 = extract_by_frequency(path_train,path_test,args.frequency)
    if args.train_data == "full":
        dataset_train = AASCDataSet(path_train,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_train)
    else:
        dataset_train = AASCDataSet(path_train_frequency5,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_train)
    if args.test_data == "full":
        dataset_test = AASCDataSet(path_test,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_test,mode="test")
    else:
        dataset_test = AASCDataSet(path_test_frequency5,ent_vocab=entvocab,MAX_LEN=args.MAX_LEN,matrix=matrix_test,mode="test")
    #return dataset_train,dataset_test,entvocab
    return dataset_train,dataset_test,entvocab,matrix_train,matrix_test


if __name__ == "__main__":
    path = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/citationcontexts.txt"
    entitypath = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/entity.txt"
    fe = open(entitypath)
    ent_vocab = {"UNKNOWN":0,"MASK":1}
    for i,line in enumerate(fe):
        ent_vocab[line.rstrip("\n")] = i+2
