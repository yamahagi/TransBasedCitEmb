import os
import torch
import json
from torch.utils.data import Dataset
from transformers import RobertaTokenizer,BertTokenizer
import re
import pandas as pd
import csv
from utils import build_label_vocab, build_temp_ent_vocab,build_ent_vocab
import numpy as np
import random

WORD_PADDING_INDEX = 1
ENTITY_PADDING_INDEX = 1

def makecitationmatrix_PeerRead(path,path_emb,ent_vocab):
    dict1 = {}
    df = pd.read_csv(path)
    textemb = np.load(path_emb)
    target_ids = df["target_id"]
    source_ids = df["source_id"]
    for i,(target_id,source_id) in enumerate(zip(target_ids,source_ids)):
        emb = textemb[i]
        if ent_vocab[target_id] not in dict1:
            dict1[ent_vocab[target_id]] = {ent_vocab[source_id]:emb}
        else:
            dict1[ent_vocab[target_id]][ent_vocab[source_id]] = emb
    return dict1

def makecitationmatrix_AASC(path,path_emb,ent_vocab):
    dict1 = {}
    df = pd.read_csv(path,quotechar="'")
    textemb = np.load(path_emb)
    target_ids = df["target_id"]
    source_ids = df["source_id"]
    for i,(target_id,source_id) in enumerate(zip(target_ids,source_ids)):
        emb = textemb[i]
        if ent_vocab[target_id] not in dict1:
            dict1[ent_vocab[target_id]] = {ent_vocab[source_id]:emb}
        else:
            dict1[ent_vocab[target_id]][ent_vocab[source_id]] = emb
    return dict1

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

    def collate_fn(self, batch):
        input_keys = ['target_ids','source_ids',"position_ids","contexts","token_type_ids","attention_masks","mask_positions"]
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
            batch_x["attention_masks"].append(adj)
            batch_y["target_ids"].append(sample["target_id"])
            batch_y["source_ids"].append(sample["source_id"])
        for k, v in batch_x.items():
            if k == 'attention_masks':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)

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

    def collate_fn(self, batch):
        input_keys = ['target_ids','source_ids',"position_ids","token_type_ids","attention_masks","mask_positions","contexts","masked_lm_labels"]
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
            if sample["MASK_position"] == 0:
                batch_x["masked_lm_labels"].append(sample["target_id"])
            else:
                batch_x["masked_lm_labels"].append(sample["source_id"])
            adj = torch.ones(3,3,dtype=torch.int)
            batch_x["attention_masks"].append(adj)
            batch_y["target_ids"].append(sample["target_id"])
            batch_y["source_ids"].append(sample["source_id"])

        for k, v in batch_x.items():
            if k == 'attention_masks':
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

#AASCのnode classificationデータを読み込む^
def load_data_SVM(model,entvocab):
    taskn = -1
    taskdict = {}
    ftrain = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_train.txt")
    len1 = 0
    for line in ftrain:
        len1 += 1
    ftrain = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_train.txt")
    taskn = -1
    X_train = []
    y_train = []
    with torch.no_grad():
        for i,line in enumerate(ftrain):
            l = line[:-1].split("\t")
            paper = l[0]
            task = l[1]
            if task not in taskdict:
                taskn += 1
                taskdict[task] = taskn
            if i % 1000 == 0:
                print("all")
                print(len1)
                print(i)
            target_ids = torch.tensor([entvocab[paper]])
            position_ids = torch.tensor([[i for i in range(3)]])
            token_type_ids = torch.tensor([[1] + [0]*2])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(2,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(3,2,dtype=torch.int)),dim=1)
            outputs = model.get_embeddings(target_ids=target_ids.cuda(),position_ids=position_ids.cuda(),token_type_ids=token_type_ids.cuda(),attention_masks=adj.cuda())
            entity_logits = outputs["sequence_output"][0][0]
            X_train.append(np.array(entity_logits.cpu()))
            y_train.append(taskdict[task])
        ftest = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_test.txt")
        X_test = []
        y_test = []
        for line in ftest:
            l = line[:-1].split("\t")
            paper = l[0]
            task = l[1]
            target_ids = torch.tensor([entvocab[paper]])
            position_ids = torch.tensor([[i for i in range(3)]])
            token_type_ids = torch.tensor([[1] + [0]*2])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(2,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(3,2,dtype=torch.int)),dim=1)
            outputs = model.get_embeddings(target_ids=target_ids.cuda(),position_ids=position_ids.cuda(),token_type_ids=token_type_ids.cuda(),attention_masks=adj.cuda())
            entity_logits = outputs["sequence_output"][0][0]
            X_test.append(np.array(entity_logits.cpu()))
            y_test.append(taskdict[task])
    return X_train,y_train,X_test,y_test

#AASCのintent identificationデータを読み込む
def load_data_intent_identification(model,entvocab):
    intentn = -1
    intentdict = {}
    f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/id2intent.txt")
    X = []
    y = []
    with torch.no_grad():
        for i,line in enumerate(f):
            l = line[:-1].split("\t")
            if i == 0:
                continue
            target_id = l[0]
            source_id = l[1]
            intent = l[2]
            if intent not in intentdict:
                intentn += 1
                intentdict[intent] = intentn
            target_ids = torch.tensor([entvocab[target_id]])
            position_ids = torch.tensor([[i for i in range(3)]])
            token_type_ids = torch.tensor([[1] + [0]*2])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(2,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(3,2,dtype=torch.int)),dim=1)
            outputs = model.get_embeddings(target_ids=target_ids.cuda(),position_ids=position_ids.cuda(),token_type_ids=token_type_ids.cuda(),attention_masks=adj.cuda())
            target_logits = outputs["sequence_output"][0][0]
            target_ids = torch.tensor([entvocab[source_id]])
            position_ids = torch.tensor([[i for i in range(3)]])
            token_type_ids = torch.tensor([[1] + [0]*2])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(2,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(3,2,dtype=torch.int)),dim=1)
            outputs = model.get_embeddings(target_ids=target_ids.cuda(),position_ids=position_ids.cuda(),token_type_ids=token_type_ids.cuda(),attention_masks=adj.cuda())
            source_logits = outputs["sequence_output"][0][0]
            X.append(np.concatenate([np.array(target_logits.cpu()),np.array(source_logits.cpu())]))
            y.append(intentdict[intent])
    return X,y

if __name__ == "__main__":
    path = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/citationcontexts.txt"
    entitypath = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/entity.txt"
    fe = open(entitypath)
    ent_vocab = {"UNKNOWN":0,"MASK":1}
    for i,line in enumerate(fe):
        ent_vocab[line.rstrip("\n")] = i+2
