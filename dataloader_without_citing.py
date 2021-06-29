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
import settings

#make masked paper prediction data
def make_MPP_data(dic_data,WINDOW_SIZE,MAX_LEN,tokenizer,ent_vocab,mask_position):
    #mask cited paper(source_id)
    target_id = dic_data["target_id"]
    source_id = dic_data["source_id"]
    left_citation_tokenized = dic_data["left_citation_tokenized"]
    right_citation_tokenized = dic_data["right_citation_tokenized"]
    citationcontextl = []
    masked_ids = []
    position_ids = []
    token_type_ids = []
    if mask_position == "cited":
        #append citing id
        citationcontextl.extend([tokenizer.cls_token_id])
        masked_ids.extend([-1])
        position_ids.extend([0])
        token_type_ids.extend([0])
        #append citation context
        citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab["MASK"]] + right_citation_tokenized[:WINDOW_SIZE])
        position_ids.extend([1+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
        masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
        token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
    else:
        #append citing id
        citationcontextl.extend([tokenizer.cls_token_id,ent_vocab["MASK"],tokenizer.sep_token_id])
        masked_ids.extend([-1,ent_vocab[target_id],-1])
        position_ids.extend([0,1,2])
        token_type_ids.extend([0,1,0])
        #append citation context
        citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE])
        position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
        masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [-1] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
        token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
    data = {'input_ids': citationcontextl[:MAX_LEN],'masked_lm_labels' : masked_ids[:MAX_LEN],'position_ids': position_ids[:MAX_LEN],'token_type_ids': token_type_ids[:MAX_LEN]}
    return data

def make_json(df,jsonpath,tokenizer):
    target_ids = df["target_id"]
    source_ids = df["source_id"]
    left_citation_texts = df["left_citated_text"]
    right_citation_texts = df["right_citated_text"]
    dic_json = []
    for i,(target_id,source_id,left_citation_text,right_citation_text) in enumerate(zip(target_ids,source_ids,left_citation_texts,right_citation_texts)):
        if i % 1000 == 0:
            print(i)
        left_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(left_citation_text))
        right_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(right_citation_text))
        dic_data = {"target_id":target_id,"source_id":source_id,"left_citation_tokenized":left_citation_tokenized,"right_citation_tokenized":right_citation_tokenized}
        dic_json.append(dic_data)
    fids = open(jsonpath,"w")
    json.dump(dic_json,fids)


class PeerReadDataSet(Dataset):
    def __init__(self, path, ent_vocab, WINDOW_SIZE, MAX_LEN, pretrained_model):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.MAX_LEN = MAX_LEN
        self.data = []
        if pretrained_model == "scibert":
            self.tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
        else:
            self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case =False)
        df = pd.read_csv(path)
        jsonpath = os.path.join(self.dirname,self.filename[:-4]+"_window"+str(WINDOW_SIZE)+"_MAXLEN"+str(MAX_LEN)+"_pretrainedmodel"+str(pretrained_model)+".json")
        if not(os.path.exists(jsonpath)):
            make_json(df,jsonpath,self.tokenizer)
        fids = open(jsonpath)
        dic_json = json.load(fids)
        for dic_data in dic_json:
            data = make_MPP_data(dic_data,WINDOW_SIZE,MAX_LEN,self.tokenizer,ent_vocab,"cited")
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class AASCDataSet(Dataset):
    def __init__(self, path, ent_vocab,WINDOW_SIZE,MAX_LEN,pretrained_model,mode="train"):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.MAX_LEN = MAX_LEN
        self.data = []
        if pretrained_model == "scibert":
            self.tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
        else:
            self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case =False)
        df = pd.read_csv(path,quotechar="'")
        jsonpath = os.path.join(self.dirname,self.filename[:-4]+"_window"+str(WINDOW_SIZE)+"_MAXLEN"+str(MAX_LEN)+"_pretrainedmodel"+str(pretrained_model)+"_TBCN.json")
        if not(os.path.exists(jsonpath)):
             make_json(df,jsonpath,self.tokenizer)
        fids = open(jsonpath)
        dic_json = json.load(fids)
        for dic_data in dic_json:
            data = make_MPP_data(dic_data,WINDOW_SIZE,MAX_LEN,self.tokenizer,ent_vocab,"cited")
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class AASCDataSetRANDOM(Dataset):
    def __init__(self, path, ent_vocab,WINDOW_SIZE,MAX_LEN,pretrained_model,mode="train"):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.MAX_LEN = MAX_LEN
        self.WINDOW_SIZE = WINDOW_SIZE
        self.ent_vocab = ent_vocab
        self.data = []
        if pretrained_model == "scibert":
            self.tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
        else:
            self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case =False)
        df = pd.read_csv(path,quotechar="'")
        jsonpath = os.path.join(self.dirname,self.filename[:-4]+"_window"+str(WINDOW_SIZE)+"_MAXLEN"+str(MAX_LEN)+"_pretrainedmodel"+str(pretrained_model)+"_TBCN.json")
        if not(os.path.exists(jsonpath)):
             make_json(df,jsonpath,self.tokenizer)
        fids = open(jsonpath)
        dic_json = json.load(fids)
        for dic_data in dic_json:
            if random.random() < 0.5 or mode == "test":
                data = make_MPP_data(dic_data,WINDOW_SIZE,MAX_LEN,self.tokenizer,ent_vocab,"cited")
            else:
                data = make_MPP_data(dic_data,WINDOW_SIZE,MAX_LEN,self.tokenizer,ent_vocab,"citing")
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class AASCDataSetBOTH(Dataset):
    def __init__(self, path, ent_vocab,WINDOW_SIZE,MAX_LEN,pretrained_model,mode="train"):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.MAX_LEN = MAX_LEN
        self.data = []
        self.ent_vocab = ent_vocab
        if pretrained_model == "scibert":
            self.tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
        else:
            self.tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case =False)
        df = pd.read_csv(path,quotechar="'")
        jsonpath = os.path.join(self.dirname,self.filename[:-4]+"_window"+str(WINDOW_SIZE)+"_MAXLEN"+str(MAX_LEN)+"_pretrainedmodel"+str(pretrained_model)+"_TBCN.json")
        if not(os.path.exists(jsonpath)):
             make_json(df,jsonpath,self.tokenizer)
        fids = open(jsonpath)
        dic_json = json.load(fids)
        for dic_data in dic_json:
            data_cited_mask = make_MPP_data(dic_data,WINDOW_SIZE,MAX_LEN,self.tokenizer,ent_vocab,"cited")
            data_citing_mask = make_MPP_data(dic_data,WINDOW_SIZE,MAX_LEN,self.tokenizer,ent_vocab,"citing")
            if mode == "train":
                self.data.extend([data_cited_mask,data_citing_mask])
            else:
                self.data.extend([data_cited_mask])

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
        ent_vocab = {"UNKNOWN":0,"MASK":1}
        for i,entity in enumerate(entitylist):
            ent_vocab[entity] = i+2
        return path_train[:-4]+"_frequency"+str(frequency)+".csv",path_test[:-4]+"_frequency"+str(frequency)+".csv",ent_vocab
    path = settings.citation_recommendation_dir
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    ent_vocab = build_ent_vocab(path_train)
    path_train_frequency5,path_test_frequency5,ent_vocab_frequency5 = extract_by_frequency(path_train,path_test,args.frequency)
    datasetdict = {"tail":PeerReadDataSet,"random":PeerReadDataSetRANDOM,"both":PeerReadDataSetBOTH}
    cur_dataset = datasetdict[args.mask_type]
    if args.train_data == "full":
        dataset_train = cur_dataset(path_train,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="train")
    else:
        dataset_train = cur_dataset(path_train_frequency5,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="train")
    if args.test_data == "full":
        dataset_test = cur_dataset(path_test,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="test")
    else:
        dataset_test = cur_dataset(path_test_frequency5,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="test")
    return dataset_train,dataset_test_frequency5,ent_vocab

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
        ent_vocab = {"UNKNOWN":0,"MASK":1}
        for i,entity in enumerate(entitylist):
            ent_vocab[entity] = i+2
        return path_train[:-4]+"_frequency"+str(frequency)+".csv",path_test[:-4]+"_frequency"+str(frequency)+".csv",ent_vocab
    path = settings.citation_recommendation_dir
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    ent_vocab = build_ent_vocab(path_train)
    path_train_frequency5,path_test_frequency5,ent_vocab_frequency5 = extract_by_frequency(path_train,path_test,args.frequency)
    #randomでMASKするように一旦変更
    datasetdict = {"tail":AASCDataSet,"random":AASCDataSetRANDOM,"both":AASCDataSetBOTH}
    cur_dataset = datasetdict[args.mask_type]
    if args.train_data == "full":
        dataset_train = cur_dataset(path_train,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="train")
    else:
        dataset_train = cur_dataset(path_train_frequency5,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="train")
    if args.test_data == "full":
        dataset_test = cur_dataset(path_test,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="test")
    else:
        dataset_test = cur_dataset(path_test_frequency5,ent_vocab=ent_vocab,WINDOW_SIZE=args.WINDOW_SIZE,MAX_LEN=args.MAX_LEN,pretrained_model=args.pretrained_model,mode="test")
    print("----loading data done----")
    return dataset_train,dataset_test,ent_vocab


def load_data_intent_identification(model,ent_vocab):
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
            masked_lm_labels = torch.tensor([[-1] *256])
            position_ids = torch.tensor([[i for i in range(256)]])
            token_type_ids = torch.tensor([[1] + [0]*255])
            input_ids = torch.tensor([[ent_vocab[target_id]] + [-1]*255])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(255,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(256,255,dtype=torch.int)),dim=1)
            output = model(input_ids=input_ids.cuda(),position_ids=position_ids.cuda(),token_type_ids=token_type_ids.cuda(),masked_lm_labels=masked_lm_labels.cuda(),attention_mask=torch.stack([adj],dim=0).cuda())
            target_logits = output["sequence_output"][0][0]
            masked_lm_labels = torch.tensor([[-1] *256])
            position_ids = torch.tensor([[i for i in range(256)]])
            token_type_ids = torch.tensor([[1] + [0]*255])
            input_ids = torch.tensor([[ent_vocab[source_id]] + [-1]*255])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(255,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(256,255,dtype=torch.int)),dim=1)
            output = model(input_ids=input_ids.cuda(),position_ids=position_ids.cuda(),token_type_ids=token_type_ids.cuda(),masked_lm_labels=masked_lm_labels.cuda(),attention_mask=torch.stack([adj],dim=0).cuda())
            source_logits = output["sequence_output"][0][0]
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
