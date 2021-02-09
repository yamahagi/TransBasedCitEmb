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

WORD_PADDING_INDEX = 1
ENTITY_PADDING_INDEX = 1

class PeerReadDataSet(Dataset):
    def __init__(self, path, ent_vocab, WINDOW_SIZE, MAX_LEN):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.data = []
        self.tokenizer =  BertTokenizer.from_pretrained('pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
        df = pd.read_csv(path)
        jsonpath = os.path.join(self.dirname,self.filename[:-4]+"_window"+str(WINDOW_SIZE)+"_MAXLEN"+str(MAX_LEN)+".json")
        if os.path.exists(jsonpath):
            fids = open(jsonpath)
            dl = json.load(fids)
            self.data = dl
        else:
            target_ids = df["target_id"]
            source_ids = df["source_id"]
            left_citation_texts = df["left_citated_text"]
            right_citation_texts = df["right_citated_text"]
            citationcontextl = []
            masked_ids = []
            position_ids = []
            for i,(target_id,source_id,left_citation_text,right_citation_text) in enumerate(zip(target_ids,source_ids,left_citation_texts,right_citation_texts)):
                if i % 1000 == 0:
                    print(i)
                citationcontextl = []
                masked_ids = []
                position_ids = []
                token_type_ids = []
                citationcontextl.append(self.tokenizer.cls_token_id)
                citationcontextl.append(ent_vocab[target_id])
                citationcontextl.append(self.tokenizer.sep_token_id)
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                left_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(left_citation_text))
                right_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(right_citation_text))
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab["MASK"]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
                self.data.append({
                    'input_ids': citationcontextl[:MAX_LEN],
                    'masked_lm_labels' : masked_ids[:MAX_LEN],
                    'position_ids': position_ids[:MAX_LEN],
                    'token_type_ids': token_type_ids[:MAX_LEN],
                })
            fids = open(jsonpath,"w")
            json.dump(self.data,fids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask"]
        target_keys = ["masked_lm_labels","word_seq_len"]
        max_words = MAX_LEN
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}
        
        for sample in batch:
            word_pad = max_words - len(sample["input_ids"])
            if word_pad > 0:
                batch_x["input_ids"].append(sample["input_ids"]+[-1]*word_pad)
                batch_x["position_ids"].append(sample["position_ids"]+[0]*word_pad)
                batch_x["token_type_ids"].append(sample["token_type_ids"]+[0]*word_pad)
                batch_x["n_word_nodes"].append(max_words)
                batch_x["masked_lm_labels"].append(sample["masked_lm_labels"]+[-1]*word_pad)
                adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
                adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
                adj = torch.cat((adj,torch.zeros(MAX_LEN,word_pad,dtype=torch.int)),dim=1)
                #attention_maskは普通に文章内に対して1で文章外に対して0でいい
                batch_x['attention_mask'].append(adj)
                batch_y["masked_lm_labels"].append(sample["masked_lm_labels"]+[-1]*word_pad)
                batch_y["word_seq_len"].append(len(sample["input_ids"]))
            else:
                batch_x["input_ids"].append(sample["input_ids"])
                batch_x["position_ids"].append(sample["position_ids"])
                batch_x["token_type_ids"].append(sample["token_type_ids"])
                batch_x["n_word_nodes"].append(max_words)
                batch_x["masked_lm_labels"].append(sample["masked_lm_labels"])
                adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
                #attention_maskは普通に文章内に対して1で文章外に対して0でいい
                batch_x['attention_mask'].append(adj)
                batch_y["masked_lm_labels"].append(sample["masked_lm_labels"])
                batch_y["word_seq_len"].append(len(sample["input_ids"]))

        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)

class AASCDataSet(Dataset):
    def __init__(self, path, ent_vocab,WINDOW_SIZE,MAX_LEN):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.data = []
        self.tokenizer =  BertTokenizer.from_pretrained('pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
        df = pd.read_csv(path,quotechar="'")
        jsonpath = os.path.join(self.dirname,self.filename[:-4]+"_window"+str(WINDOW_SIZE)+"_MAXLEN"+str(MAX_LEN)+"_TBCN.json")
        if os.path.exists(jsonpath):
            fids = open(jsonpath)
            dl = json.load(fids)
            self.data = dl
        else:
            target_ids = df["target_id"]
            source_ids = df["source_id"]
            left_citation_texts = df["left_citated_text"]
            right_citation_texts = df["right_citated_text"]
            citationcontextl = []
            masked_ids = []
            position_ids = []
            for i,(target_id,source_id,left_citation_text,right_citation_text) in enumerate(zip(target_ids,source_ids,left_citation_texts,right_citation_texts)):
                if i % 1000 == 0:
                    print(i)
                citationcontextl = []
                masked_ids = []
                position_ids = []
                token_type_ids = []
                citationcontextl.append(self.tokenizer.cls_token_id)
                citationcontextl.append(ent_vocab[target_id])
                citationcontextl.append(self.tokenizer.sep_token_id)
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                left_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(left_citation_text))
                right_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(right_citation_text))
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab["MASK"]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
                self.data.append({
                    'input_ids': citationcontextl[:MAX_LEN],
                    'masked_lm_labels' : masked_ids[:MAX_LEN],
                    'position_ids': position_ids[:MAX_LEN],
                    'token_type_ids': token_type_ids[:MAX_LEN],
                })
            fids = open(jsonpath,"w")
            json.dump(self.data,fids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask"]
        target_keys = ["masked_lm_labels","word_seq_len"]
        max_words = MAX_LEN
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}
        
        for sample in batch:
            word_pad = max_words - len(sample["input_ids"])
            if word_pad > 0:
                batch_x["input_ids"].append(sample["input_ids"]+[-1]*word_pad)
                batch_x["position_ids"].append(sample["position_ids"]+[0]*word_pad)
                batch_x["token_type_ids"].append(sample["token_type_ids"]+[0]*word_pad)
                batch_x["n_word_nodes"].append(max_words)
                batch_x["masked_lm_labels"].append(sample["masked_lm_labels"]+[-1]*word_pad)
                adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
                adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
                adj = torch.cat((adj,torch.zeros(MAX_LEN,word_pad,dtype=torch.int)),dim=1)
                #attention_maskは普通に文章内に対して1で文章外に対して0でいい
                batch_x['attention_mask'].append(adj)
                batch_y["masked_lm_labels"].append(sample["masked_lm_labels"]+[-1]*word_pad)
                batch_y["word_seq_len"].append(len(sample["input_ids"]))
            else:
                batch_x["input_ids"].append(sample["input_ids"])
                batch_x["position_ids"].append(sample["position_ids"])
                batch_x["token_type_ids"].append(sample["token_type_ids"])
                batch_x["n_word_nodes"].append(max_words)
                batch_x["masked_lm_labels"].append(sample["masked_lm_labels"])
                adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
                #attention_maskは普通に文章内に対して1で文章外に対して0でいい
                batch_x['attention_mask'].append(adj)
                batch_y["masked_lm_labels"].append(sample["masked_lm_labels"])
                batch_y["word_seq_len"].append(len(sample["input_ids"]))

        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)


#入力: directory
def load_PeerRead_graph_data(path,frequency,WINDOW_SIZE,MAX_LEN):
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
    entvocab = build_ent_vocab(path_train)
    path_train_frequency5,path_test_frequency5,entvocab_frequency5 = extract_by_frequency(path_train,path_test,frequency)
    dataset_train = PeerReadDataSet(path_train,ent_vocab=entvocab,WINDOW_SIZE=WINDOW_SIZE,MAX_LEN=MAX_LEN)
    dataset_test = PeerReadDataSet(path_test,ent_vocab=entvocab,WINDOW_SIZE=WINDOW_SIZE,MAX_LEN=MAX_LEN)
    dataset_train_frequency5 = PeerReadDataSet(path_train_frequency5,ent_vocab=entvocab,WINDOW_SIZE=WINDOW_SIZE,MAX_LEN=MAX_LEN)
    dataset_test_frequency5 = PeerReadDataSet(path_test_frequency5,ent_vocab=entvocab,WINDOW_SIZE=WINDOW_SIZE,MAX_LEN=MAX_LEN)
    return dataset_train,dataset_test_frequency5,entvocab

#入力: directory
def load_AASC_graph_data(path,frequency,WINDOW_SIZE,MAX_LEN):
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
    entvocab = build_ent_vocab(path_train)
    path_train_frequency5,path_test_frequency5,entvocab_frequency5 = extract_by_frequency(path_train,path_test,frequency)
    dataset_train = AASCDataSet(path_train,ent_vocab=entvocab,WINDOW_SIZE,MAX_LEN)
    dataset_test = AASCDataSet(path_test,ent_vocab=entvocab,WINDOW_SIZE,MAX_LEN)
    dataset_train_frequency5 = AASCDataSet(path_train_frequency5,ent_vocab=entvocab,WINDOW_SIZE,MAX_LEN)
    dataset_test_frequency5 = AASCDataSet(path_test_frequency5,ent_vocab=entvocab,WINDOW_SIZE,MAX_LEN)
    print("----loading data done----")
    return dataset_train,dataset_test,entvocab

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
            masked_lm_labels1 = torch.tensor([[-1] *512])
            position_ids1 = torch.tensor([[i for i in range(512)]])
            token_type_ids1 = torch.tensor([[1] + [0]*511])
            input_ids1 = torch.tensor([[entvocab[paper]] + [-1]*511])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(511,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(512,511,dtype=torch.int)),dim=1)
            if i % 1000 == 0:
                print("all")
                print(len1)
                print(i)
            output = model(input_ids=input_ids1.cuda(),position_ids=position_ids1.cuda(),token_type_ids=token_type_ids1.cuda(),masked_lm_labels=masked_lm_labels1.cuda(),attention_mask=torch.stack([adj],dim=0).cuda())
            entity_logits = output["sequence_output"][0][0]
            X_train.append(np.array(entity_logits.cpu()))
            y_train.append(taskdict[task])
        ftest = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_test.txt")
        X_test = []
        y_test = []
        for line in ftest:
            l = line[:-1].split("\t")
            paper = l[0]
            task = l[1]
            masked_lm_labels1 = torch.tensor([[-1] *512])
            position_ids1 = torch.tensor([[i for i in range(512)]])
            token_type_ids1 = torch.tensor([[1] + [0]*511])
            input_ids1 = torch.tensor([[entvocab[paper]] + [-1]*511])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(511,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(512,511,dtype=torch.int)),dim=1)
            output = model(input_ids=input_ids1.cuda(),position_ids=position_ids1.cuda(),token_type_ids=token_type_ids1.cuda(),masked_lm_labels=masked_lm_labels1.cuda(),attention_mask=torch.stack([adj],dim=0).cuda())
            entity_logits = output["sequence_output"][0][0]
            X_test.append(np.array(entity_logits.cpu()))
            y_test.append(taskdict[task])
    return X_train,y_train,X_test,y_test

if __name__ == "__main__":
    path = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/citationcontexts.txt"
    entitypath = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/entity.txt"
    fe = open(entitypath)
    ent_vocab = {"UNKNOWN":0,"MASK":1}
    for i,line in enumerate(fe):
        ent_vocab[line.rstrip("\n")] = i+2
