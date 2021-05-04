import pandas as pd
from collections import defaultdict
import random
from transformers import RobertaTokenizer,BertTokenizer
import torch
import numpy as np

def load_raw_data():
    dftrain5 = pd.read_csv("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train_frequency5.csv",quotechar="'")
    dftrain = pd.read_csv("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train.csv",quotechar="'")
    dftest = pd.read_csv("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/test.csv",quotechar="'")
    ftrain = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_train.txt")
    ftest = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_test.txt")
    tail_train5_dict = defaultdict(dict)
    head_train5_dict = defaultdict(dict)
    tail_all_dict = defaultdict(dict)
    head_all_dict = defaultdict(dict)
    both_all_dict = defaultdict(dict)
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain5["source_id"],dftrain5["target_id"],dftrain5["left_citated_text"],dftrain5["right_citated_text"]):
        tail_train5_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train5_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain["source_id"],dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"]):
        tail_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftest["source_id"],dftest["target_id"],dftest["left_citated_text"],dftest["right_citated_text"]):
        tail_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    tail5_number = 0
    head5_number = 0
    tailall_number = 0
    headall_number = 0
    bothall_number = 0
    X_train = []
    y_train = []
    taskdict = {}
    taskn = -1
    alln = 0
    for line in ftrain:
        alln += 1
        l = line[:-1].split("\t")
        paper = l[0]
        task = l[1]
        elements = []
        if paper in tail_train5_dict and tail_train5_dict[paper] != {}:
            tail5_number += 1
            elements = [{"data":tail_train5_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train5_dict[paper].keys())]
        elif paper in head_train5_dict and head_train5_dict[paper] != {}:
            head5_number += 1
            elements = [{"data":head_train5_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train5_dict[paper].keys())]
        X_train.append(elements)
        if task not in taskdict:
            taskn += 1
            taskdict[task] = taskn
        y_train.append(taskdict[task])
    tail5_number = 0
    head5_number = 0
    tailall_number = 0
    headall_number = 0
    bothall_number = 0
    alln = 0
    X_test = []
    y_test = []
    th = "tail"
    for line in ftest:
        alln += 1
        l = line[:-1].split("\t")
        paper = l[0]
        task = l[1]
        elements = []
        #paperがtail_train5_dictに入っている
        #tail or headどちらに取り出すべきembeddingsが入っているかをthで指し示している
        if paper in tail_train5_dict and tail_train5_dict[paper] != {}:
            tail5_number += 1
            elements = [{"data":tail_train5_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train5_dict[paper].keys())]
        elif paper in head_train5_dict and head_train5_dict[paper] != {}:
            head5_number += 1
            elements = [{"data":head_train5_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train5_dict[paper].keys())]
        X_test.append(elements)
        if task not in taskdict:
            taskn += 1
            taskdict[task] = taskn
        y_test.append(taskdict[task])
    return X_train,y_train,X_test,y_test

#それぞれの辞書をinput_idに変換
def convert_data(datas,ent_vocab):
    tokenizer =  BertTokenizer.from_pretrained('/home/ohagi_masaya/TransBasedCitEmb/pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
    MAX_LEN = 256
    WINDOW_SIZE = 125
    converted_datas = []
    converted_elements = []
    for i,elements in enumerate(datas):
        converted_elements = []
        for data in elements:
            target_id = data["target_id"]
            source_id = data["source_id"]
            left_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data["data"]["left_citated_text"]))
            right_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data["data"]["right_citated_text"]))
            citationcontextl = []
            masked_ids = []
            position_ids = []
            token_type_ids = []
            if data["th"] == "head":
                citationcontextl.append(tokenizer.cls_token_id)
                #citationcontextl.append(ent_vocab[target_id])
                citationcontextl.append(ent_vocab["MASK"])
                citationcontextl.append(tokenizer.sep_token_id)
                masked_ids.extend([-1,ent_vocab[target_id],-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
            else:
                citationcontextl.append(tokenizer.cls_token_id)
                citationcontextl.append(ent_vocab[target_id])
                citationcontextl.append(tokenizer.sep_token_id)
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab["MASK"]] + right_citation_tokenized[:WINDOW_SIZE])
                #citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
            converted_elements.append({
                'input_ids': citationcontextl[:MAX_LEN],
                'masked_lm_labels' : masked_ids[:MAX_LEN],
                'position_ids': position_ids[:MAX_LEN],
                'token_type_ids': token_type_ids[:MAX_LEN],
            })
        converted_datas.append(converted_elements)
    return converted_datas

def get_embeddings(model,datas):
    #それぞれのdataをモデルに入れてそのmaskedの場所のembeddingsを取得する
    MAX_LEN = 256
    WINDOW_SIZE = 125
    X_embeddings = []
    with torch.no_grad():
        for elements in datas:
            X_elements = []
            for data in elements:
                input_ids = torch.tensor([data["input_ids"]+[-1]*(MAX_LEN-len(data["input_ids"]))])
                position_ids = torch.tensor([data["position_ids"]+[0]*(MAX_LEN-len(data["position_ids"]))])
                token_type_ids = torch.tensor([data["token_type_ids"]+[0]*(MAX_LEN-len(data["token_type_ids"]))])
                masked_lm_labels = torch.tensor([data["masked_lm_labels"]+[-1]*(MAX_LEN-len(data["masked_lm_labels"]))])
                l = len(data["input_ids"])
                adj = torch.ones(l,l,dtype=torch.int)
                adj = torch.cat((adj,torch.ones(MAX_LEN-l,adj.shape[1],dtype=torch.int)),dim=0)
                adj = torch.cat((adj,torch.zeros(MAX_LEN,MAX_LEN-l,dtype=torch.int)),dim=1)
                output = model(input_ids=input_ids.cuda(),position_ids=position_ids.cuda(),token_type_ids=token_type_ids.cuda(),masked_lm_labels=masked_lm_labels.cuda(),attention_mask=torch.stack([adj],dim=0).cuda())
                entity_logits = output["sequence_output"][0]
                for i,label in enumerate(data["masked_lm_labels"]):
                    if label != -1:
                        entity_logits = entity_logits[i]
                        break
                X_elements.append(np.array(entity_logits.cpu()))
            X_embeddings.append(np.average(X_elements,axis=0))
    return X_embeddings

def load_data_SVM_with_context(model,ent_vocab):
    X_train,y_train,X_test,y_test = load_raw_data()
    X_train = convert_data(X_train,ent_vocab)
    X_test = convert_data(X_test,ent_vocab)
    X_train = get_embeddings(model,X_train)
    X_test = get_embeddings(model,X_test)
    return X_train,y_train,X_test,y_test


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
if __name__ == "__main__":
    ent_vocab = build_ent_vocab("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train.csv")
    load_data_SVM_with_context(ent_vocab)
