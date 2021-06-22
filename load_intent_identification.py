import pandas as pd
from collections import defaultdict
import random
from transformers import RobertaTokenizer,BertTokenizer
from dataloader_CoKE import makecitationmatrix_AASC,make_matrix
import torch
import numpy as np
from utils import build_ent_vocab
import settings
import os
import copy
import pickle
import settings
import json

def load_raw_data():
    dftrain5 = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train_frequency5.csv"),quotechar="'")
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
    f = open(os.path.join(settings.intent_identification_dir,"id2intent.txt"))
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
    tail_all_number = 0
    head_all_number = 0
    X = []
    y = []
    intentdict = {}
    intentn = -1
    paper_dict = {}
    for i,line in enumerate(f):
        if i == 0:
            continue
        l = line[:-1].split("\t")
        target_id = l[0]
        source_id = l[1]
        intent = l[2]
        X.append((target_id,source_id))
        if intent not in intentdict:
            intentn += 1
            intentdict[intent] = intentn
        y.append(intentdict[intent])
        for paper in [target_id,source_id]:
            if paper not in paper_dict:
                elements = []
                if paper in tail_train5_dict and tail_train5_dict[paper] != {}:
                    tail5_number += 1
                    elements = [{"data":tail_train5_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train5_dict[paper].keys())]
                elif paper in head_train5_dict and head_train5_dict[paper] != {}:
                    head5_number += 1
                    elements = [{"data":head_train5_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train5_dict[paper].keys())]
                elif paper in tail_all_dict and tail_all_dict[paper] != {}:
                    tail_all_number += 1
                    elements = [{"data":tail_all_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_all_dict[paper].keys())]
                elif paper in head_all_dict and head_all_dict[paper] != {}:
                    head_all_number += 1
                    elements = [{"data":head_all_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_all_dict[paper].keys())]
                else:
                    print("aaa")
                paper_dict[paper] = elements
    print(tail5_number)
    print(head5_number)
    print(tail_all_number)
    print(head_all_number)
    return X,y,paper_dict

#それぞれの辞書をinput_idに変換
def convert_data(paper_dict,ent_vocab,MAX_LEN,WINDOW_SIZE):
    tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
    converted_elements = []
    converted_paper_dict = {}
    for i,paper in enumerate(paper_dict):
        converted_elements = []
        elements = paper_dict[paper]
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
                citationcontextl.extend([tokenizer.cls_token_id,ent_vocab["MASK"],tokenizer.sep_token_id])
                masked_ids.extend([-1,ent_vocab[target_id],-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
            else:
                citationcontextl.extend([tokenizer.cls_token_id,ent_vocab[target_id],tokenizer.sep_token_id])
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab["MASK"]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
            converted_elements.append({
                'input_ids': citationcontextl[:MAX_LEN],
                'masked_lm_labels' : masked_ids[:MAX_LEN],
                'position_ids': position_ids[:MAX_LEN],
                'token_type_ids': token_type_ids[:MAX_LEN],
            })
        converted_paper_dict[paper] = converted_elements
    return converted_paper_dict

def get_embeddings(model,paper_dict,MAX_LEN,WINDOW_SIZE):
    #それぞれのdataをモデルに入れてそのmaskedの場所のembeddingsを取得する
    paper_embeddings_dict = {}
    with torch.no_grad():
        for i,paper in enumerate(paper_dict):
            elements = paper_dict[paper]
            X_elements = []
            X_label = -1
            loss_element = 0
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
                sequence_output = output["sequence_output"][0]
                outputs_final_layer = output["outputs_each_layer"][-1]
                for j,label in enumerate(data["masked_lm_labels"]):
                    if label != -1:
                        X_elements.append(np.array(outputs_final_layer[0][j].cpu()))
                        X_label = label
                        break
            embeddings_averaged = np.average(X_elements,axis=0)
            paper_embeddings_dict[paper] = embeddings_averaged
            if i % 100 == 0:
                print(i)
    return paper_embeddings_dict

def load_data_intent_identification_with_context(model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    X,y,paper_dict = load_raw_data()
    converted_path = os.path.join(settings.intent_identification_dir,"intent_identification.json")
    if os.path.exists(converted_path):
        with open(converted_path) as f:
            paper_dict = json.load(f)
    else:
        paper_dict = convert_data(paper_dict,ent_vocab,MAX_LEN,WINDOW_SIZE)
        with open(converted_path,"w") as f:
            json.dump(paper_dict,f)
    paper_embeddings_dict = get_embeddings(model,paper_dict,MAX_LEN,WINDOW_SIZE)
    X = [(paper_embeddings_dict[target_id],paper_embeddings_dict[source_id]) for (target_id,source_id) in X]
    return X,y


if __name__ == "__main__":
    ent_vocab = build_ent_vocab("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train.csv")
    load_data_SVM_with_context(ent_vocab)
