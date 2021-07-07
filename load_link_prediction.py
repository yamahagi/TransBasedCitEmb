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

#extract link prediction data
#for each node in node classification data, collect all contexts for that
#if there is data whose tail node is what we want to collect, we collect all of them
#else, we collect all data whose head node is what we want to collect
def load_raw_data():
    dftrain5 = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train_frequency5.csv"),quotechar="'")
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
    ftrain = open(os.path.join(settings.node_classification_dir,"title2task_train.txt"))
    ftest = open(os.path.join(settings.node_classification_dir,"title2task_test.txt"))
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

#convert each node into input_id
def convert_data(datas,ent_vocab,MAX_LEN,WINDOW_SIZE):
    tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
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
            #mask head node
            if data["th"] == "head":
                #append head node
                citationcontextl.extend([tokenizer.cls_token_id,ent_vocab["MASK"],tokenizer.sep_token_id])
                masked_ids.extend([-1,ent_vocab[target_id],-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                #append citation context and tail node
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
            else:
                #mask tail node
                #append head node
                citationcontextl.extend([tokenizer.cls_token_id,ent_vocab[target_id],tokenizer.sep_token_id])
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                #append citation context and tail node
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
        converted_datas.append(converted_elements)
    return converted_datas

#get average of context embeddings for each node
def get_embeddings(model,datas,MAX_LEN,WINDOW_SIZE):
    X_embeddings = []
    loss_average = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        for i,elements in enumerate(datas):
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
                outputs_final_layer = output["outputs_each_layer"][-1:]
                for j,label in enumerate(data["masked_lm_labels"]):
                    if label != -1:
                        for outputs_layer in outputs_final_layer:
                            X_elements.append(np.array(outputs_layer[0][j].cpu()))
                        X_label = label
                        break
                loss_element += output['loss'].cpu().detach().item()
            embeddings_averaged = np.average(X_elements,axis=0)
            X_embeddings.append(embeddings_averaged)
            loss_average += loss_element/len(elements)
            if i % 1000 == 0:
                print(i,loss_average/(i+1))
    return X_embeddings

def load_link_prediction(ent_vocab):
    #test_pathからそれぞれのnodeをkeyとしてciteされたnodeのlistをvalueとしたdictを読み込む
    df = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
    link_dict = defaultdict(set)
    for target_id,source_id in zip(df["target_id"],df["source_id"]):
        link_dict[ent_vocab[target_id]].add(ent_vocab[source_id])
    return link_dict

#get embeddings for each node by taking average of contexts
def load_data_link_prediction_with_context(model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    #それぞれのnodeごとに関連するdataを取り出す
    X_train,y_train,X_test,y_test = load_raw_data()
    #nodeごとにembeddingsを取り出す
    converted_path_train = os.path.join(settings.citation_recommendation_dir,"SVM_train.json")
    converted_path_test = os.path.join(settings.citation_recommendation_dir,"SVM_test.json")
    if os.path.exists(converted_path_train):
        with open(converted_path_train) as f:
            X_train = json.load(f)
    else:
        X_train = convert_data(X_train,ent_vocab,MAX_LEN,WINDOW_SIZE)
        with open(converted_path_train,"w") as f:
            json.dump(X_train,f)
    if os.path.exists(converted_path_test):
        with open(converted_path_test) as f:
            X_test = json.load(f)
    else:
        X_test = convert_data(X_test,ent_vocab,MAX_LEN,WINDOW_SIZE)
        with open(converted_path_test,"w") as f:
            json.dump(X_test,f)
    X_train = get_embeddings(model,X_train,MAX_LEN,WINDOW_SIZE)
    X_test = get_embeddings(model,X_test,MAX_LEN,WINDOW_SIZE)
    return X_train,y_train,X_test,y_test

#paper embeddingsのpathを引数とする
#entity2idのpathも引数とする
def link_prediction(model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    #link predictionのデータを読み込む
    #-> それぞれのnodeをkeyとしてciteされたnodeのlistをvalueとしたdictを読み込む
    link_dict = load_link_prediction(ent_vocab)
    #それぞれのnodeのembeddingsを読み込む
    paper_embeddings = load_data_link_prediction_with_context(ent_vocab,model)
    #dict内のkeyごとにfaissを用いて1001までnodeを最近傍探索
    query_embeddings = []
    y_true = []
    for target_id in link_dict:
        query_embeddings.append(paper_embeddings[target_id])
        y_true.append(np.array([[source_id,1.0] for source_id in link_dict[target_id]]))
    query_embeddings = np.array(query_embeddings)
    y_true = y_true
    y_true = to_typed_list(y_true)

    #prepare faiss
    d = len(query_embeddings[0])                           # dimension
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(paper_embeddings)                  # add vectors to the index
    print(index.ntotal)

    k = 1001                          # we want to see 1000 nearest neighbors(removing node itself)
    D, I = index.search(paper_embeddings[:5], k) # sanity check
    print(I)
    print(D)
    D, I = index.search(query_embeddings, k)     # actual search
    #MRRを測る
    y_pred = np.array([[[I[row][column],1.0] for column in range(len(I[row][1:]))] for row in range(len(I))])
    print(type(y_pred))
    print(y_pred.ndim)
    print(mrr_metrics(y_true,y_pred,k-100))
    #MAPを測る
    print(map_metrics(y_true,y_pred,k-100))
    #hits at 5
    print(hits_at_k(y_true,y_pred,5))

if __name__ == "__main__":
    ent_vocab = build_ent_vocab("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train.csv")
    load_data_SVM_with_context(ent_vocab)
