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

"""
import faiss
from rank_eval import map as map_metrics
from rank_eval import mrr as mrr_metrics
from rank_eval.utils import to_typed_list
from rank_eval import hits_at_k
"""

import pandas as pd
import os
from collections import defaultdict

#extract link prediction data
#for each node in node classification data, collect all contexts for that
#if there is data whose tail node is what we want to collect, we collect all of them
#else, we collect all data whose head node is what we want to collect
def load_raw_data():
    dftrain5 = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train_frequency5.csv"),quotechar="'")
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
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
    paper_dict = {}
    for i,target_id in enumerate(dftrain["target_id"]):
        if target_id not in paper_dict:
            if target_id in tail_train5_dict:
                paper_dict[target_id] = [{"data":tail_train5_dict[target_id][paper],"target_id":target_id,"source_id":paper,"th":"tail"} for paper in list(tail_train5_dict[target_id].keys())]
            elif target_id in head_train5_dict:
                paper_dict[target_id] = [{"data":head_train5_dict[target_id][paper],"target_id":paper,"source_id":target_id,"th":"head"} for paper in list(head_train5_dict[target_id].keys())]
            elif target_id in tail_all_dict:
                paper_dict[target_id] = [{"data":tail_all_dict[target_id][paper],"target_id":target_id,"source_id":paper,"th":"tail"} for paper in list(tail_all_dict[target_id].keys())]
            elif target_id in head_all_dict:
                paper_dict[target_id] = [{"data":head_all_dict[target_id][paper],"target_id":paper,"source_id":target_id,"th":"head"} for paper in list(head_all_dict[target_id].keys())]
            else:
                print("aaa")
    for i,source_id in enumerate(dftrain["source_id"]):
        if source_id not in paper_dict:
            if source_id in tail_train5_dict:
                paper_dict[source_id] = [{"data":tail_train5_dict[source_id][paper],"target_id":source_id,"source_id":paper,"th":"tail"} for paper in list(tail_train5_dict[source_id].keys())]
            elif source_id in head_train5_dict:
                paper_dict[source_id] = [{"data":head_train5_dict[source_id][paper],"target_id":paper,"source_id":source_id,"th":"head"} for paper in list(head_train5_dict[source_id].keys())]
            elif source_id in tail_all_dict:
                paper_dict[source_id] = [{"data":tail_all_dict[source_id][paper],"target_id":source_id,"source_id":paper,"th":"tail"} for paper in list(tail_all_dict[source_id].keys())]
            elif source_id in head_all_dict:
                paper_dict[source_id] = [{"data":head_all_dict[source_id][paper],"target_id":paper,"source_id":source_id,"th":"head"} for paper in list(head_all_dict[source_id].keys())]
            else:
                print("aaa")
    return paper_dict

#convert each node into input_id
def convert_data(paper_dict,ent_vocab,MAX_LEN,WINDOW_SIZE):
    tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
    converted_paper_dict = {}
    converted_elements = []
    for i,paper_id in enumerate(paper_dict):
        elements = paper_dict[paper_id]
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
        converted_paper_dict[paper_id] = converted_elements
    return converted_paper_dict

#get average of context embeddings for each node
def get_embeddings(model,paper_dict,MAX_LEN,WINDOW_SIZE):
    paper_embeddings_dict = {}
    loss_average = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        for i,paper_id in enumerate(paper_dict):
            elements = paper_dict[paper_id]
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
            paper_embeddings_dict[paper_id] = embeddings_averaged
            loss_average += loss_element/len(elements)
            if i % 1000 == 0:
                print(i,loss_average/(i+1))
    return paper_embeddings_dict

def load_link_prediction(ent_vocab):
    #test_pathからそれぞれのnodeをkeyとしてciteされたnodeのlistをvalueとしたdictを読み込む
    df = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
    link_dict = defaultdict(set)
    for target_id,source_id in zip(df["target_id"],df["source_id"]):
        link_dict[ent_vocab[target_id]-2].add(ent_vocab[source_id]-2)
    return link_dict

#get embeddings for each node by taking average of contexts
def load_data_link_prediction_with_context(args,model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    #それぞれのnodeごとに関連するdataを取り出す
    X_train,y_train,X_test,y_test = load_raw_data()
    #nodeごとにembeddingsを取り出す
    converted_path_train = os.path.join(settings.citation_recommendation_dir,"SVM_train"+args.pretrained_model+".json")
    converted_path_test = os.path.join(settings.citation_recommendation_dir,"SVM_test"+args.pretrained_model+".json")
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

def load_data_link_prediction_from_pkl(ent_vocab):
    paper_list = sorted([key for key in ent_vocab.keys()][2:],key= lambda x:ent_vocab[x])
    paper_embeddings_dict = pickle.load(open(os.path.join(settings.citation_recommendation_dir,"AASC_embeddings.pkl"),"rb"))
    paper_embeddings_list = []
    for paper in paper_list:
        paper_embeddings_list.append(paper_embeddings_dict[paper])
    return np.array(paper_embeddings_list)

#paper embeddingsのpathを引数とする
#entity2idのpathも引数とする
def save_embeddings(args,model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    #link predictionのデータを読み込む
    #それぞれのnodeのembeddingsを読み込む
    print("----loading data----")
    paper_dict = load_raw_data()
    print("----converting data----")
    paper_dict = convert_data(paper_dict,ent_vocab,MAX_LEN,WINDOW_SIZE)
    print("----making embeddings----")
    paper_embeddings_dict = get_embeddings(model,paper_dict,MAX_LEN,WINDOW_SIZE)
    print("----saving embeddings----")
    pkl_name = str(args.dataset)+"embeddings_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+".pkl"
    with open(pkl_name, "wb") as tf:
        pickle.dump(paper_embeddings_dict,tf)
    print("----loading embeddings----")
    with open(pkl_name,"rb") as tf:
        paper_embeddings_dict = pickle.load(tf)

def link_prediction(model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    #save embeddings
    link_dict = load_link_prediction(ent_vocab)
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
    print(hits_at_k(y_true,y_pred,10))

if __name__ == "__main__":
    ent_vocab = build_ent_vocab("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train.csv")
    link_prediction(ent_vocab)
