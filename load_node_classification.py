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

from itertools import product
import collections

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score
from sklearn.decomposition import PCA
from matplotlib import pyplot


#extract node classification data
#for each node in node classification data, collect all contexts for that
#if there is data whose tail node is what we want to collect, we collect all of them
#else, we collect all data whose head node is what we want to collect
def load_raw_data(args):
    if args.dataset == "AASC":
        dftrain5 = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train_frequency5.csv"),quotechar="'")
        dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
        dftest = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
        ftrain = open(os.path.join(settings.node_classification_dir,"title2task_train.txt"))
        ftest = open(os.path.join(settings.node_classification_dir,"title2task_test.txt"))
    else:
        dftrain5 = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"train_frequency5.csv"))
        dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"train.csv"))
        dftest = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"test.csv"))
        ftrain = open(os.path.join(settings.node_classification_PeerRead_dir,"title2task_PWCode_train.txt"))
        ftest = open(os.path.join(settings.node_classification_PeerRead_dir,"title2task_PWCode_test.txt"))
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
    """
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftest["source_id"],dftest["target_id"],dftest["left_citated_text"],dftest["right_citated_text"]):
        tail_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    """
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
        else:
            elements = [{"data":tail_all_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_all_dict[paper].keys())] + [{"data":head_all_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_all_dict[paper].keys())]
        if len(elements) == 0:
            print(paper)
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
        else:
            elements = [{"data":tail_all_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_all_dict[paper].keys())] + [{"data":head_all_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_all_dict[paper].keys())] 
        if len(elements) == 0:
            print(paper)
        X_test.append(elements)
        if task not in taskdict:
            taskn += 1
            taskdict[task] = taskn
        y_test.append(taskdict[task])
    print("lens")
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    return X_train,y_train,X_test,y_test

#convert each node into input_id
def convert_data(args,datas,ent_vocab,MAX_LEN,WINDOW_SIZE):
    if args.pretrained_model == "scibert":
        tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
    else:
        tokenizer =  BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case =False)
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

#get average of context embeddings for each node for all layers
def get_embeddings_all_layer(model,datas,MAX_LEN,WINDOW_SIZE):
    #それぞれのdataをモデルに入れてそのmaskedの場所のembeddingsを取得する
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
                sequence_output = output["sequence_output"][0]
                outputs_each_layer = output["outputs_each_layer"]
                for j,label in enumerate(data["masked_lm_labels"]):
                    if label != -1:
                        for k,outputs_layer in enumerate(outputs_each_layer):
                            #それぞれのlayerごとのembeddingsをappend
                            if len(X_elements) == k:
                                X_elements.append([np.array(outputs_layer[0][j].cpu())])
                            else:
                                X_elements[k].append(np.array(outputs_layer[0][j].cpu()))
                        X_label = label
                        break
            for k,X_element in enumerate(X_elements):
                embeddings_averaged = np.average(X_element,axis=0)
                if len(X_embeddings) == k:
                    X_embeddings.append([embeddings_averaged])
                else:
                    X_embeddings[k].append(embeddings_averaged)
            if i%1000 == 0:
                print(i)
    return X_embeddings

#get embeddings for each node by taking average of contexts
def load_data_SVM_with_context(args,model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    X_train,y_train,X_test,y_test = load_raw_data(args)
    if args.dataset == "AASC":
        converted_path_train = os.path.join(settings.citation_recommendation_dir,"SVM_train.json")
        converted_path_test = os.path.join(settings.citation_recommendation_dir,"SVM_test.json")
    else:
        converted_path_train = os.path.join(settings.citation_recommendation_PeerRead_dir,"SVM_train.json")
        converted_path_test = os.path.join(settings.citation_recommendation_PeerRead_dir,"SVM_test.json")
    if os.path.exists(converted_path_train):
        with open(converted_path_train) as f:
            X_train = json.load(f)
    else:
        X_train = convert_data(args,X_train,ent_vocab,MAX_LEN,WINDOW_SIZE)
        with open(converted_path_train,"w") as f:
            json.dump(X_train,f)
    if os.path.exists(converted_path_test):
        with open(converted_path_test) as f:
            X_test = json.load(f)
    else:
        X_test = convert_data(args,X_test,ent_vocab,MAX_LEN,WINDOW_SIZE)
        with open(converted_path_test,"w") as f:
            json.dump(X_test,f)
    X_train = get_embeddings(model,X_train,MAX_LEN,WINDOW_SIZE)
    X_test = get_embeddings(model,X_test,MAX_LEN,WINDOW_SIZE)
    return X_train,y_train,X_test,y_test

def load_data_SVM_from_pkl():
    paper_embeddings_dict = pickle.load(open(os.path.join(settings.citation_recommendation_dir,"AASC_embeddings.pkl"),"rb"))
    ftrain = open(os.path.join(settings.node_classification_dir,"title2task_train.txt"))
    ftest = open(os.path.join(settings.node_classification_dir,"title2task_test.txt"))
    X_train = []
    y_train = []
    taskdict = {}
    taskn = -1
    alln = 0
    for line in ftrain:
        l = line[:-1].split("\t")
        paper = l[0]
        task = l[1]
        X_train.append(paper_embeddings_dict[paper])
        if task not in taskdict:
            taskn += 1
            taskdict[task] = taskn
        y_train.append(taskdict[task])
    X_test = []
    y_test = []
    for line in ftest:
        l = line[:-1].split("\t")
        paper = l[0]
        task = l[1]
        X_test.append(paper_embeddings_dict[paper])
        if task not in taskdict:
            taskn += 1
            taskdict[task] = taskn
        y_test.append(taskdict[task])
    return X_train,y_train,X_test,y_test

#get embeddings for each node by taking average of contexts for all layer
def load_data_SVM_with_context_all_layer(model,ent_vocab,MAX_LEN,WINDOW_SIZE):
    X_train,y_train,X_test,y_test = load_raw_data()
    converted_path_train = os.path.join(settings.citation_recommendation_dir,"SVM_train.json")
    converted_path_test = os.path.join(settings.citation_recommendation_dir,"SVM_test.json")
    if os.path.exists(converted_path_train):
        with open(converted_path_train) as f:
            X_train = json.load(f)
    else:
        X_train = convert_data(args,X_train,ent_vocab,MAX_LEN,WINDOW_SIZE)
        with open(converted_path_train,"w") as f:
            json.dump(X_train,f)
    if os.path.exists(converted_path_test):
        with open(converted_path_test) as f:
            X_test = json.load(f)
    else:
        X_test = convert_data(args,X_test,ent_vocab,MAX_LEN,WINDOW_SIZE)
        with open(converted_path_test,"w") as f:
            json.dump(X_test,f)
    X_trains = get_embeddings_all_layer(model,X_train,MAX_LEN,WINDOW_SIZE)
    X_tests = get_embeddings_all_layer(model,X_test,MAX_LEN,WINDOW_SIZE)
    return X_trains,y_train,X_tests,y_test

#get each node embeddings from feed forward network by gradient descent
#currently unused
def get_representative_embeddings(model,ent_vocab,paper):
    #feed forward layerを読み込んでfreeze
    feed_forward = model.ent_lm_head
    #prepare loss function
    loss = torch.nn.CrossEntropyLoss()
    #prepare target tensor
    paper_id = ent_vocab[paper]
    target_tensor = torch.tensor([paper_id],device="cuda")
    #prepare paper tensor by random initialization
    paper_tensor = torch.rand(1,1,768,requires_grad=True,device="cuda")
    #prepare optimizer
    optimizer = torch.optim.AdamW([paper_tensor], lr=5e-3)
    #train for 100 epochs
    for epoch in range(2500):
        optimizer.zero_grad()
        output_tensor = feed_forward(paper_tensor)
        output_loss = loss(output_tensor[0],target_tensor)
        output_loss.backward()
        optimizer.step()
    return paper_tensor[0][0].detach().cpu().numpy()

#load node classification data from feed forward linear layer
def load_data_SVM_from_linear(model,ent_vocab):
    taskn = -1
    taskdict = {}
    X_train = []
    y_train = []
    ftrain = open(os.path.join(settings.node_classification_dir,"title2task_train.txt"))
    with torch.no_grad():
        for i,line in enumerate(ftrain):
            l = line[:-1].split("\t")
            paper = l[0]
            task = l[1]
            if task not in taskdict:
                taskn += 1
                taskdict[task] = taskn
            emb = np.array(model.ent_lm_head.decoder.weight[ent_vocab[paper]].cpu())
            X_train.append(emb)
            y_train.append(taskdict[task])
            if i % 1000 == 0:
                print(i)
        ftest = open(os.path.join(settings.node_classification_dir,"title2task_test.txt"))
        X_test = []
        y_test = []
        path_emb_test = os.path.join(settings.node_classification_dir,"paper_emb_dict_test_CrossEntropyLoss.binaryfile")
        for i,line in enumerate(ftest):
            l = line[:-1].split("\t")
            paper = l[0]
            task = l[1]
            emb = np.array(model.ent_lm_head.decoder.weight[ent_vocab[paper]].cpu())
            X_test.append(emb)
            y_test.append(taskdict[task])
            if i % 1000 == 0:
                print(i)
    return X_train,y_train,X_test,y_test

#load node classification data from feed forward neural network
def load_data_SVM_from_feedforward(model,ent_vocab):
    taskn = -1
    taskdict = {}
    X_train = []
    y_train = []
    ftrain = open(os.path.join(settings.node_classification_dir,"title2task_train.txt"))
    path_emb_train = os.path.join(settings.node_classification_dir,"paper_emb_dict_train_CrossEntropyLoss.binaryfile")
    if os.path.exists(path_emb_train):
        fet = open(path_emb_train,"rb")
        paper_dict_train = pickle.load(fet)
    else:
        paper_dict_train = {}
    for i,line in enumerate(ftrain):
        l = line[:-1].split("\t")
        paper = l[0]
        task = l[1]
        if task not in taskdict:
            taskn += 1
            taskdict[task] = taskn
        #entity_logits = model.ent_lm_head.decoder.weight[ent_vocab[paper]]
        if paper in paper_dict_train:
            emb = paper_dict_train[paper]
        else:
            emb = get_representative_embeddings(model,ent_vocab,paper)
            paper_dict_train[paper] = emb
        X_train.append(emb)
        y_train.append(taskdict[task])
        if i % 100 == 0:
            print(i)
    ftest = open(os.path.join(settings.node_classification_dir,"title2task_test.txt"))
    X_test = []
    y_test = []
    path_emb_test = os.path.join(settings.node_classification_dir,"paper_emb_dict_test_CrossEntropyLoss.binaryfile")
    if os.path.exists(path_emb_test):
        fet = open(path_emb_test,"rb")
        paper_dict_test = pickle.load(fet)
    else:
        paper_dict_test = {}
    for i,line in enumerate(ftest):
        l = line[:-1].split("\t")
        paper = l[0]
        task = l[1]
        #entity_logits = model.ent_lm_head.decoder.weight[ent_vocab[paper]]
        if paper in paper_dict_test:
            emb = paper_dict_test[paper]
        else:
            emb = get_representative_embeddings(model,ent_vocab,paper)
            paper_dict_test[paper] = emb
        X_test.append(emb)
        y_test.append(taskdict[task])
        if i % 100 == 0:
            print(i)
    fwtr = open(path_emb_train,"wb")
    fwte = open(path_emb_test,"wb")
    pickle.dump(paper_dict_train,fwtr)
    pickle.dump(paper_dict_test,fwte)
    return X_train,y_train,X_test,y_test

#load node classification data from AASC
def load_data_SVM(model,ent_vocab):
    ftrain = open(os.path.join(settings.node_classification_dir,"title2task_train.txt"))
    taskn = -1
    taskdict = {}
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
            entity_logits = model.ent_embeddings(torch.tensor(ent_vocab[paper]))
            X_train.append(np.array(entity_logits.cpu()))
            y_train.append(taskdict[task])
        ftest = open(os.path.join(settings.node_classification_dir,"title2task_test.txt"))
        X_test = []
        y_test = []
        for line in ftest:
            l = line[:-1].split("\t")
            paper = l[0]
            task = l[1]
            entity_logits = model.ent_embeddings(torch.tensor(ent_vocab[paper]))
            X_test.append(np.array(entity_logits.cpu()))
            y_test.append(taskdict[task])
    return X_train,y_train,X_test,y_test

#get average of context embeddings for each node from three-length sequences model
def get_embeddings_COKE(model,datas,ent_vocab,matrix):
    #それぞれのdataをモデルに入れてそのmaskedの場所のembeddingsを取得する
    X_embeddings = []
    loss_average = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)
    th_dict = {"head":0,"tail":2}
    with torch.no_grad():
        for i,elements in enumerate(datas):
            X_elements = []
            X_label = -1
            loss_element = 0
            for data in elements:
                target_ids = torch.tensor([ent_vocab[data["target_id"]]])
                source_ids = torch.tensor([ent_vocab[data["source_id"]]])
                position_ids = torch.tensor([[0,1,2]])
                contexts = torch.tensor([matrix[ent_vocab[data["target_id"]]][ent_vocab[data["source_id"]]]])
                token_type_ids = torch.tensor([[1,0,1]])
                attention_mask = torch.stack([torch.ones(3,3,dtype=torch.int)],dim=0)
                mask_positions = torch.tensor([th_dict[data["th"]]])
                output = model(target_ids=target_ids.cuda(),source_ids=source_ids.cuda(),position_ids=position_ids.cuda(),contexts=contexts.cuda(),token_type_ids=token_type_ids.cuda(),attention_mask=attention_mask.cuda(),mask_positions=mask_positions.cuda())
                sequence_output = output["sequence_output"][0]
                if data["th"] == "head":
                    sequence_output = sequence_output[0]
                    X_label = ent_vocab[data["target_id"]]
                else:
                    sequence_output = sequence_output[2]
                    X_label = ent_vocab[data["source_id"]]
                X_elements.append(np.array(sequence_output.cpu()))
                loss_element += output['loss'].cpu().detach().item()
            embeddings_averaged = np.average(X_elements,axis=0)
            X_embeddings.append(embeddings_averaged)
            averaged_tensor = torch.tensor([[embeddings_averaged]],device="cuda")
            ent_logits = model.ent_lm_head(averaged_tensor)
            target_tensor = torch.tensor([X_label],device="cuda")
            loss_average += loss_element/len(elements)
            if i % 1000 == 0:
                print(loss_average/(i+1))
    return X_embeddings

#load node classificaton data of model with 3-length sequences
def load_data_SVM_COKE(model,ent_vocab):
    X_train,y_train,X_test,y_test = load_raw_data()
    matrix = make_matrix(ent_vocab)
    X_train = get_embeddings_COKE(model,X_train,ent_vocab,matrix)
    X_test = get_embeddings_COKE(model,X_test,ent_vocab,matrix)
    return X_train,y_train,X_test,y_test

if __name__ == "__main__":
    X_train,y_train,X_test,y_test = load_data_SVM_from_pkl()
    print("SVM data load done")
    print("training start")
    print("PCA start")
    pca = PCA(n_components=2)
    pca.fit(X_test)
    X_test_visualization = pca.transform(X_test)
    print("PCA done: " + str(len(X_test)))
    print("Y length: " + str(len(y_test)))
    print("visualization start")
    fig, ax = pyplot.subplots(figsize=(20,20))
    X_test_colors = [[] for _ in range(max(y_test)+1)]
    y_test_colors = [[] for _ in range(max(y_test)+1)]
    colors_name = ["black","grey","tomato","saddlebrown","palegoldenrod","olivedrab","cyan","steelblue","midnightblue","darkviolet","magenta","pink","yellow"]
    for x,y in zip(X_test_visualization,y_test):
        X_test_colors[y].append(x)
        y_test_colors[y].append(y)
    for X_color,color in zip(X_test_colors,colors_name):
        X_color_x = np.array([X_place[0] for X_place in X_color])
        X_color_y = np.array([X_place[1] for X_place in X_color])
        ax.scatter(X_color_x,X_color_y,c=color)
    pyplot.savefig("images/TransBasedCitEmb_pkl.png") # 保存
    Cs = [2 , 2**5, 2 **10]
    gammas = [2 ** -9, 2 ** -6, 2** -3,2 ** 3, 2 ** 6, 2 ** 9]
    svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
    products = [(C,gamma) for C,gamma in product(Cs,gammas)]
    print("training done")
    for sv,product1 in zip(svs,products):
        test_label = sv.predict(X_test)
        print("正解率＝", accuracy_score(y_test, test_label))
        print("マクロ平均＝", f1_score(y_test, test_label,average="macro"))
        print("ミクロ平均＝", f1_score(y_test, test_label,average="micro"))
        print(collections.Counter(test_label))
