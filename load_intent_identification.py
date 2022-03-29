import pandas as pd
from collections import defaultdict
import random
from transformers import RobertaTokenizer,BertTokenizer,BertModel
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

from sklearn.manifold import TSNE
from matplotlib import pyplot

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score
from collections import Counter
from itertools import product
import collections

def load_raw_data(args):
    dftrain5 = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train_frequency5.csv"),quotechar="'")
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
    tail_train5_dict = defaultdict(dict)
    head_train5_dict = defaultdict(dict)
    tail_train_dict = defaultdict(dict)
    head_train_dict = defaultdict(dict)
    tail_all_dict = defaultdict(dict)
    head_all_dict = defaultdict(dict)
    both_all_dict = defaultdict(dict)
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain5["source_id"],dftrain5["target_id"],dftrain5["left_citated_text"],dftrain5["right_citated_text"]):
        tail_train5_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train5_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        tail_train_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain["source_id"],dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"]):
        tail_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        tail_train_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftest["source_id"],dftest["target_id"],dftest["left_citated_text"],dftest["right_citated_text"]):
        tail_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        both_all_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    tail5_number = 0
    head5_number = 0
    tail_train_number = 0
    head_train_number = 0
    X = []
    y = []
    intentdict = {}
    intentn = -1
    paper_dict = {}
    f = open(os.path.join(settings.intent_identification_dir,"id2intent.txt"))
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
    for (target_id,source_id) in X:
        for paper in [target_id,source_id]:
            if paper not in paper_dict:
                elements = []
                if paper in tail_train5_dict and tail_train5_dict[paper] != {}:
                    tail5_number += 1
                    elements = [{"data":tail_train5_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train5_dict[paper].keys())]
                elif paper in head_train5_dict and head_train5_dict[paper] != {}:
                    head5_number += 1
                    elements = [{"data":head_train5_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train5_dict[paper].keys())]
                elif paper in tail_train_dict and tail_train_dict[paper] != {}:
                    tail_train_number += 1
                    elements = [{"data":tail_train_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train_dict[paper].keys())]
                elif paper in head_train_dict and head_train_dict[paper] != {}:
                    head_train_number += 1
                    elements = [{"data":head_train_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train_dict[paper].keys())]
                else:
                    print("Unfound: "+paper)
                paper_dict[paper] = elements
    return X,y,paper_dict

def load_raw_data_PeerRead(args):
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"train.csv"))
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"test.csv"))
    tail_train_dict = defaultdict(dict)
    head_train_dict = defaultdict(dict)
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain["source_id"],dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"]):
        tail_train_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    tail_train_number = 0
    head_train_number = 0
    X = []
    y = []
    intentdict = {}
    intentn = -1
    paper_dict = {}
    df = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"test_intent_annotated.csv"))
    for target_id,source_id,intent in zip(df["target_id"],df["source_id"],df["intent"]):
        intent = int(intent)
        X.append((target_id,source_id))
        if intent not in intentdict:
            intentn += 1
            intentdict[intent] = intentn
        y.append(intentdict[intent])
    for (target_id,source_id) in X:
        for paper in [target_id,source_id]:
            if paper not in paper_dict:
                elements = []
                if paper in tail_train_dict and tail_train_dict[paper] != {}:
                    tail_train_number += 1
                    elements = [{"data":tail_train_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train_dict[paper].keys())]
                elif paper in head_train_dict and head_train_dict[paper] != {}:
                    head_train_number += 1
                    elements = [{"data":head_train_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train_dict[paper].keys())]
                else:
                    print("Unfound: "+paper)
                paper_dict[paper] = elements
    return X,y,paper_dict

def load_raw_data_SYNTHCI_AASC(args):
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
    tail_train_dict = defaultdict(dict)
    head_train_dict = defaultdict(dict)
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain["source_id"],dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"]):
        tail_train_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    tail_train_number = 0
    head_train_number = 0
    X = []
    y = []
    intentdict = {}
    intentn = -1
    paper_dict = {}
    df = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test_intent_annotated.csv"))
    for target_id,source_id,intent in zip(df["target_id"],df["source_id"],df["intent"]):
        intent = int(intent)
        X.append((target_id,source_id))
        if intent not in intentdict:
            intentn += 1
            intentdict[intent] = intentn
        y.append(intentdict[intent])
    for (target_id,source_id) in X:
        for paper in [target_id,source_id]:
            if paper not in paper_dict:
                elements = []
                if paper in tail_train_dict and tail_train_dict[paper] != {}:
                    tail_train_number += 1
                    elements = [{"data":tail_train_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train_dict[paper].keys())]
                elif paper in head_train_dict and head_train_dict[paper] != {}:
                    head_train_number += 1
                    elements = [{"data":head_train_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train_dict[paper].keys())]
                else:
                    print("Unfound: "+paper)
                paper_dict[paper] = elements
    return X,y,paper_dict

def load_raw_data_PeerRead(args):
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"train.csv"))
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"test.csv"))
    tail_train_dict = defaultdict(dict)
    head_train_dict = defaultdict(dict)
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain["source_id"],dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"]):
        tail_train_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    tail_train_number = 0
    head_train_number = 0
    X = []
    y = []
    intentdict = {}
    intentn = -1
    paper_dict = {}
    df = pd.read_csv(os.path.join(settings.citation_recommendation_PeerRead_dir,"test_intent_annotated.csv"))
    for target_id,source_id,intent in zip(df["target_id"],df["source_id"],df["intent"]):
        intent = int(intent)
        X.append((target_id,source_id))
        if intent not in intentdict:
            intentn += 1
            intentdict[intent] = intentn
        y.append(intentdict[intent])
    for (target_id,source_id) in X:
        for paper in [target_id,source_id]:
            if paper not in paper_dict:
                elements = []
                if paper in tail_train_dict and tail_train_dict[paper] != {}:
                    tail_train_number += 1
                    elements = [{"data":tail_train_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train_dict[paper].keys())]
                elif paper in head_train_dict and head_train_dict[paper] != {}:
                    head_train_number += 1
                    elements = [{"data":head_train_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train_dict[paper].keys())]
                else:
                    print("Unfound: "+paper)
                paper_dict[paper] = elements
    return X,y,paper_dict

def load_raw_data_SYNTHCI_AASC(args):
    dftrain = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"train.csv"),quotechar="'")
    dftest = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"),quotechar="'")
    tail_train_dict = defaultdict(dict)
    head_train_dict = defaultdict(dict)
    for source_id,target_id,left_citated_text,right_citated_text in zip(dftrain["source_id"],dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"]):
        tail_train_dict[source_id][target_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
        head_train_dict[target_id][source_id] = {"left_citated_text":left_citated_text,"right_citated_text":right_citated_text}
    tail_train_number = 0
    head_train_number = 0
    X = []
    y = []
    intentdict = {}
    intentn = -1
    paper_dict = {}
    df = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test_intent_annotated.csv"))
    for target_id,source_id,intent in zip(df["target_id"],df["source_id"],df["intent"]):
        intent = int(intent)
        X.append((target_id,source_id))
        if intent not in intentdict:
            intentn += 1
            intentdict[intent] = intentn
        y.append(intentdict[intent])
    for (target_id,source_id) in X:
        for paper in [target_id,source_id]:
            if paper not in paper_dict:
                elements = []
                if paper in tail_train_dict and tail_train_dict[paper] != {}:
                    tail_train_number += 1
                    elements = [{"data":tail_train_dict[paper][target_id],"target_id":target_id,"source_id":paper,"th":"tail"} for target_id in list(tail_train_dict[paper].keys())]
                elif paper in head_train_dict and head_train_dict[paper] != {}:
                    head_train_number += 1
                    elements = [{"data":head_train_dict[paper][source_id],"target_id":paper,"source_id":source_id,"th":"head"} for source_id in list(head_train_dict[paper].keys())]
                else:
                    print("Unfound: "+paper)
                paper_dict[paper] = elements
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

def load_data_intent_identification_with_context(args,model,ent_vocab):
    if args.dataset == "AASC":
        #X,y,paper_dict = load_raw_data(args)
        X,y,paper_dict = load_raw_data_SYNTHCI_AASC(args)
        intent_identification_dir = settings.intent_identification_dir
    else:
        X,y,paper_dict = load_raw_data_PeerRead(args)
        intent_identification_dir = settings.intent_identification_PeerRead_dir
    converted_path = os.path.join(intent_identification_dir,"intent_identification.json")
    if os.path.exists(converted_path):
        with open(converted_path) as f:
            paper_dict = json.load(f)
    else:
        paper_dict = convert_data(paper_dict,ent_vocab,args.MAX_LEN,args.WINDOW_SIZE)
        with open(converted_path,"w") as f:
            json.dump(paper_dict,f)
    converted_path = os.path.join(intent_identification_dir,"intent_identification.binaryfile")
    if os.path.exists(converted_path):
        with open(converted_path,"rb") as f:
            paper_embeddings_dict = pickle.load(f)
    else:
        paper_embeddings_dict = get_embeddings(model,paper_dict,args.MAX_LEN,args.WINDOW_SIZE)
        with open(converted_path,"wb") as f:
            pickle.dump(paper_embeddings_dict,f)
    X = [(paper_embeddings_dict[target_id],paper_embeddings_dict[source_id]) for (target_id,source_id) in X]
    return X,y

def load_data_intent_identification_scibert(ent_vocab,matrix_train,matrix_test):
    f = open(os.path.join(settings.intent_identification_dir,"id2intent.txt"))
    X = []
    y = []
    intentdict = {}
    intentn = 0
    with torch.no_grad():
        model = BertModel.from_pretrained(settings.pretrained_scibert_path)
        tokenizer = BertTokenizer.from_pretrained(settings.pretrained_scibert_path)
        for i,line in enumerate(f):
            if i == 0:
                continue
            l = line.split("\t")
            target_id = l[0]
            source_id = l[1]
            intent = l[2]
            left_citated_text = l[3]
            right_citated_text = l[4]
            left_citation_tokenized = tokenizer.tokenize(left_citated_text)[-250:]
            right_citation_tokenized = tokenizer.tokenize(right_citated_text)[:250]
            input_tokens = tokenizer.convert_tokens_to_ids(left_citation_tokenized)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(right_citation_tokenized)
            position_citation_mark = len(left_citation_tokenized)
            tokens_tensor = torch.tensor([input_tokens])
            outputs = model(tokens_tensor)
            emb = np.array(outputs[0][0][position_citation_mark].cpu())
            if intent not in intentdict:
                intentdict[intent] = intentn
                intentn += 1
            X.append(emb)
            y.append(intentdict[intent])
    return X,y

def load_data_intent_identification_from_pkl():
    f = open(os.path.join(settings.intent_identification_dir,"id2intent.txt"))
    X = []
    y = []
    intentdict = {}
    intentn = 0
    paper_embeddings_dict = pickle.load(open(os.path.join(settings.citation_recommendation_dir,"AASC_embeddings.pkl"),"rb"))
    with torch.no_grad():
        for i,line in enumerate(f):
            if i == 0:
                continue
            l = line.split("\t")
            target_id = l[0]
            source_id = l[1]
            intent = l[2]
            if intent not in intentdict:
                intentdict[intent] = intentn
                intentn += 1
            X.append(np.concatenate([paper_embeddings_dict[target_id],paper_embeddings_dict[source_id]]))
            y.append(intentdict[intent])
    return X,y

if __name__ == "__main__":
    X,y = load_data_intent_identification_from_pkl()
    tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
    X_visualization = tsne.fit_transform(X)
    print("PCA done: " + str(len(X)))
    print("Y length: " + str(len(y)))
    print("Y distribution")
    print(collections.Counter(y))
    print("visualization start")
    fig, ax = pyplot.subplots(figsize=(20,20))
    X_colors = [[] for _ in range(max(y)+1)]
    y_colors = [[] for _ in range(max(y)+1)]
    colors_name = ["black","grey","tomato","saddlebrown","palegoldenrod","olivedrab","cyan","steelblue","midnightblue","darkviolet","magenta","pink","yellow"]
    for x1,y1 in zip(X_visualization,y):
        X_colors[y1].append(x1)
        y_colors[y1].append(y1)
    for X_color,color in zip(X_colors,colors_name[:len(y_colors)]):
        X_color_x = np.array([X_place[0] for X_place in X_color])
        X_color_y = np.array([X_place[1] for X_place in X_color])
        ax.scatter(X_color_x,X_color_y,c=color)
    pyplot.savefig("PTBCN_AASC_intent_epoch.png") # 保存
    l = [i for i in range(len(X))]
    random.seed(10)
    random.shuffle(l)
    dict1 = {}
    for epoch in range(5):
        if epoch == 0:
            X_test = [X[i] for i in l[:len(l)//5]]
            y_test = [y[i] for i in l[:len(l)//5]]
            X_train = [X[i] for i in l[len(l)//5:]]
            y_train = [y[i] for i in l[len(l)//5:]]
        elif epoch == 4:
            X_test = [X[i] for i in l[len(l)*epoch//5:]]
            y_test = [y[i] for i in l[len(l)*epoch//5:]]
            X_train = [X[i] for i in l[:len(l)*epoch//5]]
            y_train = [y[i] for i in l[:len(l)*epoch//5]]
        else:
            X_test = [X[i] for i in l[len(l)*epoch//5:len(l)*(epoch+1)//5]]
            y_test = [y[i] for i in l[len(l)*epoch//5:len(l)*(epoch+1)//5]]
            X_train = [X[i] for i in l[:len(l)*epoch//5]+l[len(l)*(epoch+1)//5:]]
            y_train = [y[i] for i in l[:len(l)*epoch//5]+l[len(l)*(epoch+1)//5:]]
        print("training start")
        Cs = [2, 2**5, 2 **10]
        gammas = [2 ** -6, 2** -3,2**-1, 2**1,2 ** 3, 2 ** 6]
        svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
        products = [(C,gamma) for C,gamma in product(Cs,gammas)]
        print("training done")
        for sv,product1 in zip(svs,products):
            test_label = sv.predict(X_test)
            s1 = "C:"+str(product1[0])+","+"gamma:"+str(product1[1])
            if s1 not in dict1:
                dict1[s1] = {"正解率":accuracy_score(y_test, test_label),"マクロ平均":f1_score(y_test, test_label,average="macro"),"ミクロ平均":f1_score(y_test, test_label,average="micro"),"分類結果":collections.Counter(test_label)}
            else:
                dict1[s1]["正解率"] += accuracy_score(y_test, test_label)
                dict1[s1]["マクロ平均"] += f1_score(y_test, test_label,average="macro")
                dict1[s1]["ミクロ平均"] += f1_score(y_test, test_label,average="micro")
                for key in dict1[s1]["分類結果"]:
                    dict1[s1]["分類結果"][key] += collections.Counter(test_label)[key]
    for sv,product1 in zip(svs,products):
        s1 = "C:"+str(product1[0])+","+"gamma:"+str(product1[1])
        print("正解率＝", dict1[s1]["正解率"]/5)
        print("マクロ平均＝", dict1[s1]["マクロ平均"]/5)
        print("ミクロ平均＝", dict1[s1]["ミクロ平均"]/5)
        print(dict1[s1]["分類結果"])
