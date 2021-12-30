import os
import sys
from collections import OrderedDict

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForMaskedLM, BertModel
import fitlog
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json

sys.path.append('../')
from model import PTBCN
from metrics import Evaluation
from utils import build_ent_vocab
from dataloader import load_AASC_graph_data,load_PeerRead_graph_data
from collections import Counter
from itertools import product
import collections
from tqdm import tqdm
import random
import pickle
from collections import Counter

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score
from sklearn.decomposition import PCA
from matplotlib import pyplot

import pandas as pd
import csv
from transformers import BertForSequenceClassification, AdamW, BertConfig
import math
import settings

from collections import defaultdict
from sklearn.model_selection import KFold

from sklearn.metrics import classification_report

#あまりにcitation intent identificationがうまくいかないので元データを確かめてみようという試み
#とりあえず元データに対するscibertを用いたtrainとそれに対する評価を行う
#また、元データをunsupervisedなplotを行うことでAASCのplotがうまくいかないのはそもそものデータの問題なのかそれかcitation intent identificationにplotが向いていないのかの再確認を行う
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC',help="data directory path")
    parser.add_argument('--dataset', type=str, default='AASC',help="AASC or PeerRead")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--frequency', type=int, default=5, help="frequency to remove rare entity")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--beta', type=float, default=0.999, help="beta_2 of adam")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--warm_up', type=float, default=0.1, help="warmup proportion or steps")
    parser.add_argument('--epoch', type=int, default=15, help="number of epochs")
    parser.add_argument('--debug', action='store_true', help="do not log")
    parser.add_argument('--model_path', type=str, default="../model/",
                        help="the path of directory containing model and entity embeddings.")
    parser.add_argument('--pretrained_model', type=str, default="scibert", help="scibert or bert")
    return parser.parse_args()


def load_data_intent_identification(textdict):
    intentn = -1
    intentdict = {}
    tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
    f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/train.jsonl")
    X_AASC_input_ids = []
    X_AASC_attention_masks = []
    y_AASC = []
    X_train_input_ids = []
    X_train_attention_masks = []
    y_train = []
    with torch.no_grad():
        jsl = []
        for i,line in enumerate(f):
            js = json.loads(line)
            jsl.append(js)
        for js in jsl:
            if i == 0:
                continue
            target_id = js["citing_paper_id"]
            source_id = js["cited_paper_id"]
            intent = js["intent"]
            text = js["text"]
            if intent not in intentdict:
                intentn += 1
                intentdict[intent] = intentn
            encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True, # Special Tokenの追加
                        max_length = 512,           # 文章の長さを固定（Padding/Trancatinating）
                        pad_to_max_length = True,# PADDINGで埋める
                        return_attention_mask = True,   # Attention maksの作成
                        return_tensors = 'pt',     #  Pytorch tensorsで返す
                   )
            if text in textdict and (textdict[text][0] == target_id and textdict[text][1] == source_id):
                X_AASC_input_ids.append(encoded_dict['input_ids'])
                X_AASC_attention_masks.append(encoded_dict['attention_mask'])
                y_AASC.append(intentdict[intent])
            else:
                X_train_input_ids.append(encoded_dict['input_ids'])
                X_train_attention_masks.append(encoded_dict['attention_mask'])
                y_train.append(intentdict[intent])
    X_train_input_ids = torch.cat(X_train_input_ids,dim=0)
    X_train_attention_masks = torch.cat(X_train_attention_masks,dim=0)
    f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/test.jsonl")
    X_test_input_ids = []
    X_test_attention_masks = []
    y_test = []
    with torch.no_grad():
        jsl = []
        for i,line in enumerate(f):
            js = json.loads(line)
            jsl.append(js)
        for js in jsl:
            if i == 0:
                continue
            target_id = js["citing_paper_id"]
            source_id = js["cited_paper_id"]
            intent = js["intent"]
            text = js["text"]
            if intent not in intentdict:
                intentn += 1
                intentdict[intent] = intentn
            encoded_dict = tokenizer.encode_plus(
                        text,
                        add_special_tokens = True, # Special Tokenの追加
                        max_length = 512,           # 文章の長さを固定（Padding/Trancatinating）
                        pad_to_max_length = True,# PADDINGで埋める
                        return_attention_mask = True,   # Attention maksの作成
                        return_tensors = 'pt',     #  Pytorch tensorsで返す
                   )
            if text in textdict and (textdict[text][0] == target_id and textdict[text][1] == source_id):
                X_AASC_input_ids.append(encoded_dict['input_ids'])
                X_AASC_attention_masks.append(encoded_dict['attention_mask'])
                y_AASC.append(intentdict[intent])
            else:
                X_test_input_ids.append(encoded_dict['input_ids'])
                X_test_attention_masks.append(encoded_dict['attention_mask'])
                y_test.append(intentdict[intent])
    X_test_input_ids = torch.cat(X_test_input_ids,dim=0)
    X_test_attention_masks = torch.cat(X_test_attention_masks,dim=0)
    X_AASC_input_ids = torch.cat(X_AASC_input_ids,dim=0)
    X_AASC_attention_masks = torch.cat(X_AASC_attention_masks,dim=0)
    return X_train_input_ids,X_train_attention_masks,y_train,X_test_input_ids,X_test_attention_masks,y_test,X_AASC_input_ids,X_AASC_attention_masks,y_AASC

def load_data_intent_identification():
    intentn = -1
    intentdict = {}
    tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
    f_train = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/train.jsonl")
    f_test = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/test.jsonl")
    X_input_ids = []
    X_attention_masks = []
    y = []
    with torch.no_grad():
        for f in [f_train,f_test]:
            jsl = []
            for i,line in enumerate(f):
                js = json.loads(line)
                jsl.append(js)
            for i,js in enumerate(jsl):
                if i == 0:
                    print(js.keys())
                target_id = js["citing_paper_id"]
                source_id = js["cited_paper_id"]
                intent = js["intent"]
                text = js["text"]
                if intent not in intentdict:
                    intentn += 1
                    intentdict[intent] = intentn
                encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True, # Special Tokenの追加
                            max_length = 512,           # 文章の長さを固定（Padding/Trancatinating）
                            pad_to_max_length = True,# PADDINGで埋める
                            return_attention_mask = True,   # Attention maksの作成
                            return_tensors = 'pt',     #  Pytorch tensorsで返す
                       )
                X_input_ids.append(encoded_dict['input_ids'])
                X_attention_masks.append(encoded_dict['attention_mask'])
                y.append(intentdict[intent])
    X_input_ids = torch.cat(X_input_ids,dim=0)
    X_attention_masks = torch.cat(X_attention_masks,dim=0)
    return X_input_ids,X_attention_masks,y,intentdict

def load_data_intent_identification_AASC():
    intentn = -1
    intentdict = {}
    textdict = {}
    f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/id2intent.txt")
    X = []
    y = []
    paperdict = defaultdict(dict)
    for i,line in enumerate(f):
        if i == 0:
            continue
        l = line.rstrip("\n").split("\t")
        target_id = l[0]
        source_id = l[1]
        intent = l[2]
        text = l[3]
        if intent not in intentdict:
            intentn += 1
            intentdict[intent] = intentn
        X.append({"target_id":target_id,"source_id":source_id,"text":text})
        paperdict[target_id][source_id] = intent
        textdict[text] = (target_id,source_id)
        y.append(intentdict[intent])
    return X,y,textdict

def load_data_intent_identification_for_tagging():
    tokenizer =  BertTokenizer.from_pretrained(settings.pretrained_scibert_path, do_lower_case =False)
    f_train = "/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train_intent.csv"
    f_test = "/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/test_intent.csv"
    X_input_ids = []
    X_attention_masks = []
    X_paper_ids = []
    print("---data loading---")
    with torch.no_grad():
        for f in [f_train,f_test]:
            df = pd.read_csv(f)
            for i,(target_id,context,source_id) in enumerate(zip(df["target_id"],df["context"],df["source_id"])):
                citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))[:256]
                input_ids = citation_tokenized
                word_pad = 256-len(input_ids)
                adj = torch.ones(len(input_ids), len(input_ids), dtype=torch.int)
                adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
                adj = torch.cat((adj,torch.zeros(256,word_pad,dtype=torch.int)),dim=1)
                input_ids += [0]*word_pad
                X_input_ids.append(torch.tensor(input_ids))
                X_attention_masks.append(adj)
                X_paper_ids.append((target_id,source_id))
                if i % 1000 == 0:
                    print(i)
    return X_input_ids,X_attention_masks,X_paper_ids

def plot_intent_identification(X_AASC_input_ids,X_AASC_attention_masks,y_AASC,model):
    X = []
    y = []
    i = 0
    with torch.no_grad():
        """
        for input_ids,attention_masks,label in zip(X_train_input_ids,X_train_attention_masks,y_train):
            input_ids = input_ids.unsqueeze(0)
            attention_masks = attention_masks.unsqueeze(0)
            y.append(label)
            label = torch.tensor([label])
            outputs = model(input_ids=input_ids.cuda(),attention_mask=attention_masks.cuda(),labels=label.cuda())
            X.append(np.array(outputs["hidden_states"][-1][0][0].cpu()))
        """
        for input_ids,attention_masks,label in zip(X_AASC_input_ids,X_AASC_attention_masks,y_AASC):
            input_ids = input_ids.unsqueeze(0)
            attention_masks = attention_masks.unsqueeze(0)
            y.append(label)
            label = torch.tensor([label])
            outputs = model(input_ids=input_ids.cuda(),attention_mask=attention_masks.cuda(),labels=label.cuda())
            X.append(np.array(outputs["hidden_states"][-1][0][0].cpu()))
        print("intent identification data load done")
        print("PCA start")
        pca = PCA(n_components=2)
        pca.fit(X)
        X_visualization = pca.transform(X)
        print(X_visualization[:3])
        print("PCA done: " + str(len(X)))
        print(len(X_visualization))
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
        pyplot.savefig("images/TransBasedCitEmb_intent_identification_scibert_AASC_before.png") # 保存


class ACLARCDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, label):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label = torch.tensor(label)
        self.datanum = len(self.input_ids)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        out_input_ids = self.input_ids[idx]
        out_attention_masks = self.attention_masks[idx]
        out_label = self.label[idx]
        return out_input_ids,out_attention_masks,out_label

def annotation(model):
    X_input_ids,X_attention_masks,X_paper_ids = load_data_intent_identification_for_tagging()
    model.load_state_dict(torch.load(os.path.join(settings.model_path,"scibert_intent_identification_epoch15.bin")))
    model.cuda()
    model.eval()
    print("---started annotation---")
    f = open(os.path.join(settings.citation_recommendation_dir,"intent_annotated_AASC.tsv"),"w")
    f.write("target_id\tsource_id\tintent\n")
    with torch.no_grad():
        for i,(input_ids,attention_masks,paper_ids) in enumerate(zip(X_input_ids,X_attention_masks,X_paper_ids)):
            target_id = paper_ids[0]
            source_id = paper_ids[1]
            outputs = model(input_ids=input_ids.unsqueeze(0).cuda(),
                            attention_mask=attention_masks.unsqueeze(0).cuda())
            logits = outputs["logits"]
            logits = logits.detach().cpu().numpy()
            pred =  list(np.argmax(logits, axis=1))[0]
            f.write(target_id+"\t"+source_id+"\t"+str(pred)+"\n")
            if i % 1000 == 0:
                print(i)

def calculate_accuracy(intentdict):
    f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/id2intent.txt")
    df = pd.read_csv("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/intent_annotated_AASC.tsv",delimiter="\t")
    fw = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/intent_annotated_AASC_corrected.tsv","w")
    fw.write("target_id\tsource_id\tintent\n")
    paperdict = defaultdict(dict)
    predl = []
    ansl = []
    for target_id,source_id,intent in zip(df["target_id"],df["source_id"],df["intent"]):
        paperdict[target_id][source_id] = int(intent)
    for i,line in enumerate(f):
        if i == 0:
            continue
        l = line[:-1].split("\t")
        target_id = l[0]
        source_id = l[1]
        intent = l[2]
        predl.append(paperdict[target_id][source_id])
        ansl.append(intentdict[intent])
        paperdict[target_id][source_id] = int(intentdict[intent])
    print(classification_report(ansl,predl))
    for target_id in paperdict:
        for source_id in paperdict[target_id]:
            fw.write(target_id+"\t"+source_id+"\t"+str(paperdict[target_id][source_id])+"\n")



def main():
    args = parse_args()
    #X,y,textdict = load_data_intent_identification_AASC()
    #X_train_input_ids,X_train_attention_masks,y_train,X_test_input_ids,X_test_attention_masks,y_test,X_AASC_input_ids,X_AASC_attention_masks,y_AASC = load_data_intent_identification(textdict)
    X_input_ids,X_attention_masks,y,intentdict = load_data_intent_identification()
    print(intentdict)
    #calculate_accuracy(intentdict)
    train_set = ACLARCDataset(X_input_ids,X_attention_masks,y)
    num_label = max(y)+1
    model = BertForSequenceClassification.from_pretrained(
        "../pretrainedmodel/scibert_scivocab_uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = num_label, # The number of output labels--2 for binary classification.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = True, # Whether the model returns all hidden-states.
    )
    optimizer = AdamW(model.parameters(),
                  lr = 5e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = os.cpu_count()//2)
    model.cuda()
    model.train()
    """
    for epoch_i in range(1, args.epoch+1):
        total_train_loss = 0
        print("epoch: "+str(epoch_i))
        print(len(train_dataloader))
        for step, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            b_input_ids = batch[0].cuda()
            b_attention_masks = batch[1].cuda()
            b_labels = batch[2].cuda()
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_attention_masks,
                             labels=b_labels)
            loss = outputs["loss"]
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        if epoch_i % 5 == 0:
            print(avg_train_loss)
            torch.save(model.state_dict(),os.path.join(settings.model_path,"scibert_intent_identification_epoch"+str(epoch_i)+"_1.bin"))
    annotation(model)
    """
    calculate_accuracy(intentdict)
    """
    macro_f1_score = 0.0
    micro_f1_score = 0.0
    kf = KFold(n_splits=5)
    kf.get_n_splits(X_input_ids)
    for i,(train_index, test_index) in enumerate(kf.split(X_input_ids)):
        print(i)
        X_train_input_ids = torch.cat([X_input_ids[index] for index in train_index],dim=0)
        X_train_attention_masks = torch.cat([X_attention_masks[index] for index in train_index],dim=0)
        y_train = [y[index] for index in train_index]
        X_test_input_ids = torch.cat([X_input_ids[index] for index in test_index],dim=0)
        X_test_attention_masks = torch.cat([X_attention_masks[index] for index in test_index],dim=0)
        y_test = [y[index] for index in test_index]
        train_set = ACLARCDataset(X_train_input_ids,X_train_attention_masks,y_train)
        test_set = ACLARCDataset(X_test_input_ids,X_test_attention_masks,y_test)
        print("distributions of intent")
        print(collections.Counter(y_train))
        print(collections.Counter(y_test))
        num_label = max(y_train+y_test)+1
        model = BertForSequenceClassification.from_pretrained(
            "../pretrainedmodel/scibert_scivocab_uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = num_label, # The number of output labels--2 for binary classification.
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = True, # Whether the model returns all hidden-states.
        )
        optimizer = AdamW(model.parameters(),
                      lr = 5e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = os.cpu_count()//2)
        model.cuda()
        model.train()
        for epoch_i in range(0, args.epoch):
            total_train_loss = 0
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                b_input_ids = batch[0].cuda()
                b_attention_masks = batch[1].cuda()
                b_labels = batch[2].cuda()
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_attention_masks,
                                 labels=b_labels)
                loss = outputs["loss"]
                total_train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            avg_train_loss = total_train_loss / len(train_dataloader)
            if epoch_i % 5 == 0:
                print(avg_train_loss)
        total_eval_accuracy = 0
        total_eval_loss = 0
        pred = []
        seikail = []
        model.eval()
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = args.batch_size, shuffle = False, num_workers = os.cpu_count()//2)
        with torch.no_grad():
            for batch in test_dataloader:
                b_input_ids = batch[0].cuda()
                b_attention_masks = batch[1].cuda()
                b_labels = batch[2].cuda()
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_attention_masks,
                                 labels=b_labels)
                loss = outputs["loss"]
                logits = outputs["logits"]
                total_eval_loss += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                pred += list(np.argmax(logits, axis=1))
                seikail += list(label_ids)
        macro_f1_score += f1_score(seikail, pred, average='macro')
        micro_f1_score += f1_score(seikail, pred, average='micro')
    print(macro_f1_score/5)
    print(micro_f1_score/5)
    """


if __name__ == '__main__':
    main()
