import os
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForMaskedLM

import fitlog

sys.path.append('../')
from model import PTBCN
from metrics import MacroMetric
from metrics import Evaluation
from utils import build_ent_vocab
from dataloader import load_AASC_graph_data,load_PeerRead_graph_data,load_data_SVM, load_data_intent_identification
from collections import Counter
from itertools import product
import collections
from tqdm import tqdm
import random
import pickle

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score

import pandas as pd
import csv



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC',help="data directory path")
    parser.add_argument('--dataset', type=str, default='AASC',help="AASC or PeerRead")
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help="fitlog directory path")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--frequency', type=int, default=5, help="frequency to remove rare entity")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--beta', type=float, default=0.999, help="beta_2 of adam")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--warm_up', type=float, default=0.1, help="warmup proportion or steps")
    parser.add_argument('--epoch', type=int, default=20, help="number of epochs")
    parser.add_argument('--grad_accumulation', type=int, default=1, help="gradient accumulation")
    parser.add_argument('--gpu', type=str, default='all', help="run script on which devices")
    parser.add_argument('--debug', action='store_true', help="do not log")
    parser.add_argument('--model_path', type=str, default="../model/",
                        help="the path of directory containing model and entity embeddings.")
    parser.add_argument('--WINDOW_SIZE', type=int, default=125, help="the length of context length")
    parser.add_argument('--MAX_LEN', type=int, default=256, help="MAX length of the input")
    parser.add_argument('--train', type=bool, default=True, help="train or not")
    parser.add_argument('--predict', type=bool, default=True, help="predict or not")
    parser.add_argument('--node_classification', type=bool, default=True, help="conduct node classification or not")
    parser.add_argument('--pretrained_model', type=str, default="scibert", help="scibert or bert")
    return parser.parse_args()

def predict(args,epoch,model,ent_vocab,test_set):
    testloader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=os.cpu_count()//2)
    test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=args.batch_size, sampler=None,num_workers=os.cpu_count()//2,collate_fn=test_set.collate_fn)
    mrr_all = 0
    Recallat5_all = 0
    Recallat10_all = 0
    Recallat30_all = 0
    Recallat50_all = 0
    MAP_all = 0
    l_all = 0
    l_prev = 0
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_randomMASK.txt","w")
    with torch.no_grad():
        for (inputs,labels) in test_data_iter:
            outputs = model(input_ids=inputs["input_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),masked_lm_labels=inputs["masked_lm_labels"].cuda(),attention_mask=inputs["attention_mask"].cuda())
            MAP,mrr,Recallat5,Recallat10,Recallat30,Recallat50,l = Evaluation(outputs["entity_logits"].cpu(),inputs["masked_lm_labels"])
            mrr_all += mrr
            Recallat5_all += Recallat5
            Recallat10_all += Recallat10
            Recallat30_all += Recallat30
            Recallat50_all += Recallat50
            MAP_all += MAP
            l_all += l
            if l_all - l_prev > 500:
                l_prev = l_all
                print(l_all)
                print(mrr_all/l_all)
        fw.write("MRR\n")
        fw.write(str(mrr_all/l_all)+"\n")
        fw.write("Recallat5\n")
        fw.write(str(Recallat5_all/l_all)+"\n")
        fw.write("Recallat10\n")
        fw.write(str(Recallat10_all/l_all)+"\n")
        fw.write("Recallat30\n")
        fw.write(str(Recallat30_all/l_all)+"\n")
        fw.write("Recallat50\n")
        fw.write(str(Recallat50_all/l_all)+"\n")
        fw.write("MAP\n")
        fw.write(str(MAP_all/l_all)+"\n")
        print("MRR")
        print(mrr_all/l_all)
        print("Recallat5")
        print(Recallat5_all/l_all)
        print("Recallat10")
        print(Recallat10_all/l_all)
        print("Recallat30")
        print(Recallat30_all/l_all)
        print("Recallat50")
        print(Recallat50_all/l_all)
        print("MAP")
        print(MAP_all/l_all)

def node_classification(args,epoch,model,ent_vocab):
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_nodeclassification_randomMASK.txt","w")
    X_train,y_train,X_test,y_test = load_data_SVM(model,ent_vocab)
    print("SVM data load done")
    print("training start")
    Cs = [2 , 2**5, 2 **10]
    gammas = [2 ** -9, 2 ** -6, 2** -3,2 ** 3, 2 ** 6, 2 ** 9]
    svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
    products = [(C,gamma) for C,gamma in product(Cs,gammas)]
    print("training done")
    for sv,product1 in zip(svs,products):
        test_label = sv.predict(X_test)
        fw.write("C:"+str(product1[0])+","+"gamma:"+str(product1[1])+"\n")
        fw.write("正解率="+str(accuracy_score(y_test, test_label))+"\n")
        fw.write("マクロ平均="+str(f1_score(y_test, test_label,average="macro"))+"\n")
        fw.write("ミクロ平均="+str(f1_score(y_test, test_label,average="micro"))+"\n")
        fw.write(str(collections.Counter(test_label))+"\n")
        print("正解率＝", accuracy_score(y_test, test_label))
        print("マクロ平均＝", f1_score(y_test, test_label,average="macro"))
        print("ミクロ平均＝", f1_score(y_test, test_label,average="micro"))
        print(collections.Counter(test_label))

def intent_identification(args,epoch,model,ent_vocab):
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_intentidentification_randomMASK.txt","w")
    X,y = load_data_intent_identification(model,ent_vocab)
    print("intent identification data load done")
    l = [i for i in range(len(X))]
    random.shuffle(l)
    for epoch in range(5):
        if epoch == 0:
            X_train = [X[i] for i in l[:len(l)//5]]
            y_train = [y[i] for i in l[:len(l)//5]]
            X_test = [X[i] for i in l[len(l)//5:]]
            y_test = [y[i] for i in l[len(l)//5:]]
        elif epoch == 4:
            X_train = [X[i] for i in l[len(l)*epoch//5:]]
            y_train = [y[i] for i in l[len(l)*epoch//5:]]
            X_test = [X[i] for i in l[:len(l)*epoch//5]]
            y_test = [y[i] for i in l[:len(l)*epoch//5]]
        else:
            X_train = [X[i] for i in l[len(l)*epoch//5:len(l)*(epoch+1)//5]]
            y_train = [y[i] for i in l[len(l)*epoch//5:len(l)*(epoch+1)//5]]
            X_test = [X[i] for i in l[:len(l)*epoch//5]+l[len(l)*(epoch+1)//5:]]
            y_test = [y[i] for i in l[:len(l)*epoch//5]+l[len(l)*(epoch+1)//5:]]
        print("training start")
        Cs = [2 , 2**5, 2 **10]
        gammas = [2 ** -9, 2 ** -6, 2** -3,2 ** 3, 2 ** 6, 2 ** 9]
        svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
        products = [(C,gamma) for C,gamma in product(Cs,gammas)]
        print("training done")
        for sv,product1 in zip(svs,products):
            test_label = sv.predict(X_test)
            fw.write("C:"+str(product1[0])+","+"gamma:"+str(product1[1])+"\n")
            fw.write("正解率="+str(accuracy_score(y_test, test_label))+"\n")
            fw.write("マクロ平均="+str(f1_score(y_test, test_label,average="macro"))+"\n")
            fw.write("ミクロ平均="+str(f1_score(y_test, test_label,average="micro"))+"\n")
            fw.write(str(collections.Counter(test_label))+"\n")
            print("正解率＝", accuracy_score(y_test, test_label))
            print("マクロ平均＝", f1_score(y_test, test_label,average="macro"))
            print("ミクロ平均＝", f1_score(y_test, test_label,average="micro"))
            print(collections.Counter(test_label))

def get_embeddings(model,ent_vocab,epoch):
    entityl = []
    print("get embeddings start")
    with torch.no_grad():
        for i,key in enumerate(ent_vocab):
            if i > 10:
                break
            if i % 10000 == 0:
                print(i)
            input_ids = []
            position_ids = []
            token_type_ids = []
            masked_lm_labels = []
            attention_mask = []
            input_ids.append([ent_vocab[key]]+[-1]*255)
            position_ids.append([i for i in range(256)])
            token_type_ids.append([1]+[0]*255)
            masked_lm_labels.append([-1 for i in range(256)])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(255,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(256,255,dtype=torch.int)),dim=1)
            attention_mask.append(adj)
            outputs = model(input_ids=torch.tensor(input_ids).cuda(),attention_mask=torch.stack(attention_mask,dim=0).cuda(),token_type_ids=torch.tensor(token_type_ids).cuda(),position_ids=torch.tensor(position_ids).cuda(),masked_lm_labels=torch.tensor(masked_lm_labels).cuda())
            entity_logits = outputs["sequence_output"][0][0]
            entityl.append({key:np.array(entity_logits.cpu())})
        pathentity = "/home/ohagi_masaya/TransBasedCitEmb/model/"+"AASC"+"epoch"+str(epoch)+".pickle"
        f = open(pathentity,"wb")
        pickle.dump(entityl,f)

def collate_fn(batch):
    input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask"]
    target_keys = ["masked_lm_labels","word_seq_len"]
    max_words = self.MAX_LEN
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
            adj = torch.cat((adj,torch.zeros(self.MAX_LEN,word_pad,dtype=torch.int)),dim=1)
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

def main():
    args = parse_args()

    if args.debug:
        fitlog.debug()
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #load entity embeddings
    #TODO 初期化をSPECTERで行う
    train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir,args.frequency,args.WINDOW_SIZE,args.MAX_LEN,args.pretrained_model)
    num_ent = len(ent_vocab)

    # load parameters
    if args.pretrained_model == "scibert":
        model = PTBCN.from_pretrained('../pretrainedmodel/scibert_scivocab_uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    else:
        model = PTBCN.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    model.change_type_embeddings()
    print('parameters of SciBERT has been loaded.')

    # fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    devices = list(range(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("GPU NO")

    if args.data_dir[-1] == "/":
        data_dir_modelname = os.path.basename(args.data_dir[:-1])
    else:
        data_dir_modelname = os.path.basename(args.data_dir)
    model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+".bin"
    pretrained_model_path = os.path.join(args.model_path,model_name)
    print("train start")
    for epoch in range(1,args.epoch+1):
        train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir,args.frequency,args.WINDOW_SIZE,args.MAX_LEN,args.pretrained_model)
        #train iter
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = 2, collate_fn=collate_fn)
        for (inputs,labels) in train_dataloader:
            optimizer.zero_grad()
            outputs = model(input_ids=inputs["input_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),masked_lm_labels=inputs["masked_lm_labels"].cuda(),attention_mask=inputs["attention_mask"].cuda())
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            #train iter
            predict(args,epoch,model,ent_vocab,test_set)
            node_classification(args,epoch,model,ent_vocab)
            intent_identification(args,epoch,model,ent_vocab) 
            #save model
            model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_randomMASK.bin"
            torch.save(model.state_dict(),os.path.join(args.model_path,model_name))
            get_embeddings(model,ent_vocab,epoch)
    print("train end")

if __name__ == '__main__':
    main()
