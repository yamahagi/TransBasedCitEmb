import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertModel
from dataloader_CoKE import load_AASC_graph_data,load_PeerRead_graph_data,make_matrix
from load_intent_identification import load_data_intent_identification_scibert

from metrics import Evaluation

from load_node_classification import load_data_SVM_COKE

sys.path.append('../')
from model import PTBCNCOKE
from collections import Counter
from itertools import product
import collections
from tqdm import tqdm

import pandas as pd
import csv

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score
from sklearn.decomposition import PCA
from matplotlib import pyplot
import settings


#pathはsettings.pyで管理
#それ以外のhyper parameterをargsで管理
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AASC',help="AASC or PeerRead")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--frequency', type=int, default=5, help="frequency to remove rare entity")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--MAX_LEN', type=int, default=3, help="MAX length of the input")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="scibert", help="scibert or bert")
    parser.add_argument('--train_data', type=str, default="excluded", help="remove low frequency data or not [excluded,full] during training")
    parser.add_argument('--test_data', type=str, default="excluded", help="remove low frequency data or not [excluded,full]")
    return parser.parse_args()

#test evaluation for citation recommendation
def predict(args,epoch,model,ent_vocab,test_set):
    matrix = make_matrix(ent_vocab)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 2, collate_fn=Collate_fn(args.MAX_LEN,matrix).collate_fn)
    mrr_all = 0
    Recallat5_all = Recallat10_all = Recallat30_all = Recallat50_all = 0
    MAP_all = 0
    l_all = 0
    l_prev = 0
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"pretrained_model"+str(args.pretrained_model)+"_COKE.txt","w")
    with torch.no_grad():
        for (inputs,labels) in test_dataloader:
            outputs = model(target_ids=inputs["target_ids"].cuda(),source_ids=inputs["source_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),contexts=inputs["contexts"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),attention_mask=inputs["attention_mask"].cuda(),mask_positions=inputs["mask_positions"].cuda())
            masked_lm_labels = []
            for mask_position,target_id,source_id in zip(inputs["mask_positions"],inputs["target_ids"],inputs["source_ids"]):
                if mask_position == 0:
                    masked_lm_labels.append([target_id,-1,-1])
                else:
                    masked_lm_labels.append([-1,-1,source_id])
            MAP,mrr,Recallat5,Recallat10,Recallat30,Recallat50,l = Evaluation(outputs["entity_logits"].cpu(),masked_lm_labels)
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
        s = ""
        s += "MRR\n" + str(mrr_all/l_all)+"\n"
        s += "Recallat5\n" + str(Recallat5_all/l_all)+"\n"
        s += "Recallat10\n" + str(Recallat10_all/l_all)+"\n"
        s += "Recallat30\n" + str(Recallat30_all/l_all)+"\n"
        s += "MAP\n" + str(MAP_all/l_all)+"\n"
        fw.write(s)
        print(s)

def node_classification(args,epoch,model,ent_vocab):
    X_train,y_train,X_test,y_test = load_data_SVM_COKE(model,ent_vocab)
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
    pyplot.savefig("images/TransBasedCitEmb_COKE.png") # 保存
    Cs = [2 , 2**5, 2 **10]
    gammas = [2 ** -9, 2 ** -6, 2** -3,2 ** 3, 2 ** 6, 2 ** 9]
    svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
    products = [(C,gamma) for C,gamma in product(Cs,gammas)]
    print("training done")
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"pretrained_model"+str(args.pretrained_model)+"_nodeclassification_COKE.txt","w")
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

class Collate_fn():
    def __init__(self,MAX_LEN,matrix):
        self.max_words = MAX_LEN
        self.matrix = matrix
    def collate_fn(self, batch):
        input_keys = ['target_ids','source_ids',"position_ids","contexts","token_type_ids","attention_mask","mask_positions"]
        target_keys = ["target_ids","source_ids"]
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}

        for sample in batch:
            batch_x["position_ids"].append([0,1,2])
            batch_x["token_type_ids"].append([1,0,1])
            batch_x["target_ids"].append(sample["target_id"])
            batch_x["source_ids"].append(sample["source_id"])
            batch_x["contexts"].append(self.matrix[sample["target_id"]][sample["source_id"]])
            batch_x["mask_positions"].append(sample["MASK_position"])
            adj = torch.ones(3,3,dtype=torch.int)
            batch_x["attention_mask"].append(adj)
            batch_y["target_ids"].append(sample["target_id"])
            batch_y["source_ids"].append(sample["source_id"])
        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)

def plot_intent_identification(ent_vocab,matrix_train,matrix_test):
    X,y = load_data_intent_identification_scibert(ent_vocab,matrix_train,matrix_test)
    pca = PCA(n_components=2)
    pca.fit(X)
    X_visualization = pca.transform(X)
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
    pyplot.savefig("images/TransBasedCitEmb_intent_identification_scibert.png")


def main():
    args = parse_args()

    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("GPU NO")
    
    if args.dataset == "AASC":
        #train_set, test_set, ent_vocab = load_AASC_graph_data(args)
        train_set, test_set, ent_vocab,matrix_train,matrix_test = load_AASC_graph_data(args)
    else:
        train_set, test_set, ent_vocab = load_PeerRead_graph_data(args)
    plot_intent_identification(ent_vocab,matrix_train,matrix_test)
    sys.exit()
    num_ent = len(ent_vocab)
    if args.pretrained_model == "scibert":
        model = PTBCNCOKE.from_pretrained(settings.pretrained_scibert_path,num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    else:
        model = PTBCNCOKE.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    model.change_type_embeddings()
    model.cuda()
    
    #fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_pretrainedmodel"+str(args.pretrained_model)+"_COKE.bin"
    pretrained_model_path = os.path.join(settings.model_path,model_name)
    print("train start")
    if args.train:
        matrix = make_matrix(ent_vocab)
        for epoch in range(1,args.epoch+1):
            train_set, test_set, ent_vocab = load_AASC_graph_data(args)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = 2, collate_fn=Collate_fn(args.MAX_LEN,matrix).collate_fn)
            print("epoch: "+str(epoch))
            with tqdm(train_dataloader) as pbar:
                for i,(inputs,labels) in enumerate(pbar):
                    optimizer.zero_grad()
                    outputs = model(target_ids=inputs["target_ids"].cuda(),source_ids=inputs["source_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),contexts=inputs["contexts"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),attention_mask=inputs["attention_mask"].cuda(),mask_positions=inputs["mask_positions"].cuda())
                    loss = outputs["loss"]
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(collections.OrderedDict(loss=loss.detach().cpu().numpy()))
            if epoch % 25 == 0:
                #save model
                model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_pretrainedmodel"+str(args.pretrained_model)+"_COKE.bin"
                torch.save(model.state_dict(),os.path.join(settings.model_path,model_name))
    print("train end")
    if args.predict:
        for i in range(1,5):
            epoch = i*25
            model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_pretrainedmodel"+str(args.pretrained_model)+"_COKE.bin"
            model.load_state_dict(torch.load(os.path.join(settings.model_path,model_name)))
            model.eval()
            predict(args,epoch,model,ent_vocab,test_set)
            node_classification(args,epoch,model,ent_vocab)


if __name__ == '__main__':
    main()
