import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import BertConfig, BertTokenizer, BertForMaskedLM

sys.path.append('../')
from model import PTBCN
from metrics import Evaluation
from utils import build_ent_vocab
from dataloader_without_citing import load_AASC_graph_data,load_PeerRead_graph_data
from load_node_classification import load_data_SVM_with_context,load_data_SVM_from_feedforward,load_data_SVM_with_context_all_layer
from itertools import product
import collections
from tqdm import tqdm
import random
import pickle

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score
from sklearn.decomposition import PCA
from matplotlib import pyplot

import pandas as pd
import csv
import settings


#pathはsettings.pyで管理
#それ以外のhyper parameterをargsで管理
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AASC',help="AASC or PeerRead")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--frequency', type=int, default=5, help="frequency to remove rare entity")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--epoch', type=int, default=5, help="number of epochs")
    parser.add_argument('--WINDOW_SIZE', type=int, default=125, help="the length of context length")
    parser.add_argument('--MAX_LEN', type=int, default=256, help="MAX length of the input")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="scibert", help="scibert or bert")
    parser.add_argument('--mask_type', type=str, default="tail", help="mask [tail,random,both] paper")
    parser.add_argument('--train_data', type=str, default="excluded", help="remove low frequency data or not [excluded,full] during training")
    parser.add_argument('--test_data', type=str, default="excluded", help="remove low frequency data or not [excluded,full]")
    return parser.parse_args()

#test evaluation for citation recommendation
def predict(args,epoch,model,ent_vocab,test_set):
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 2, collate_fn=Collate_fn(args.MAX_LEN).collate_fn)
    mrr_all = 0
    Recallat5_all = Recallat10_all = Recallat30_all = Recallat50_all = 0
    MAP_all = 0
    l_all = 0
    l_prev = 0
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_"+args.mask_type+"_without_citing.txt","w")
    with torch.no_grad():
        for (inputs,labels) in test_dataloader:
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
        s = ""
        s += "MRR\n" + str(mrr_all/l_all)+"\n"
        s += "Recallat5\n" + str(Recallat5_all/l_all)+"\n"
        s += "Recallat10\n" + str(Recallat10_all/l_all)+"\n"
        s += "Recallat30\n" + str(Recallat30_all/l_all)+"\n"
        s += "MAP\n" + str(MAP_all/l_all)+"\n"
        fw.write(s)
        print(s)

def node_classification(args,epoch,model,ent_vocab):
    X_train,y_train,X_test,y_test = load_data_SVM_with_context(model,ent_vocab,args.MAX_LEN,args.WINDOW_SIZE)
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
    pyplot.savefig("images/TransBasedCitEmb_randomMASK.png") # 保存
    Cs = [2 , 2**5, 2 **10]
    gammas = [2 ** -9, 2 ** -6, 2** -3,2 ** 3, 2 ** 6, 2 ** 9]
    svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
    products = [(C,gamma) for C,gamma in product(Cs,gammas)]
    print("training done")
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_"+args.mask_type+"_nodeclassification.txt","w")
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
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_"+args.mask_type+"_intentidentification.txt","w")
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

class Collate_fn():
    def __init__(self,MAX_LEN):
        self.max_words = MAX_LEN
    def collate_fn(self,batch):
        input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask"]
        target_keys = ["masked_lm_labels","word_seq_len"]
        batch_x = {n: [] for n in input_keys}
        batch_y = {n: [] for n in target_keys}

        for sample in batch:
            word_pad = self.max_words - len(sample["input_ids"])
            if word_pad > 0:
                batch_x["input_ids"].append(sample["input_ids"]+[-1]*word_pad)
                batch_x["position_ids"].append(sample["position_ids"]+[0]*word_pad)
                batch_x["token_type_ids"].append(sample["token_type_ids"]+[0]*word_pad)
                batch_x["n_word_nodes"].append(self.max_words)
                batch_x["masked_lm_labels"].append(sample["masked_lm_labels"]+[-1]*word_pad)
                adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
                adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
                adj = torch.cat((adj,torch.zeros(self.max_words,word_pad,dtype=torch.int)),dim=1)
                #attention_maskは普通に文章内に対して1で文章外に対して0でいい
                batch_x['attention_mask'].append(adj)
                batch_y["masked_lm_labels"].append(sample["masked_lm_labels"]+[-1]*word_pad)
                batch_y["word_seq_len"].append(len(sample["input_ids"]))
            else:
                batch_x["input_ids"].append(sample["input_ids"])
                batch_x["position_ids"].append(sample["position_ids"])
                batch_x["token_type_ids"].append(sample["token_type_ids"])
                batch_x["n_word_nodes"].append(self.max_words)
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

#citation contextだけでもいいんじゃないか説を検証
def main():
    args = parse_args()

    #load entity embeddings
    #TODO 初期化をSPECTERで行う
    train_set, test_set, ent_vocab = load_AASC_graph_data(args)
    num_ent = len(ent_vocab)

    devices = list(range(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("GPU NO")

    # load parameters
    if args.pretrained_model == "scibert":
        model = PTBCN.from_pretrained(settings.pretrained_scibert_path,num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    else:
        model = PTBCN.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    model.change_type_embeddings()
    model.cuda()
    print('parameters of SciBERT has been loaded.')

    # fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_without_citing.bin"
    pretrained_model_path = os.path.join(settings.model_path,model_name)
    print("train start")
    if args.train:
        for epoch in range(1,args.epoch+1):
            train_set, test_set, ent_vocab = load_AASC_graph_data(args)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = os.cpu_count()//2, collate_fn=Collate_fn(args.MAX_LEN).collate_fn)
            print("epoch: "+str(epoch))
            with tqdm(train_dataloader) as pbar:
                for i,(inputs,labels) in enumerate(pbar):
                    optimizer.zero_grad()
                    outputs = model(input_ids=inputs["input_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),masked_lm_labels=inputs["masked_lm_labels"].cuda(),attention_mask=inputs["attention_mask"].cuda())
                    loss = outputs["loss"]
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(collections.OrderedDict(loss=loss.detach().cpu().numpy()))
            if epoch % 5 == 0:
                #save model
                model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_without_citing.bin"
                torch.save(model.state_dict(),os.path.join(settings.model_path,model_name))
    print("train end")
    if args.predict:
        for i in range(1,2):
            epoch = i*5
            model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_without_citing.bin"
            model.load_state_dict(torch.load(os.path.join(settings.model_path,model_name)))
            model.eval()
            predict(args,epoch,model,ent_vocab,train_set)
            #node_classification(args,epoch,model,ent_vocab)


if __name__ == '__main__':
    main()