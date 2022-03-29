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
from utils import build_ent_vocab,count_times
from dataloader import load_AASC_graph_data,load_PeerRead_graph_data,make_adjacent_matrix
from load_node_classification import load_data_SVM_with_context,load_data_SVM_from_feedforward,load_data_SVM_with_context_all_layer,load_data_SVM_from_linear
from load_intent_identification import load_data_intent_identification_with_context
from itertools import product
import collections
from collections import defaultdict
from tqdm import tqdm
import random
import pickle

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.decomposition import PCA
from matplotlib import pyplot

import seaborn as sns

import pandas as pd
import csv
import settings

from load_link_prediction import save_embeddings

import matplotlib.pyplot as plt


#pathはsettings.pyで管理
#それ以外のhyper parameterをargsで管理
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AASC',choices=['AASC','PeerRead'],help="AASC or PeerRead")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--frequency', type=int, default=5, help="frequency to remove rare entity")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--gamma', type=float, default=0.8, help="gamma for StructureAwareCrossEntropy")
    parser.add_argument('--epoch', type=int, default=5, help="number of epochs")
    parser.add_argument('--WINDOW_SIZE', type=int, default=125,choices=[125,250], help="the length of context length")
    parser.add_argument('--MAX_LEN', type=int, default=256,choices=[256,512], help="MAX length of the input")
    parser.add_argument('--final_layer', type=str, default="linear",choices=["feedforward","linear"], help="choose final layer feedforward or linear layer")
    parser.add_argument('--loss_type', type=str, default="CrossEntropy", choices=["CrossEntropy","StructureAwareCrossEntropy","MeanSquaredError"],help="CrossEntropy, StructureAwareCrossEntropy or MeanSquaredError Loss")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--pretrained_model', type=str, default="scibert", choices=["scibert","bert","nopretrain"], help="scibert or bert")
    parser.add_argument('--mask_type', type=str, default="tail", choices=["tail","random","both"], help="mask [tail,random,both] paper")
    parser.add_argument('--train_data', type=str, default="excluded", choices=["excluded","full"], help="remove low frequency data or not [excluded,full] during training")
    parser.add_argument('--test_data', type=str, default="excluded", choices=["excluded","full"], help="remove low frequency data or not [excluded,full]")
    return parser.parse_args()

def attention_visualization(args,epoch,model,ent_vocab,test_set,source_times_dict):
    #sampled test dataに対してvisualizationを行う
    #test setから100個ほどsampling
    #attentionを可視化
    #結果を添付
    return 0

def highlight(word, attn):
  html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn)), int(255*(1 - attn)))
  return '<span style="background-color: {}">{}</span>'.format(html_color, word)

def mk_html(index, batch, preds, attention_weight):
  sentence = batch.Text[0][index]
  label = batch.Label[index].item()
  pred = preds[index].item()

  label_str = id2cat[label]
  pred_str = id2cat[pred]

  html = "正解カテゴリ: {}<br>予測カテゴリ: {}<br>".format(label_str, pred_str)

  # 文章の長さ分のzero tensorを宣言
  seq_len = attention_weight.size()[2]
  all_attens = torch.zeros(seq_len).to(device)

  for i in range(12):
    all_attens += attention_weight[index, i, 0, :]

  for word, attn in zip(sentence, all_attens):
    if tokenizer.convert_ids_to_tokens([word.tolist()])[0] == "[SEP]":
      break
    html += highlight(tokenizer.convert_ids_to_tokens([word.numpy().tolist()])[0], attn)
  html += "<br><br>"
  return html

def plot_losses(args,losses,epoch):
    losses = np.array(losses)
    x = np.array([i+1 for i in range(len(losses))])
    fig = plt.figure()
    plt.plot(x, losses)
    fig_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_epoch"+str(epoch)+".png"
    fig.savefig("./images/"+fig_name)

#予測結果を書き込む関数を作りたい
#データとともに予測された論文を5つほど提示
#MRRも提示
def predict_write_csv(args,epoch,model,ent_vocab,test_set,source_times_dict):
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 2, collate_fn=Collate_fn(args.MAX_LEN).collate_fn)


#test evaluation for citation recommendation
def predict(args,epoch,model,ent_vocab,test_set,source_times_dict):
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 2, collate_fn=Collate_fn(args.MAX_LEN).collate_fn)
    mrr_all = 0
    Recallat5_all = Recallat10_all = Recallat30_all = Recallat50_all = 0
    MAP_all = 0
    l_all = 0
    l_prev = 0
    fw = open("../results/"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+".txt","w")
    fcsv = open("../results/"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+".csv","w")
    writer = csv.writer(fcsv)
    writer.writerow(["target_id","source_id","top5","MRR"])
    ent_vocab_reverse = {ent_vocab[key]:key for key in ent_vocab}
    score_per_times = defaultdict(list)
    with torch.no_grad():
        for (inputs,labels) in test_dataloader:
            outputs = model(input_ids=inputs["input_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),masked_lm_labels=inputs["masked_lm_labels"].cuda(),attention_mask=inputs["attention_mask"].cuda())
            results_dict,rank_array,score_per_times =  Evaluation(outputs["entity_logits"].cpu(),inputs["masked_lm_labels"],source_times_dict,score_per_times)
            #resultsを取り出す
            MAP_all += results_dict["MAP"]
            mrr_all += results_dict["MRR"]
            Recallat5_all += results_dict["R@5"]
            Recallat10_all += results_dict["R@10"]
            Recallat30_all += results_dict["R@30"]
            Recallat50_all += results_dict["R@50"]
            l_all += results_dict["batch_len"]
            if l_all - l_prev > 500:
                l_prev = l_all
                print(l_all)
                print(mrr_all/l_all)
                print(inputs["target_ids"])
            #csvにtarget_idとsource_id,rank_array上位5つとMRRを書き込む
            writer.writerow([inputs["target_ids"],inputs["source_ids"],rank_array[:5],results_dict["MRR"]])
        s = ""
        s += "MRR\n" + str(mrr_all/l_all)+"\n"
        s += "Recallat5\n" + str(Recallat5_all/l_all)+"\n"
        s += "Recallat10\n" + str(Recallat10_all/l_all)+"\n"
        s += "Recallat30\n" + str(Recallat30_all/l_all)+"\n"
        s += "MAP\n" + str(MAP_all/l_all)+"\n"
        fw.write(s)
        print(s)
        for key in score_per_times:
            score_per_times[key] = sum(score_per_times[key])/len(score_per_times[key])
        #print("MRR per times in train")
        #print(score_per_times)


def node_classification(args,epoch,model,ent_vocab):
    def draw_table(X,y):
        #yをsortする
        argy = np.argsort(y)
        #argyの順列がtableのそれとなる
        #arrayを用意
        heatmap = np.zeros((len(X),len(X)))
        #table[i][j]はX[argy[i]]とX[argy[j]]の内積
        for i in range(len(heatmap)):
            for j in range(len(heatmap)):
                heatmap[i][j] = np.dot(X[argy[i]],X[argy[j]])/(np.linalg.norm(X[argy[i]])*np.linalg.norm(X[argy[j]]))
        pyplot.figure()
        sns.heatmap(heatmap)
        pyplot.savefig("images/TransBasedCitEmb_table.png")
        pyplot.close()
    X_train,y_train,X_test,y_test,papers_train,papers_test = load_data_SVM_with_context(args,model,ent_vocab,args.MAX_LEN,args.WINDOW_SIZE)
    #X_train,y_train,X_test,y_test = load_data_SVM_from_linear(model,ent_vocab)
    #draw_table(X_train+X_test,y_train+y_test)
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
    gammas = [2 ** -9, 2 ** -6, 2** -3,2**-1,2**1,2 ** 3, 2 ** 6, 2 ** 9]
    print(len(X_train))
    print(len(y_train))
    print(collections.Counter(y_train))
    svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
    products = [(C,gamma) for C,gamma in product(Cs,gammas)]
    print("training done")
    fw = open("../results/"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_nodeclassification.txt","w")
    fcsv = open("../results/"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_nodeclassification.csv","w")
    #macroスコア+microスコアが最も高いものを保持する
    argmax_label = []
    maxsum = 0.0
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
        if f1_score(y_test, test_label,average="macro") + f1_score(y_test, test_label,average="micro") > maxsum:
            maxsum = f1_score(y_test, test_label,average="macro") + f1_score(y_test, test_label,average="micro")
            argmax_label = test_label
    writer = csv.writer(fcsv)
    writer.writerow(["paper_id","label"])
    for paper,label in zip(papers_test,argmax_label):
        writer.writerow([paper,label])

def intent_identification(args,epoch,model,ent_vocab):
    #fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_"+args.mask_type+"_intentidentification.txt","w")
    #X,y = load_data_intent_identification_with_context(model,ent_vocab,args.MAX_LEN,args.WINDOW_SIZE)
    X,y = load_data_intent_identification_with_context(args,model,ent_vocab)
    X_concat = [np.concatenate([x[0],x[1]]) for x in X]
    X_minus = [x[0]-x[1] for x in X]
    X_plus = [x[0]+x[1] for x in X]
    """
    print("intent identification data load done")
    print("PCA start")
    pca = PCA(n_components=2)
    pca.fit(X_concat)
    X_visualization = pca.transform(X_concat)
    print("PCA done: " + str(len(X_concat)))
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
    pyplot.savefig("images/TransBasedCitEmb_intent_identification_concat.png") # 保存
    print("PCA start")
    pca = PCA(n_components=2)
    pca.fit(X_minus)
    X_visualization = pca.transform(X_minus)
    print("PCA done: " + str(len(X_minus)))
    print("Y length: " + str(len(y)))
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
    pyplot.savefig("images/TransBasedCitEmb_intent_identification_minus.png") # 保存
    """
    l = [i for i in range(len(X))]
    #fix random seed to 10
    random.seed(10)
    random.shuffle(l)
    dict1 = {}
    macro_f1 = 0
    micro_f1 = 0
    for epoch in range(5):
        if epoch == 0:
            X_train = [X_concat[i] for i in l[:len(l)//5]]
            y_train = [y[i] for i in l[:len(l)//5]]
            X_test = [X_concat[i] for i in l[len(l)//5:]]
            y_test = [y[i] for i in l[len(l)//5:]]
            print(collections.Counter(y_train))
        elif epoch == 4:
            X_train = [X_concat[i] for i in l[len(l)*epoch//5:]]
            y_train = [y[i] for i in l[len(l)*epoch//5:]]
            X_test = [X_concat[i] for i in l[:len(l)*epoch//5]]
            y_test = [y[i] for i in l[:len(l)*epoch//5]]
        else:
            X_train = [X_concat[i] for i in l[len(l)*epoch//5:len(l)*(epoch+1)//5]]
            y_train = [y[i] for i in l[len(l)*epoch//5:len(l)*(epoch+1)//5]]
            X_test = [X_concat[i] for i in l[:len(l)*epoch//5]+l[len(l)*(epoch+1)//5:]]
            y_test = [y[i] for i in l[:len(l)*epoch//5]+l[len(l)*(epoch+1)//5:]]
        """
        Cs = [2 , 2**5, 2 **10]
        gammas = [2 ** -9, 2 ** -6, 2** -3,2 ** 3, 2 ** 6, 2 ** 9]
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
        """
        randomforest = RandomForestClassifier()
        randomforest.fit(X_train,y_train)
        test_label = randomforest.predict(X_test)
        macro_f1 += f1_score(y_test, test_label,average="macro")
        micro_f1 += f1_score(y_test, test_label,average="micro")
        print("training done")
    print("results")
    print(macro_f1/5)
    print(micro_f1/5)
    """
    for sv,product1 in zip(svs,products):
        s1 = "C:"+str(product1[0])+","+"gamma:"+str(product1[1])
        print("正解率＝", dict1[s1]["正解率"]/5)
        print("マクロ平均＝", dict1[s1]["マクロ平均"]/5)
        print("ミクロ平均＝", dict1[s1]["ミクロ平均"]/5)
        print(dict1[s1]["分類結果"])
    """

class Collate_fn():
    def __init__(self,MAX_LEN):
        self.max_words = MAX_LEN
    def collate_fn(self,batch):
        input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask","target_ids","source_ids"]
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
                batch_x["target_ids"].append(sample["target_id"])
                batch_x["source_ids"].append(sample["source_id"])
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
                batch_x["target_ids"].append(sample["target_id"])
                batch_x["source_ids"].append(sample["source_id"])
                adj = torch.ones(len(sample['input_ids']), len(sample['input_ids']), dtype=torch.int)
                #attention_maskは普通に文章内に対して1で文章外に対して0でいい
                batch_x['attention_mask'].append(adj)
                batch_y["masked_lm_labels"].append(sample["masked_lm_labels"])
                batch_y["word_seq_len"].append(len(sample["input_ids"]))
        for k, v in batch_x.items():
            if k == 'attention_mask':
                batch_x[k] = torch.stack(v, dim=0)
            elif k == "target_ids" or k == "source_ids":
                continue
            else:
                batch_x[k] = torch.tensor(v)
        for k, v in batch_y.items():
            batch_y[k] = torch.tensor(v)
        return (batch_x, batch_y)

def main():
    args = parse_args()
    print("arguments")
    print(args)

    #load entity embeddings
    #TODO 初期化をSPECTERで行う
    if args.dataset == "AASC":
        train_set, test_set, ent_vocab = load_AASC_graph_data(args)
    else:
        train_set, test_set, ent_vocab = load_PeerRead_graph_data(args)
    num_ent = len(ent_vocab)
    adj = make_adjacent_matrix(train_set)
    #train data内のcited paperの分布を調べる
    source_times_dict = count_times(args,ent_vocab)
    #print(source_times_dict)

    seed = 11
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    
    # load parameters
    if args.pretrained_model == "scibert":
        model = PTBCN.from_pretrained(settings.pretrained_scibert_path,num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN,final_layer=args.final_layer,loss_type=args.loss_type,adj=adj,gamma=args.gamma)
    elif args.pretrained_model == "bert":
        model = PTBCN.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN,final_layer=args.final_layer,loss_type=args.loss_type,adj=adj,gamma=args.gamma)
    else:
        model = PTBCN.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN,final_layer=args.final_layer,loss_type=args.loss_type,adj=adj,gamma=args.gamma)
        for i,param in enumerate(model.parameters()):
            nn.init.normal_(param,mean=0.0,std=0.2)
    model.change_type_embeddings()
    model.cuda()
    model.train()
    print('parameters of SciBERT has been loaded.')
    
    devices = list(range(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("GPU NO")

    # fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)

    model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_number1.bin"
    #model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_number1.bin"
    pretrained_model_path = os.path.join(settings.model_path,model_name)
    print(model_name)
    if args.loss_type == "StructureAwareCrossEntropy":
        print("StructureAwareCrossEntropyLoss: gamma")
        print(args.gamma)
    print("train start")
    if args.train:
        losses = []
        for epoch in range(1,args.epoch+1):
            if args.dataset == "AASC":
                if epoch == 1 or args.mask_type == "random":
                    train_set, test_set, ent_vocab = load_AASC_graph_data(args)
            else:
                if epoch == 1 or args.mask_type == "random":
                    train_set, test_set, ent_vocab = load_PeerRead_graph_data(args)
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
                    losses.append(loss.detach().cpu().numpy())
            scheduler.step()
            plot_losses(args,losses,epoch)
            if epoch % 5 == 0 or epoch == args.epoch:
                #save model
                model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+".bin"
                #model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_number2.bin"
                torch.save(model.state_dict(),os.path.join(settings.model_path,model_name))
    print("train end")
    if args.predict:
        for i in range(1,args.epoch//5+1):
            epoch = i*5
            model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_number1.bin"
            #model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_"+args.mask_type+"_"+args.final_layer+"_"+args.loss_type+"_number1.bin"
            model.load_state_dict(torch.load(os.path.join(settings.model_path,model_name)))
            model.eval()
            #save_embeddings(model,ent_vocab,args.MAX_LEN,args.WINDOW_SIZE)
            predict(args,epoch,model,ent_vocab,test_set,source_times_dict)
            #node_classification(args,epoch,model,ent_vocab)
            #intent_identification(args,epoch,model,ent_vocab)


if __name__ == '__main__':
    main()
