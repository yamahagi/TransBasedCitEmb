import os
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer
from transformers import BertConfig, BertTokenizer, BertForMaskedLM

import fitlog
from fastNLP import FitlogCallback, WarmupCallback, GradientClipCallback
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Trainer, Tester

sys.path.append('../')
from dataloader import SCIGraphDataSet,PeerReadDataSet,AASCDataSet
from model import PTBCN,confirm
from metrics import MacroMetric
from metrics import MRR
from utils import build_label_vocab, build_temp_ent_vocab
from collections import Counter
from itertools import product
import collections
from tqdm import tqdm
import random

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score

import pandas as pd
import csv

class ClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 384).cuda()
        self.fc2 = nn.Linear(384, 12).cuda()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return x

def train_classification(model,final_layer,X_train_batch,y_train_batch,taskweight):
    #optimizer = optim.SGD(list(model.parameters())+list(final_layer.fc1.parameters())+list(final_layer.fc2.parameters()), lr=5e-5)
    optimizer = optim.SGD(list(final_layer.fc1.parameters())+list(final_layer.fc2.parameters()), lr=5e-5)
    loss = nn.CrossEntropyLoss(weight=torch.tensor(taskweight).cuda())
    for X_train,y_train in tqdm(zip(X_train_batch,y_train_batch)):
        flg = 0
        optimizer.zero_grad()
        for x in X_train:
            output = model(input_ids=x["input_ids"].cuda(),position_ids=x["position_ids"].cuda(),token_type_ids=x["token_type_ids"].cuda(),masked_lm_labels=x["masked_lm_labels"].cuda(),attention_mask=x["attention_mask"].cuda())
            entity_logits = output["sequence_output"][0][0]
            output = final_layer(entity_logits)
            if flg == 0:
                outputs = torch.stack([output],dim=0)
                flg = 1
            else:
                output = torch.stack([output],dim=0)
                outputs = torch.cat([outputs,output],dim=0)
        loss1 = loss(outputs,torch.tensor(y_train).cuda())
        loss1.backward()
        optimizer.step()

def predict_classification(model,final_layer,X_test,y_test):
    predicted = []
    with torch.no_grad():
        for x in X_test:
            output = model(input_ids=x["input_ids"].cuda(),position_ids=x["position_ids"].cuda(),token_type_ids=x["token_type_ids"].cuda(),masked_lm_labels=x["masked_lm_labels"].cuda(),attention_mask=x["attention_mask"].cuda())
            entity_logits = output["sequence_output"][0][0]
            output = final_layer(entity_logits)
            argmax = np.argmax(np.array(output.cpu()))
            predicted.append(argmax)
        print("正解率＝", accuracy_score(y_test, predicted))
        print("マクロ平均＝", f1_score(y_test, predicted,average="macro"))
        print("ミクロ平均＝", f1_score(y_test, predicted,average="micro"))
        print(collections.Counter(predicted))




def makedatabatch(X,y):
    #dataをrandomにバッチ化する
    data = [(x,y) for x,y in zip(X,y)]
    taskweight = [0 for i in range(12)]
    for (x,y) in data:
        taskweight[y] += 1
    data_random = random.sample(data,len(data))
    X_random_batch = []
    y_random_batch = []
    flg = 0
    while flg == 0:
        if len(data_random) <= 10:
            X_random_batch.append([xy[0] for xy in data_random])
            y_random_batch.append([xy[1] for xy in data_random])
            flg = 1
        else:
            X_random_batch.append([xy[0] for xy in data_random[:10]])
            y_random_batch.append([xy[1] for xy in data_random[:10]])
            data_random = data_random[10:]
    su = sum(taskweight)
    taskweight = [su/taskw for taskw in taskweight]
    return X_random_batch,y_random_batch,taskweight


#AASCのnode classificationデータを読み込む^
def load_data_SVM(model,entvocab):
    taskn = -1
    taskdict = {}
    ftrain = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_train.txt")
    len1 = 0
    for line in ftrain:
        len1 += 1
    ftrain = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_train.txt")
    taskn = -1
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
            masked_lm_labels1 = torch.tensor([[-1] *512])
            position_ids1 = torch.tensor([[i for i in range(512)]])
            token_type_ids1 = torch.tensor([[1] + [0]*511])
            input_ids1 = torch.tensor([[entvocab[paper]] + [-1]*511])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(511,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(512,511,dtype=torch.int)),dim=1)
            if i % 1000 == 0:
                print("all")
                print(len1)
                print(i)
            output = model(input_ids=input_ids1.cuda(),position_ids=position_ids1.cuda(),token_type_ids=token_type_ids1.cuda(),masked_lm_labels=masked_lm_labels1.cuda(),attention_mask=torch.stack([adj],dim=0).cuda())
            entity_logits = output["sequence_output"][0][0]
            X_train.append(np.array(entity_logits.cpu()))
            y_train.append(taskdict[task])
        ftest = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/title2task_test.txt")
        X_test = []
        y_test = []
        for line in ftest:
            l = line[:-1].split("\t")
            paper = l[0]
            task = l[1]
            masked_lm_labels1 = torch.tensor([[-1] *512])
            position_ids1 = torch.tensor([[i for i in range(512)]])
            token_type_ids1 = torch.tensor([[1] + [0]*511])
            input_ids1 = torch.tensor([[entvocab[paper]] + [-1]*511])
            adj = torch.ones(1, 1, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(511,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(512,511,dtype=torch.int)),dim=1)
            output = model(input_ids=input_ids1.cuda(),position_ids=position_ids1.cuda(),token_type_ids=token_type_ids1.cuda(),masked_lm_labels=masked_lm_labels1.cuda(),attention_mask=torch.stack([adj],dim=0).cuda())
            entity_logits = output["sequence_output"][0][0]
            X_test.append(np.array(entity_logits.cpu()))
            y_test.append(taskdict[task])
    return X_train,y_train,X_test,y_test

#入力: directory
def load_PeerRead_graph_data(path,frequency):
    def extract_by_frequency(path_train, path_test,frequency):
        dftrain = pd.read_csv(path_train)
        dftest = pd.read_csv(path_test)
        source_cut_train = dftrain[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        source_cut_test = dftest[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        ftrain_fre = open(path_train[:-4]+"_frequency"+str(frequency)+".csv","w")
        ftest_fre = open(path_test[:-4]+"_frequency"+str(frequency)+".csv","w")
        wtrain = csv.writer(ftrain_fre)
        wtest = csv.writer(ftest_fre)
        wtrain.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        wtest.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        source_train_keys = source_cut_train.source_id.value_counts().keys()
        source_test_keys = source_cut_test.source_id.value_counts().keys()
        dic1 = {}
        train_counts = source_cut_train.source_id.value_counts()
        test_counts = source_cut_test.source_id.value_counts()
        for key in source_train_keys:
            dic1[key] = train_counts[key]
        for key in source_test_keys:
            if key in dic1:
                dic1[key] += test_counts[key]
            else:
                dic1[key] = test_counts[key]
        frequencylist = []
        for key in dic1:
            if dic1[key] >= frequency:
                frequencylist.append(key)
        dftrain = dftrain.loc[dftrain["source_id"].isin(frequencylist)]
        dftest = dftest.loc[dftest["source_id"].isin(frequencylist)]
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"],dftrain["source_id"]):
            wtrain.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftrain_fre.close()
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftest["target_id"],dftest["left_citated_text"],dftest["right_citated_text"],dftest["source_id"]):
            wtest.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftest_fre.close()
        entitylist = list(set(list(dftrain["source_id"].values) + list(dftrain["target_id"].values) + list(dftest["source_id"].values) + list(dftest["target_id"].values)))
        entvocab = {"UNKNOWN":0,"MASK":1}
        for i,entity in enumerate(entitylist):
            entvocab[entity] = i+2
        return path_train[:-4]+"_frequency"+str(frequency)+".csv",path_test[:-4]+"_frequency"+str(frequency)+".csv",entvocab
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    path_train,path_test,entvocab = extract_by_frequency(path_train,path_test,frequency)
    dataset_test = PeerReadDataSet(path_test,ent_vocab=entvocab)
    print("test data load done")
    dataset_train = PeerReadDataSet(path_train,ent_vocab=entvocab)
    print("train data load done")
    return dataset_train,dataset_test,entvocab

#入力: directory
def load_AASC_graph_data(path,frequency):
    def extract_by_frequency(path_train, path_test,frequency):
        dftrain = pd.read_csv(path_train,quotechar="'")
        dftest = pd.read_csv(path_test,quotechar="'")
        source_cut_train = dftrain[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        source_cut_test = dftest[['target_id', 'source_id']].drop_duplicates(subset=['target_id', 'source_id'])
        ftrain_fre = open(path_train[:-4]+"_frequency"+str(frequency)+".csv","w")
        ftest_fre = open(path_test[:-4]+"_frequency"+str(frequency)+".csv","w")
        wtrain = csv.writer(ftrain_fre,quotechar="'")
        wtest = csv.writer(ftest_fre,quotechar="'")
        wtrain.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        wtest.writerow(["target_id","left_citated_text","right_citated_text","source_id"])
        source_train_keys = source_cut_train.source_id.value_counts().keys()
        source_test_keys = source_cut_test.source_id.value_counts().keys()
        dic1 = {}
        train_counts = source_cut_train.source_id.value_counts()
        test_counts = source_cut_test.source_id.value_counts()
        for key in source_train_keys:
            dic1[key] = train_counts[key]
        for key in source_test_keys:
            if key in dic1:
                dic1[key] += test_counts[key]
            else:
                dic1[key] = test_counts[key]
        frequencylist = []
        for key in dic1:
            if dic1[key] >= frequency:
                frequencylist.append(key)
        dftrain = dftrain.loc[dftrain["source_id"].isin(frequencylist)]
        dftest = dftest.loc[dftest["source_id"].isin(frequencylist)]
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftrain["target_id"],dftrain["left_citated_text"],dftrain["right_citated_text"],dftrain["source_id"]):
            wtrain.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftrain_fre.close()
        for target_id,left_citated_text,right_citated_text,source_id in zip(dftest["target_id"],dftest["left_citated_text"],dftest["right_citated_text"],dftest["source_id"]):
            wtest.writerow([target_id,left_citated_text,right_citated_text,source_id])
        ftest_fre.close()
        entitylist = list(set(list(dftrain["source_id"].values) + list(dftrain["target_id"].values) + list(dftest["source_id"].values) + list(dftest["target_id"].values)))
        entvocab = {"UNKNOWN":0,"MASK":1}
        for i,entity in enumerate(entitylist):
            entvocab[entity] = i+2
        return path_train[:-4]+"_frequency"+str(frequency)+".csv",path_test[:-4]+"_frequency"+str(frequency)+".csv",entvocab
    path_train = os.path.join(path,"train.csv")
    path_test = os.path.join(path,"test.csv")
    path_train,path_test,entvocab = extract_by_frequency(path_train,path_test,frequency)
    dataset_test = AASCDataSet(path_test,ent_vocab=entvocab)
    print("test data load done")
    dataset_train = AASCDataSet(path_train,ent_vocab=entvocab)
    print("train data load done")
    return dataset_train,dataset_test,entvocab

def exploit_true_labels(masked_lm_labels_batch):
    true_labels = []
    for masked_lm_labels in masked_lm_labels_batch:
        for i,masked_lm_label in enumerate(masked_lm_labels):
            if masked_lm_label != -1:
                true_labels.append((i,masked_lm_label))
    return true_labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC',
                        help="data directory path")
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help="fitlog directory path")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--frequency', type=int, default=5, help="frequency to remove rare entity")
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--ent_lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--beta', type=float, default=0.999, help="beta_2 of adam")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--warm_up', type=float, default=0.1, help="warmup proportion or steps")
    parser.add_argument('--epoch', type=int, default=3, help="number of epochs")
    parser.add_argument('--grad_accumulation', type=int, default=1, help="gradient accumulation")
    parser.add_argument('--gpu', type=str, default='all', help="run script on which devices")
    parser.add_argument('--debug', action='store_true', help="do not log")
    parser.add_argument('--model_path', type=str, default="../model/",
                        help="the path of directory containing model and entity embeddings.")
    parser.add_argument('--ent_dim', type=int, default=200, help="dimension of entity embeddings")
    parser.add_argument('--ip_config', type=str, default='emb_ip.cfg')
    parser.add_argument('--name', type=str, default='test', help="experiment name")
    return parser.parse_args()


def train(model,train_data_iter):
    model.train()
    total_loss = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for (inputs,labels) in tqdm(train_data_iter):
        optimizer.zero_grad()
        outputs = model(input_ids=inputs["input_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),masked_lm_labels=inputs["masked_lm_labels"].cuda(),attention_mask=inputs["attention_mask"].cuda())
        loss = criterion(outputs["entity_logits"].view(-1,outputs["entity_logits"].size(-1)), inputs["masked_lm_labels"].cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

def main():
    args = parse_args()

    if args.debug:
        fitlog.debug()
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #word_mask_indexを取得？
    tokenizer = BertTokenizer.from_pretrained('pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
    word_mask_index = tokenizer.mask_token_id
    word_vocab_size = len(tokenizer)
    #path_train = os.path.join(args.data_dir,"train.txt")
    #path_test = os.path.join(args.data_dir,"test.txt")
    #train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir)
    #train_set, test_set, ent_vocab = load_PeerRead_graph_data(args.data_dir,args.frequency)
    train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir,args.frequency)



    #load entity embeddings
    #TODO 初期化をDoc2Vecで行う
    num_ent = len(ent_vocab)

    # load parameters
    model = PTBCN.from_pretrained('pretrainedmodel/scibert_scivocab_uncased',
		    num_ent=len(ent_vocab),
		    ent_lr=args.ent_lr)
    model.change_type_embeddings()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics = [MacroMetric(pred='pred', target='target')]
    devices = list(range(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("GPU NO")

    #fitlog_callback = FitlogCallback(tester=tester, log_loss_every=100, verbose=1)
    gradient_clip_callback = GradientClipCallback(clip_value=1, clip_type='norm')
    warmup_callback = WarmupCallback(warmup=args.warm_up, schedule='linear')
    #emb_callback = EmbUpdateCallback(model.ent_embeddings)
    #all_callbacks = [gradient_clip_callback, emb_callback]

    bsz = args.batch_size // args.grad_accumulation
    testloader = torch.utils.data.DataLoader(test_set,batch_size=2,shuffle=False,num_workers=4)
    test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=args.batch_size, sampler=None,num_workers=4,collate_fn=test_set.collate_fn)
    train_data_iter = TorchLoaderIter(dataset=train_set,
                                      batch_size=bsz,
                                      sampler=RandomSampler(),
                                      num_workers=os.cpu_count()//2,
                                      collate_fn=train_set.collate_fn)
    trainer = Trainer(train_data=train_data_iter,
                      model=model,
                      optimizer=optimizer,
                      loss=LossInForward(),
                      batch_size=bsz,
                      update_every=args.grad_accumulation,
                      n_epochs=1,
                      metrics=None,
                      callbacks=[gradient_clip_callback, warmup_callback],
                      device=devices,
                      save_path=args.model_path,
                      use_tqdm=True)
    trainer._load_model(model,"DataParallel_2021-01-29-17-29-35-606006")
    a = trainer._load_model(model,"DataParallel_2021-01-29-17-29-35-606006")

    #test
    X_train,y_train,X_test,y_test = load_data_SVM(model,ent_vocab)
    print("SVM data load done")
    print("training start")
    Cs = [2 , 2**5, 2 **10]
    gammas = [2 ** -9, 2 ** -6, 2** -3,2 ** 3, 2 ** 6, 2 ** 9]
    svs = [svm.SVC(C=C, gamma=gamma).fit(X_train, y_train) for C, gamma in product(Cs, gammas)]
    print("training done")
    for sv in svs:
        test_label = sv.predict(X_test)
        print("正解率＝", accuracy_score(y_test, test_label))
        print("マクロ平均＝", f1_score(y_test, test_label,average="macro"))
        print("ミクロ平均＝", f1_score(y_test, test_label,average="micro"))
        print(collections.Counter(test_label))



if __name__ == '__main__':
    main()
