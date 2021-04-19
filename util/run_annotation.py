import os
import sys

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
from metrics import MacroMetric
from metrics import Evaluation
from utils import build_ent_vocab
from dataloader import load_AASC_graph_data,load_PeerRead_graph_data,load_data_SVM
from collections import Counter
from itertools import product
import collections
from tqdm import tqdm
import random
import pickle
from collections import Counter

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score

import pandas as pd
import csv
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from imblearn import over_sampling
import math


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
    parser.add_argument('--epoch', type=int, default=15, help="number of epochs")
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

def load_data_intent_identification():
    intentn = -1
    intentdict = {}
    #f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/id2intent.txt")
    f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/train.jsonl")
    X_train = []
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
            #target_id = l[0]
            #source_id = l[1]
            intent = js["intent"]
            text = js["text"]
            if intent not in intentdict:
                intentn += 1
                intentdict[intent] = intentn
            X_train.append({"text":text,"intent":intent})
            #X.append({"left_citated_text":left_citated_text,"right_citated_text":right_citated_text})
            y_train.append(intentdict[intent])
    f = open("/home/ohagi_masaya/TransBasedCitEmb/dataset/citationintent/scicite/acl-arc-dataset/test.jsonl")
    X_test = []
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
            #target_id = l[0]
            #source_id = l[1]
            intent = js["intent"]
            text = js["text"]
            if intent not in intentdict:
                intentn += 1
                intentdict[intent] = intentn
            X_test.append({"text":text,"intent":intent})
            #X.append({"left_citated_text":left_citated_text,"right_citated_text":right_citated_text})
            y_test.append(intentdict[intent])
    return X_train,y_train,X_test,y_test,intentdict

def intent_identification(args,epoch,model,ent_vocab):
    fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_intentidentification_randomMASK.txt","w")
    X_train,y_train,X_test,y_test = load_data_intent_identification(model,ent_vocab)
    print("intent identification data load done")
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
def oversampling(X_train,y_train):
    ydict = {}
    for x,y in zip(X_train,y_train):
        if y not in ydict:
            ydict[y] = [x]
        else:
            ydict[y].append(x)
    numdict = {}
    m = 0
    for key in ydict:
        numdict[key] = len(ydict[key])
        m = max(m,numdict[key])
    for key in numdict:
        numdict[key] /= m
    X_train_new = []
    y_train_new = []
    Xy_train_new = []
    for key in numdict:
        l = len(ydict[key]*min(math.floor(1/numdict[key]),3))
        for x in ydict[key]*min(math.floor(1/numdict[key]),3):
            Xy_train_new.append((x,key))
    for x,y in Xy_train_new:
        X_train_new.append(x)
        y_train_new.append(y)
    return X_train_new,y_train_new

def load_data_AASC():
    dftrain = pd.read_csv("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/train.csv",quotechar="'")
    dftest = pd.read_csv("/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC/test.csv",quotechar="'")
    X_train = []
    for target_id,source_id,left_citated_text,right_citated_text in zip(dftrain["target_id"],dftrain["source_id"],dftrain["left_citated_text"],dftrain["right_citated_text"]):
        X_train.append({"target_id":target_id,"source_id":source_id,"left_citated_text":left_citated_text,"right_citated_text":right_citated_text})
    X_test = []
    for target_id,source_id,left_citated_text,right_citated_text in zip(dftest["target_id"],dftest["source_id"],dftest["left_citated_text"],dftest["right_citated_text"]):
        X_test.append({"target_id":target_id,"source_id":source_id,"left_citated_text":left_citated_text,"right_citated_text":right_citated_text})
    return X_train,X_test

def main():
    args = parse_args()
    batch_size = 12

    if args.debug:
        fitlog.debug()
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #load entity embeddings
    #TODO 初期化をSPECTERで行う
    #train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir,args.frequency,args.WINDOW_SIZE,args.MAX_LEN,args.pretrained_model)
    #num_ent = len(ent_vocab)

    # load parameters
    """
    if args.pretrained_model == "scibert":
        model = PTBCN.from_pretrained('../pretrainedmodel/scibert_scivocab_uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    else:
        model = PTBCN.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    model.change_type_embeddings()
    print('parameters of SciBERT has been loaded.')

    # fine-tune
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.bias', 'layer_norm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, args.beta), eps=1e-6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    metrics = [MacroMetric(pred='pred', target='target')]
    devices = list(range(torch.cuda.device_count()))
    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("GPU NO")

    gradient_clip_callback = GradientClipCallback(clip_value=1, clip_type='norm')
    warmup_callback = WarmupCallback(warmup=args.warm_up, schedule='linear')
    """
    bsz = args.batch_size // args.grad_accumulation
    if args.data_dir[-1] == "/":
        data_dir_modelname = os.path.basename(args.data_dir[:-1])
    else:
        data_dir_modelname = os.path.basename(args.data_dir)
    X_train,y_train,X_test,y_test,intentdict = load_data_intent_identification()
    ydict = {}
    for i in y_train+y_test:
        if i not in ydict:
            ydict[i] = 1
        else:
            ydict[i] += 1
    print(ydict)
    """
    l = [i for i in range(len(X))]
    random.shuffle(l)
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
        #X_train, y_train = oversampling(X_train, y_train)
    """
    print(collections.Counter(y_train))
    print(intentdict)
    tokenizer = BertTokenizer.from_pretrained('../../pretrainedmodel/scibert_scivocab_uncased', do_lower_case=True)
    epochs = 50
    input_ids = []
    attention_masks = []
    num_label = max(y_train+y_test)+1
    for x,y1 in zip(X_train,y_train):
        text = x["text"]
        """
        left_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x["left_citated_text"]))
        right_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x["right_citated_text"]))
        """
        text_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        xlen = len(text_tokenized[:512])
        #xlen = len(left_citation_tokenized[-256:])+len(right_citation_tokenized[:256])
        word_pad = 512-xlen
        tokenized_ids = text_tokenized[:512]+[0]*(512-xlen)
        #tokenized_ids = left_citation_tokenized[-256:] + right_citation_tokenized[:256] + [0]*(512-xlen)
        adj = torch.ones(xlen, xlen, dtype=torch.int)
        adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
        adj = torch.cat((adj,torch.zeros(512,word_pad,dtype=torch.int)),dim=1)
        # Add the encoded sentence to the list.
        input_ids.append(tokenized_ids)
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(adj)
    print("load train done")
    # Convert the lists into tensors.
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.stack(attention_masks, dim=0)
    labels = torch.tensor(y_train)
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    input_ids = []
    attention_masks = []
    for x,y1 in zip(X_test,y_test):
        text = x["text"]
        text_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        xlen = len(text_tokenized[:512])
        #xlen = len(left_citation_tokenized[-256:])+len(right_citation_tokenized[:256])
        word_pad = 512-xlen
        tokenized_ids = text_tokenized[:512]+[0]*(512-xlen)
        """
        left_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x["left_citated_text"]))
        right_citation_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x["right_citated_text"]))
        xlen = len(left_citation_tokenized[-256:])+len(right_citation_tokenized[:256])
        word_pad = 512-xlen
        tokenized_ids = left_citation_tokenized[-256:] + right_citation_tokenized[:256] + [0]*(512-xlen)
        """
        adj = torch.ones(xlen, xlen, dtype=torch.int)
        adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
        adj = torch.cat((adj,torch.zeros(512,word_pad,dtype=torch.int)),dim=1)
        # Add the encoded sentence to the list.
        input_ids.append(tokenized_ids)
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(adj)
    print("load test done")
    # Convert the lists into tensors.
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.stack(attention_masks, dim=0)
    labels = torch.tensor(y_test)
    test_dataset = TensorDataset(input_ids, attention_masks, labels)
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler = RandomSampler(train_dataset), # Select batches randomly
        batch_size = batch_size # Trains with this batch size.
    )

    test_dataloader = DataLoader(
        test_dataset,  # The training samples.
        sampler = None, # Select batches randomly
        batch_size = 1 # Trains with this batch size.
    )
    total_steps = len(train_dataloader) * epochs
    model = BertForSequenceClassification.from_pretrained(
        "../../pretrainedmodel/scibert_scivocab_uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = num_label, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    optimizer = AdamW(model.parameters(),
                  lr = 5e-6, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                        num_warmup_steps = 0, # Default value in run_glue.py
                                        num_training_steps = total_steps)
    """
    model.cuda()
    model.load_state_dict(torch.load("../../model/scibert_intentclassification.bin"))
    model.cuda()
    X_train,X_test = load_data_AASC()
    fw = open("train.txt","w")
    pred = []
    with torch.no_grad():
        for i,x in enumerate(X_train):
            if i%2500 == 0:
                print(i)
                print("len")
                print(len(pred))
            left_citated_text = x["left_citated_text"]
            right_citated_text = x["right_citated_text"]
            target_id = x["target_id"]
            source_id = x["source_id"]
            left_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(left_citated_text))[-50:]
            right_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(right_citated_text))[:50]
            xlen = len(left_tokenized+right_tokenized)
            input_id = torch.tensor(left_tokenized+right_tokenized+[0]*(512-xlen)).unsqueeze(0).cuda()
            word_pad = 512-xlen
            adj = torch.ones(xlen, xlen, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(512,word_pad,dtype=torch.int)),dim=1)
            adj = adj.unsqueeze(0).cuda()
            label = torch.tensor([1]).unsqueeze(0).cuda()
            outputs = model(input_ids=input_id,
                             attention_mask=adj,
                             labels=label)
            logits = outputs["logits"]
            logits = logits.detach().cpu().numpy()
            pred += list(np.argmax(logits, axis=1))
            fw.write(target_id+"\t"+str(pred[-1])+"\t"+source_id+"\n")
    fw = open("test.txt","w")
    pred = []
    with torch.no_grad():
        for i,x in enumerate(X_test):
            if i%2500 == 0:
                print(i)
                print("len")
                print(len(pred))
            left_citated_text = x["left_citated_text"]
            right_citated_text = x["right_citated_text"]
            target_id = x["target_id"]
            source_id = x["source_id"]
            left_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(left_citated_text))[-50:]
            right_tokenized = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(right_citated_text))[:50]
            xlen = len(left_tokenized+right_tokenized)
            input_id = torch.tensor(left_tokenized+right_tokenized+[0]*(512-xlen)).unsqueeze(0).cuda()
            word_pad = 512-xlen
            adj = torch.ones(xlen, xlen, dtype=torch.int)
            adj = torch.cat((adj,torch.ones(word_pad,adj.shape[1],dtype=torch.int)),dim=0)
            adj = torch.cat((adj,torch.zeros(512,word_pad,dtype=torch.int)),dim=1)
            adj = adj.unsqueeze(0).cuda()
            label = torch.tensor([1]).unsqueeze(0).cuda()
            outputs = model(input_ids=input_id,
                             attention_mask=adj,
                             labels=label)
            logits = outputs["logits"]
            logits = logits.detach().cpu().numpy()
            pred += list(np.argmax(logits, axis=1))
            fw.write(target_id+"\t"+str(pred[-1])+"\t"+source_id+"\n")
    """
    """
    model.train()
    for epoch_i in range(0, epochs):
        total_train_loss = 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            b_input_ids = batch[0].cuda()
            b_input_mask = batch[1].cuda()
            b_labels = batch[2].cuda()
            outputs = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_dataloader)
        if epoch_i % 10 == 0:
            print(avg_train_loss)
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0
    pred = []
    seikail = []
    model.eval()
    for batch in test_dataloader:
        b_input_ids = batch[0].cuda()
        b_input_mask = batch[1].cuda()
        b_labels = batch[2].cuda()
        with torch.no_grad():
            outputs = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred += list(np.argmax(logits, axis=1))
            seikail += list(label_ids)
    print(collections.Counter(pred))
    print("macro")
    print(f1_score(seikail, pred, average='macro'))
    print("micro")
    print(f1_score(seikail, pred, average='micro'))
    print("accuracy")
    print(accuracy_score(seikail, pred))
    torch.save(model.state_dict(),"../model/scibert_intentclassification.bin")
    """

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



if __name__ == '__main__':
    main()
