import os
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer
from transformers import BertConfig, BertTokenizer, BertModel
from dataloader_CoKE import load_AASC_graph_data,load_PeerRead_graph_data,load_data_SVM,load_data_intent_identification,Collate_fn

import fitlog
from fastNLP import FitlogCallback, WarmupCallback, GradientClipCallback
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Trainer, Tester
from metrics import Evaluation

sys.path.append('../')
from model import PTBCNCOKE
from collections import Counter
from itertools import product
import collections
from tqdm import tqdm
import random

import pandas as pd
import csv

from sklearn import svm
from sklearn.metrics import accuracy_score,f1_score
import settings



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/ohagi_masaya/TransBasedCitEmb/dataset/AASC',help="data directory path")
    parser.add_argument('--dataset', type=str, default='AASC',help="AASC or PeerRead")
    parser.add_argument('--log_dir', type=str, default='./logs/',
                        help="fitlog directory path")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument('--frequency', type=int, default=5, help="frequency to remove rare entity")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")
    parser.add_argument('--beta', type=float, default=0.999, help="beta_2 of adam")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="weight decay")
    parser.add_argument('--warm_up', type=float, default=0.1, help="warmup proportion or steps")
    parser.add_argument('--epoch', type=int, default=3, help="number of epochs")
    parser.add_argument('--grad_accumulation', type=int, default=1, help="gradient accumulation")
    parser.add_argument('--gpu', type=str, default='all', help="run script on which devices")
    parser.add_argument('--debug', action='store_true', help="do not log")
    parser.add_argument('--model_path', type=str, default="../model/",
                        help="the path of directory containing model and entity embeddings.")
    parser.add_argument('--WINDOW_SIZE', type=int, default=250, help="the length of context length")
    parser.add_argument('--MAX_LEN', type=int, default=512, help="MAX length of the input")
    parser.add_argument('--train', type=bool, default=True, help="train or not")
    parser.add_argument('--predict', type=bool, default=True, help="predict or not")
    parser.add_argument('--node_classification', type=bool, default=True, help="conduct node classification or not")
    parser.add_argument('--intent_identification', type=bool, default=True, help="conduct intent identification or not")
    parser.add_argument('--pretrained_model', type=str, default="scibert", help="scibert or bert")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.debug:
        fitlog.debug()
    if args.gpu != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        print("GPU OK")
    else:
        print("GPU NO")
    
    if args.dataset == "AASC":
        train_set, test_set, ent_vocab = load_AASC_graph_data(settings.citation_recommendation_dir,args.frequency,args.MAX_LEN)
    else:
        train_set, test_set, ent_vocab = load_PeerRead_graph_data(settings.citation_recommendation_dir,args.frequency,args.MAX_LEN)
    num_ent = len(ent_vocab)
    if args.pretrained_model == "scibert":
        model = PTBCNCOKE.from_pretrained(settings.pretrained_scibert_path,num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    else:
        model = PTBCNCOKE.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    model.change_type_embeddings()
    #fine-tune
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.bias', 'layer_norm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    devices = list(range(torch.cuda.device_count()))

    model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
    pretrained_model_path = os.path.join(args.model_path,model_name)
    print("train start")
    if args.train == True:
        for epoch in range(1,args.epoch+1):
            train_set, test_set, ent_vocab = load_AASC_graph_data(settings.citation_recommendation_dir,args.frequency,args.WINDOW_SIZE,args.MAX_LEN,settings.pretrained_scibert_path)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True, num_workers = 2, collate_fn=Collate_fn(args.MAX_LEN).collate_fn)
            print("epoch: "+str(epoch))
            with tqdm(train_dataloader) as pbar:
                for i,(inputs,labels) in enumerate(pbar):
                    optimizer.zero_grad()
                    outputs = model(target_ids=inputs["target_ids"].cuda(),source_ids=inputs["source_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),contexts=inputs["contexts"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),masked_lm_labels=inputs["masked_lm_labels"].cuda(),attention_mask=inputs["attention_mask"].cuda(),mask_positions=inputs["mask_positions"].cuda())
                    loss = outputs["loss"]
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix(collections.OrderedDict(loss=loss.detach().cpu().numpy()))
            if epoch % 5 == 0:
                #save model
                model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_randomMASK.bin"
                torch.save(model.state_dict(),os.path.join(settings.model_path,model_name))
    print("train end")
    if args.predict == True:
        for i in range(1,2):
            epoch = i*5
            model_name = "model_"+"epoch"+str(epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_randomMASK.bin"
            model.load_state_dict(torch.load(os.path.join(settings.model_path,model_name)))
            model.eval()
            predict(args,epoch,model,ent_vocab,test_set)
            node_classification(args,epoch,model,ent_vocab)


if __name__ == '__main__':
    main()
