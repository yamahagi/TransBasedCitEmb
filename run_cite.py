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
from utils import build_label_vocab, build_temp_ent_vocab,build_ent_vocab
from collections import Counter

import pandas as pd
import csv


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
    parser.add_argument('--window_size', type=int, default=250, help="experiment name")
    parser.add_argument('--MAX_LEN', type=int, default=250, help="experiment name")
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
    tokenizer = BertTokenizer.from_pretrained('../pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
    word_mask_index = tokenizer.mask_token_id
    word_vocab_size = len(tokenizer)
    #path_train = os.path.join(args.data_dir,"train.txt")
    #path_test = os.path.join(args.data_dir,"test.txt")
    #train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir)
    #train_set, test_set, ent_vocab = load_PeerRead_graph_data(args.data_dir,args.frequency)
    if args.dataset == "AASC":
        train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir,args.frequency,args.window_size,args.MAX_LEN)
    elif args.dataset == "PeerRead":
        train_set, test_set, ent_vocab = load_PeerRead_graph_data(args.data_dir,args.frequency,args.window_size,args.MAX_LEN)



    #load entity embeddings
    #TODO 初期化をSPECTERで行う
    num_ent = len(ent_vocab)

    # load parameters
    model = PTBCN.from_pretrained('../pretrainedmodel/scibert_scivocab_uncased',
		    num_ent=len(ent_vocab),
		    ent_lr=args.ent_lr,
                    MAX_LEN=args.MAX_LEN)
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
    if args.data_dir[-1] == "/":
        data_dir_modelname = os.path.basename(args.data_dir[:-1])
    else:
        data_dir_modelname = os.path.basename(args.data_dir)
    model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(data_dir_modelname)+".bin"
    pretrained_model_path = os.path.join(args.model_path,model_name)
    print("train start")
    for i in range(args.epoch):
        model_name = "model_"+"epoch"+str(args.epoch-i)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(data_dir_modelname)+".bin"
        pretrained_model_path = os.path.join(args.model_path,model_name)
        if os.path.exists(pretrained_model_path):
            model.load_state_dict(torch.load(pretrained_model_path))
            for j in range(1,i+1):
                trainer.train(load_best_model=False)
                if args.epoch-i+j % 5 == 0:
                    model_name = "model_"+"epoch"+str(args.epoch-i+j)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(data_dir_modelname)+".bin"
                    torch.save(model.state_dict(),os.path.join(args.model_path,model_name))
            break
    else:
        for i in range(1,args.epoch+1):
            trainer.train(load_best_model=False)
            if i % 5 == 0:
                model_name = "model_"+"epoch"+str(i)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(data_dir_modelname)+".bin"
                torch.save(model.state_dict(),os.path.join(args.model_path,model_name))
    print("train end")

    #test
    testloader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=1)
    test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=args.batch_size, sampler=None,num_workers=1,collate_fn=test_set.collate_fn)
    mrr_all = 0
    l_all = 0
    l_prev = 0
    with torch.no_grad():
        for (inputs,labels) in test_data_iter:
            outputs = model(input_ids=inputs["input_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),masked_lm_labels=inputs["masked_lm_labels"].cuda(),attention_mask=inputs["attention_mask"].cuda())
            mrr,l = confirm(outputs["entity_logits"],inputs["masked_lm_labels"])
            mrr_all += mrr
            l_all += l
            if l_all - l_prev > 100:
                l_prev = l_all
                print(l_all)
    print(mrr_all)
    print(l_all)
    print(mrr_all/l_all)


if __name__ == '__main__':
    main()
