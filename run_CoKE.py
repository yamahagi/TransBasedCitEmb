import os
import sys

import argparse
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from transformers import RobertaConfig, RobertaTokenizer
from transformers import BertConfig, BertTokenizer, BertModel
from dataloader_CoKE import load_AASC_graph_data,load_PeerRead_graph_data

import fitlog
from fastNLP import FitlogCallback, WarmupCallback, GradientClipCallback
from fastNLP import RandomSampler, TorchLoaderIter, LossInForward, Trainer, Tester

sys.path.append('../')
from model import PTBCNCOKE
from collections import Counter
from itertools import product
import collections
from tqdm import tqdm
import random

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
        train_set, test_set, ent_vocab = load_AASC_graph_data(args.data_dir,args.frequency,args.MAX_LEN)
    else:
        train_set, test_set, ent_vocab = load_PeerRead_graph_data(args.data_dir,args.frequency,args.MAX_LEN)
   
    num_ent = len(ent_vocab)
    if args.pretrained_model == "scibert":
        model = PTBCNCOKE.from_pretrained('/home/ohagi_masaya/TransBasedCitEmb/pretrainedmodel/scibert_scivocab_uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    else:
        model = PTBCNCOKE.from_pretrained('bert-base-uncased',num_ent=len(ent_vocab),MAX_LEN=args.MAX_LEN)
    model.change_type_embeddings()
    print('parameters of SciBERT has been loaded.')
    #fine-tune
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

    devices = list(range(torch.cuda.device_count()))

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
    model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
    pretrained_model_path = os.path.join(args.model_path,model_name)
    print("train start")
    if args.train:
        for i in range(args.epoch):
            model_name = "model_"+"epoch"+str(args.epoch-i)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
            optimizer_name = "optimizer_"+"epoch"+str(args.epoch-i)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
            pretrained_model_path = os.path.join(args.model_path,model_name)
            pretrained_optimizer_path = os.path.join(args.model_path,optimizer_name)
            if os.path.exists(pretrained_model_path):
                print("found")
                print(pretrained_model_path)
                model.load_state_dict(torch.load(pretrained_model_path))
                optimizer.load_state_dict(torch.load(pretrained_optimizer_path))
                for j in range(1,i+1):
                    trainer.train(load_best_model=False)
                    if args.epoch-i+j % 2 == 0:
                        model_name = "model_"+"epoch"+str(args.epoch-i+j)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
                        optimizer_name = "optimizer_"+"epoch"+str(args.epoch-i+j)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
                        torch.save(model.state_dict(),os.path.join(args.model_path,model_name))
                        torch.save(optimizer.state_dict(), os.path.join(args.model_path,optimizer_name))
                break
        else:
            for i in range(1,args.epoch+1):
                trainer.train(load_best_model=False)
                if i%2 == 0:
                    model_name = "model_"+"epoch"+str(i)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
                    optimizer_name = "optimizer_"+"epoch"+str(i)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
                    torch.save(model.state_dict(),os.path.join(args.model_path,model_name))
                    torch.save(optimizer.state_dict(), os.path.join(args.model_path,optimizer_name))
    print("train end")
    #test
    testloader = torch.utils.data.DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=1)
    test_data_iter = TorchLoaderIter(dataset=test_set, batch_size=args.batch_size, sampler=None,num_workers=1,collate_fn=test_set.collate_fn)
    mrr_all = 0
    Recallat5_all = 0
    Recallat10_all = 0
    Recallat30_all = 0
    Recallat50_all = 0
    MAP_all = 0
    l_all = 0
    l_prev = 0
    if args.predict:
        model_name = "model_"+"epoch"+str(args.epoch)+"_batchsize"+str(args.batch_size)+"_learningrate"+str(args.lr)+"_data"+str(args.dataset)+"_WINDOWSIZE"+str(args.WINDOW_SIZE)+"_MAXLEN"+str(args.MAX_LEN)+"_pretrainedmodel"+str(args.pretrained_model)+"_eachMASK.bin"
        pretrained_model_path = os.path.join(args.model_path,model_name)
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()
        fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(args.epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_eachMASK.txt","w")
        with torch.no_grad():
            for (inputs,labels) in test_data_iter:
                outputs = model(target_ids=inputs["target_ids"].cuda(),source_ids=inputs["source_ids"].cuda(),position_ids=inputs["position_ids"].cuda(),token_type_ids=inputs["token_type_ids"].cuda(),attention_mask=inputs["attention_masks"].cuda(),mask_positions=inputs["mask_positions"].cuda(),contexts=inputs["contexts"].cuda())
                MAP,mrr,Recallat5,Recallat10,Recallat30,Recallat50,l = Evaluation(outputs["entity_logits"],outputs["masked_lm_labels"])
                mrr_all += mrr
                Recallat5_all += Recallat5
                Recallat10_all += Recallat10
                Recallat30_all += Recallat30
                Recallat50_all += Recallat50
                MAP_all += MAP
                l_all += l
                if l_all - l_prev > 100:
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

    if args.node_classification and args.dataset == "AASC":
        fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(args.epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_nodeclassification_eachMASK.txt","w")
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
    
    if args.intent_identification and args.dataset == "AASC":
        fw = open("../results/"+"batch_size"+str(args.batch_size)+"epoch"+str(args.epoch)+"dataset"+str(args.dataset)+"WINDOW_SIZE"+str(args.WINDOW_SIZE)+"MAX_LEN"+str(args.MAX_LEN)+"pretrained_model"+str(args.pretrained_model)+"_intentidentification.txt","w")
        X,y = load_data_intent_identification(model,ent_vocab)
        print("intent identification data load done")
        l = [i for i in range(len(X))]
        random.shuffle(l)
        for epoch in range(5):
            if i == 0:
                X_train = [X[i] for i in l[:len(l)//5]]
                y_train = [y[i] for i in l[:len(l)//5]]
                X_test = [X[i] for i in l[len(l)//5:]]
                y_test = [y[i] for i in l[len(l)//5:]]
            elif i == 4:
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


if __name__ == '__main__':
    main()
