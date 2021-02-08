import os
import torch
import json
from torch.utils.data import Dataset
from transformers import RobertaTokenizer,BertTokenizer
import re
import pandas as pd
import csv

WORD_PADDING_INDEX = 1
ENTITY_PADDING_INDEX = 1
#before
MAX_LEN = 512
WINDOW_SIZE = 250
#after
"""
MAX_LEN = 256
WINDOW_SIZE = 100
"""

class PeerReadDataSet(Dataset):
    def __init__(self, path, ent_vocab):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.data = []
        self.tokenizer =  BertTokenizer.from_pretrained('pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
        df = pd.read_csv(path)
        if os.path.exists(os.path.join(self.dirname,self.filename[:-4]+".json")):
            fids = open(os.path.join(self.dirname,self.filename[:-4]+".json"))
            dl = json.load(fids)
            self.data = dl
        else:
            target_ids = df["target_id"]
            source_ids = df["source_id"]
            left_citation_texts = df["left_citated_text"]
            right_citation_texts = df["right_citated_text"]
            citationcontextl = []
            masked_ids = []
            position_ids = []
            for i,(target_id,source_id,left_citation_text,right_citation_text) in enumerate(zip(target_ids,source_ids,left_citation_texts,right_citation_texts)):
                if i % 1000 == 0:
                    print(i)
                citationcontextl = []
                masked_ids = []
                position_ids = []
                token_type_ids = []
                citationcontextl.append(self.tokenizer.cls_token_id)
                citationcontextl.append(ent_vocab[target_id])
                citationcontextl.append(self.tokenizer.sep_token_id)
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                left_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(left_citation_text))
                right_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(right_citation_text))
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab["MASK"]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
                self.data.append({
                    'input_ids': citationcontextl[:MAX_LEN],
                    'masked_lm_labels' : masked_ids[:MAX_LEN],
                    'position_ids': position_ids[:MAX_LEN],
                    'token_type_ids': token_type_ids[:MAX_LEN],
                })
            fids = open(os.path.join(self.dirname,self.filename[:-4]+".json"),"w")
            json.dump(self.data,fids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask"]
        target_keys = ["masked_lm_labels","word_seq_len"]
        max_words = MAX_LEN
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
                adj = torch.cat((adj,torch.zeros(MAX_LEN,word_pad,dtype=torch.int)),dim=1)
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

class AASCDataSet(Dataset):
    def __init__(self, path, ent_vocab):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.data = []
        self.tokenizer =  BertTokenizer.from_pretrained('pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
        df = pd.read_csv(path,quotechar="'")
        if os.path.exists(os.path.join(self.dirname,self.filename[:-4]+"_TBCN.json")):
            fids = open(os.path.join(self.dirname,self.filename[:-4]+"_TBCN.json"))
            dl = json.load(fids)
            self.data = dl
        else:
            target_ids = df["target_id"]
            source_ids = df["source_id"]
            left_citation_texts = df["left_citated_text"]
            right_citation_texts = df["right_citated_text"]
            citationcontextl = []
            masked_ids = []
            position_ids = []
            for i,(target_id,source_id,left_citation_text,right_citation_text) in enumerate(zip(target_ids,source_ids,left_citation_texts,right_citation_texts)):
                if i % 1000 == 0:
                    print(i)
                citationcontextl = []
                masked_ids = []
                position_ids = []
                token_type_ids = []
                citationcontextl.append(self.tokenizer.cls_token_id)
                citationcontextl.append(ent_vocab[target_id])
                citationcontextl.append(self.tokenizer.sep_token_id)
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                left_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(left_citation_text))
                right_citation_tokenized = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(right_citation_text))
                citationcontextl.extend(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab["MASK"]] + right_citation_tokenized[:WINDOW_SIZE])
                position_ids.extend([3+i for i in range(len(left_citation_tokenized[-WINDOW_SIZE:] + [ent_vocab[source_id]] + right_citation_tokenized[:WINDOW_SIZE]))])
                masked_ids.extend([-1]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [ent_vocab[source_id]] + [-1]*len(right_citation_tokenized[:WINDOW_SIZE]))
                token_type_ids.extend([0]*len(left_citation_tokenized[-WINDOW_SIZE:]) + [1] + [0]*len(right_citation_tokenized[:WINDOW_SIZE]))
                self.data.append({
                    'input_ids': citationcontextl[:MAX_LEN],
                    'masked_lm_labels' : masked_ids[:MAX_LEN],
                    'position_ids': position_ids[:MAX_LEN],
                    'token_type_ids': token_type_ids[:MAX_LEN],
                })
            fids = open(os.path.join(self.dirname,self.filename[:-4]+"_TBCN.json"),"w")
            json.dump(self.data,fids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask"]
        target_keys = ["masked_lm_labels","word_seq_len"]
        max_words = MAX_LEN
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
                adj = torch.cat((adj,torch.zeros(MAX_LEN,word_pad,dtype=torch.int)),dim=1)
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

class SCIGraphDataSet(Dataset):
    def __init__(self, path, ent_vocab):
        self.path = path
        self.dirname = os.path.dirname(path)
        self.filename = os.path.basename(path)
        self.data = []
        self.tokenizer =  BertTokenizer.from_pretrained('pretrainedmodel/scibert_scivocab_uncased', do_lower_case =False)
        f = open(path)
        rACL = re.compile("ACLITE")
        if os.path.exists(os.path.join(self.dirname,self.filename[:-4]+".json")):
            fids = open(os.path.join(self.dirname,self.filename[:-4]+".json"))
            dl = json.load(fids)
            self.data = dl
        else:
            for i,line in enumerate(f):
                if i %10000 == 0:
                    print(i)
                l = line.rstrip("\n").split("\t")
                fromcite = l[0]
                tocite = l[2]
                citationcontext = l[1]
                citationcontexts = []
                curs = ""
                citationcontexts_tmp = citationcontext.split()
                for context in citationcontexts_tmp:
                    if context == "targetCITE":
                        if curs != "":
                            citationcontexts.append(curs[:-1])
                        citationcontexts.append(context)
                        curs = ""
                    elif re.search(rACL,context):
                        if curs != "":
                            citationcontexts.append(curs[:-1])
                        citationcontexts.append(context)
                        curs = ""
                    elif context == "CITE-UNKNOWN":
                        if curs != "":
                            citationcontexts.append(curs[:-1])
                        citationcontexts.append(context)
                        curs = ""
                    else:
                        curs += context + " "
                if curs != "":
                    citationcontexts.append(curs)
                citationcontextl = []
                masked_ids = []
                position_ids = []
                token_type_ids = []
                curposition = 0
                citationcontextl.append(self.tokenizer.cls_token_id)
                citationcontextl.append(ent_vocab[fromcite])
                citationcontextl.append(self.tokenizer.sep_token_id)
                masked_ids.extend([-1,-1,-1])
                position_ids.extend([0,1,2])
                token_type_ids.extend([0,1,0])
                curposition = 3
                for citationcontext in citationcontexts:
                    if citationcontext == "targetCITE":
                        citationcontextl.extend([ent_vocab["MASK"]])
                        token_type_ids.extend([1])
                        #masked_ids.extend([1])
                        masked_ids.extend([ent_vocab[tocite]])
                        position_ids.extend([curposition])
                        curposition += 1
                    elif re.search(rACL,context):
                        citationcontextl.extend([ent_vocab[context[7:]]])
                        token_type_ids.extend([1])
                        #masked_ids.extend([0])
                        masked_ids.extend([-1])
                        position_ids.extend([curposition])
                        curposition += 1
                    elif citationcontext == "CITE-UNKNOWN":
                        citationcontextl.extend([0])
                        token_type_ids.extend([1])
                        #masked_ids.extend([0])
                        masked_ids.extend([-1])
                        position_ids.extend([curposition])
                        curposition += 1
                    else:
                        tokenized_context = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(citationcontext))
                        citationcontextl.extend(tokenized_context)
                        token_type_ids.extend([0 for _ in range(len(tokenized_context))])
                        #masked_ids.extend([0 for _ in range(len(tokenized_context))])
                        masked_ids.extend([-1 for _ in range(len(tokenized_context))])
                        position_ids.extend([curposition+i for i in range(len(tokenized_context))])
                        curposition += len(tokenized_context)
                if len(citationcontextl) > MAX_LEN:
                    print("exceed")
                    citationcontextl = citationcontextl[:MAX_LEN]
                    masked_ids = masked_ids[:MAX_LEN]
                    position_ids = position_ids[:MAX_LEN]
                    token_type_ids = token_type_ids[:MAX_LEN]
                if i % 100000 == 0:
                    print(line)
                    print({
                    'input_ids': citationcontextl,
                    'masked_lm_labels' : masked_ids,
                    'position_ids': position_ids,
                    'token_type_ids': token_type_ids,
                    })
                if len(citationcontextl) != len(masked_ids) or len(masked_ids) != len(position_ids) or len(position_ids) != len(token_type_ids):
                    print("size diff")

                self.data.append({
                    'input_ids': citationcontextl,
                    'masked_lm_labels' : masked_ids,
                    'position_ids': position_ids,
                    'token_type_ids': token_type_ids,
                })
            fids = open(os.path.join(self.dirname,self.filename[:-4]+".json"),"w")
            json.dump(self.data,fids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, batch):
        input_keys = ['input_ids','masked_lm_labels',"position_ids","token_type_ids","n_word_nodes","attention_mask"]
        target_keys = ["masked_lm_labels","word_seq_len"]
        max_words = MAX_LEN
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
                adj = torch.cat((adj,torch.zeros(MAX_LEN,word_pad,dtype=torch.int)),dim=1)
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


if __name__ == "__main__":
    path = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/citationcontexts.txt"
    entitypath = "/home/ohagi_masaya/M1/TransBasedCitEmb/dataset/entity.txt"
    fe = open(entitypath)
    ent_vocab = {"UNKNOWN":0,"MASK":1}
    for i,line in enumerate(fe):
        ent_vocab[line.rstrip("\n")] = i+2
