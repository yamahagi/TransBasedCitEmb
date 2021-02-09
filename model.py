import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, BertConfig
import math
import numpy as np

class PTBCN(BertForMaskedLM):
    config_class = BertConfig
    base_model_prefix = "bert"
    def __init__(self, config, num_ent, ent_lr, MAX_LEN):
        super().__init__(config)
        self.ent_lm_head = EntLMHead(config,num_ent)
        self.ent_embeddings = nn.Embedding(num_ent, 768, padding_idx=0)
        self.MAX_LEN = MAX_LEN
        #self.apply(self._init_weights)

    def change_type_embeddings(self):
        self.config.type_vocab_size = 2
        single_emb = self.bert.embeddings.token_type_embeddings
        self.bert.embeddings.token_type_embeddings = nn.Embedding(2, self.config.hidden_size)
        self.bert.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            masked_lm_labels=None,
    ):
        input_embeds = []
        for i,b in enumerate(input_ids):
            input_id = input_ids[i]
            token_type_id = token_type_ids[i]
            emb = []
            for j in range(self.MAX_LEN):
                if token_type_id[j] == 0:
                    if input_id[j] != -1:
                        emb.append(self.bert.embeddings.word_embeddings(input_id[j]))
                    else:
                        emb.append(self.bert.embeddings.word_embeddings(torch.tensor(0).cuda()))
                else:
                    emb.append(self.ent_embeddings(input_id[j]))
            input_embed = torch.cat([embedding.unsqueeze(0) for embedding in emb],dim = 0)
            input_embeds.append(input_embed)
        input_embeds = torch.cat([embedding.unsqueeze(0) for embedding in input_embeds],dim = 0).cuda()
        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
        )
        sequence_output = outputs[0]  # batch x seq_len x hidden_size
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        ent_logits = self.ent_lm_head(sequence_output)
        ent_predict = torch.argmax(ent_logits, dim=-1)
        ent_masked_lm_loss = loss_fct(ent_logits.view(-1, ent_logits.size(-1)), masked_lm_labels.view(-1))
        loss = ent_masked_lm_loss
        return {'loss': loss,
                'entity_pred': ent_predict,
                'entity_logits': ent_logits,
                'sequence_output': sequence_output}


class BertLayerNorm(nn.Module):
    """LayerNormalization層 """

    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps
    def forward(self, x):
        # meanのdimに(単語数の軸インデックス, 特徴量ベクトルの軸インデックス)を与えるように修正しています。)
        u = x.mean(dim=(1, 2), keepdim=True)
        s = (x - u).pow(2).mean(dim=(1, 2), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

def CrossEntropy(ent_logits_batch,masked_lm_labels_batch):
    l = len(ent_logits[0])
    for ent_logits,masked_lm_labels in zip(ent_logits_batch,masked_lm_labels):
        if masked_lm_label == -1:
            continue
        else:
            for i,ent in enumerate(ent_logit):
                if i == masked_lm_label:
                    loss += 1
                else:
                    loss += 1


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initialy created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class EntLMHead(nn.Module):
    def __init__(self, config,num_ent):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, num_ent, bias=False)
        self.dropout = nn.Dropout()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.decoder(x)

        return x
