import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertForMaskedLM, BertConfig
import math
import numpy as np

#input sequence: [citing paper id,left_citation_context,masked cited id, right_citation_context]
class PTBCN(BertForMaskedLM):
    config_class = BertConfig
    base_model_prefix = "bert"
    def __init__(self, config, num_ent, MAX_LEN, final_layer):
        super().__init__(config)
        if final_layer == "feedforward":
            self.ent_lm_head = EntLMHead(config,num_ent)
        else:
            self.ent_lm_head = EntLinearLMHead(config,num_ent)
        self.ent_embeddings = nn.Embedding(num_ent, 768)
        self.MAX_LEN = MAX_LEN

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
            embedding = []
            for j in range(self.MAX_LEN):
                if token_type_id[j] == 0:
                    if input_id[j] != -1:
                        embedding.append(self.bert.embeddings.word_embeddings(input_id[j]))
                    else:
                        embedding.append(self.bert.embeddings.word_embeddings(torch.tensor(0).cuda()))
                else:
                    embedding.append(self.ent_embeddings(input_id[j]))
            input_embed = torch.cat([emb.unsqueeze(0) for emb in embedding],dim = 0)
            input_embeds.append(input_embed)
        input_embeds = torch.cat([embedding.unsqueeze(0) for embedding in input_embeds],dim = 0).cuda()
        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            output_hidden_states=True
        )
        sequence_output = outputs[0]  # batch x seq_len x hidden_size
        outputs_each_layer = outputs[2]
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        ent_logits = self.ent_lm_head(sequence_output)
        ent_predict = torch.argmax(ent_logits, dim=-1)
        ent_masked_lm_loss = loss_fct(ent_logits.view(-1, ent_logits.size(-1)), masked_lm_labels.view(-1))
        loss = ent_masked_lm_loss
        return {'loss': loss,
                'entity_pred': ent_predict,
                'entity_logits': ent_logits,
                'sequence_output': sequence_output,
                'outputs_each_layer': outputs_each_layer}

#input sequence: [citing paper id, frozen SciBERT embeddings of citation context, cited paper id]
class PTBCNCOKE(BertForMaskedLM):
    config_class = BertConfig
    base_model_prefix = "bert"
    def __init__(self, config, num_ent, MAX_LEN):
        super().__init__(config)
        self.ent_lm_head = EntLMHead(config,num_ent)
        self.ent_embeddings = nn.Embedding(num_ent, 768)
        self.MAX_LEN = MAX_LEN

    def change_type_embeddings(self):
        self.config.type_vocab_size = 2
        single_emb = self.bert.embeddings.token_type_embeddings
        self.bert.embeddings.token_type_embeddings = nn.Embedding(2, self.config.hidden_size)
        self.bert.embeddings.token_type_embeddings.weight = torch.nn.Parameter(single_emb.weight.repeat([2, 1]))

    def forward(
            self,
            target_ids=None,
            source_ids=None,
            position_ids=None,
            contexts=None,
            token_type_ids=None,
            attention_mask=None,
            mask_positions=None,
    ):
        input_embeds = []
        masked_lm_labels = []
        mask_id = 1
        for target_id,context,source_id,mask_position in zip(target_ids,contexts,source_ids,mask_positions):
            emb = []
            masked_lm_label = []
            if mask_position == 0:
                emb.append(self.ent_embeddings(torch.tensor(mask_id).cuda()))
                emb.append(context.cuda())
                emb.append(self.ent_embeddings(source_id.cuda()))
                masked_lm_label.append(torch.tensor([target_id,-1,-1]))
            else:
                emb.append(self.ent_embeddings(target_id.cuda()))
                emb.append(context.cuda())
                emb.append(self.ent_embeddings(torch.tensor(mask_id).cuda()))
                masked_lm_label.append(torch.tensor([-1,-1,source_id]))
            input_embed = torch.cat([embedding.unsqueeze(0) for embedding in emb],dim = 0)
            masked_lm_label = torch.cat([label.unsqueeze(0) for label in masked_lm_label],dim=0)
            input_embeds.append(input_embed)
            masked_lm_labels.append(masked_lm_label)
        input_embeds = torch.cat([embedding.unsqueeze(0) for embedding in input_embeds],dim = 0).cuda()
        masked_lm_labels = torch.cat([masked_lm_label.unsqueeze(0) for masked_lm_label in masked_lm_labels],dim = 0).cuda()
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
                'sequence_output': sequence_output,
                'masked_lm_labels': masked_lm_labels}

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

class EntLinearLMHead(nn.Module):
    def __init__(self, config,num_ent):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, num_ent)

    def forward(self, features, **kwargs):
        x = self.decoder(features)
        return x
