import torch
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
import time



def mean_reciprocal_rank(true_values, predict_values):
    mrr_result = []
    for true, predict in zip(true_values, predict_values):

        idxs = np.where(true == 1)[0]  ## label value 1 else 0

        rank_array = np.argsort(-predict)

        mrr = []
        for idx in idxs:
            rank = np.where(rank_array == idx)[0] + 1

            mrr.append(rank)

        minor_mrr = min(mrr)
        minor_mrr = (1.0 / minor_mrr)
        mrr_result.append(minor_mrr)

    return np.mean(mrr_result)

def RecallatK(ent_preds,seikail):
    #predl:前から順番にrankが高いentity numberを載せる
    predl = []
    for ent_pred in ent_preds:
        tmpla = np.argsort(ent_pred)
        tmplb = [0]*len(ent_pred)
        for i,ent in enumerate(tmpla):
            tmplb[ent] = i
        predl.append(tmplb)
    Rat5 = 0
    Rat10 = 0
    Rat30 = 0
    Rat50 = 0
    Rat80 = 0
    for seikai,pred in seikail,predl:
        if seikai in pred[:5]:
            Rat5 += 1
        if seikai in pred[:10]:
            Rat10 += 1
        if seikai in pred[:30]:
            Rat30 += 1
        if seikai in pred[:50]:
            Rat50 += 1
        if seikai in pred[:80]:
            Rat80 += 1
    Rat5 /= len(seikail)
    Rat10 /= len(seikail)
    Rat30 /= len(seikail)
    Rat50 /= len(seikail)
    Rat80 /= len(seikail)
    return Rat5,Rat10,Rat30,Rat50,Rat80

def Evaluation(ent_logits_batch,masked_lm_labels_batch):
    ans = 0
    mrr = 0
    MAP = 0
    l = 0
    recallat5 = 0
    recallat10 = 0
    recallat30 = 0
    recallat50 = 0
    for ent_logits,masked_lm_labels in zip(ent_logits_batch,masked_lm_labels_batch):
        for i,masked_lm_label in enumerate(masked_lm_labels):
            if masked_lm_label != -1:
                ans = masked_lm_label
                rank_array = torch.argsort(ent_logits[i])
                break
        rank_array = list(rank_array)[::-1]
        if ans not in rank_array[:5000]:
            l += 1
            continue
        rank = list(rank_array).index(ans)+1
        mrr += 1/rank
        if rank <= 30:
            MAP += 1/rank
        if rank <= 5:
            recallat5 += 1
        if rank <= 10:
            recallat10 += 1
        if rank <= 30:
            recallat30 += 1
        if rank <= 50:
            recallat50 += 1
        l += 1
    return MAP,mrr,recallat5,recallat10,recallat30,recallat50,l

def RecallatK(ent_logits_batch,masked_lm_labels_batch):
    ans = 0
    mrr = 0
    l = 0
    recallat5 = 0
    recallat10 = 0
    recallat30 = 0
    recallat50 = 0
    for ent_logits,masked_lm_labels in zip(ent_logits_batch,masked_lm_labels_batch):
        for i,masked_lm_label in enumerate(masked_lm_labels):
            if masked_lm_label != -1:
                ans = masked_lm_label
                rank_array = torch.argsort(ent_logits[i])[::-1]
                break
        rank = rank_array.index(ans)+1
        if rank <= 5:
            recallat5 += 1
        if rank <= 10:
            recallat10 += 1
        if rank <= 30:
            recallat30 += 1
        if rank <= 50:
            recallat50 += 1
        l += 1
    return recallat5,recallat10,recallat30,recallat50,l
def MeanAveragePrecision(ent_logits_batch,masked_lm_labels_batch):
    ans = 0
    mrr = 0
    l = 0
    MAP = 0
    for ent_logits,masked_lm_labels in zip(ent_logits_batch,masked_lm_labels_batch):
        for i,masked_lm_label in enumerate(masked_lm_labels):
            if masked_lm_label != -1:
                ans = masked_lm_label
                rank_array = torch.argsort(ent_logits[i])
        for i in range(1,len(rank_array)+1):
            rank = rank_array[-i]
            if rank == ans:
                if i <= 30:
                    MAP += 1/i
                    l += 1
                break
    return MAP,l
