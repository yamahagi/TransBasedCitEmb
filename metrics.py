import torch
from fastNLP.core.metrics import MetricBase
from fastNLP.core.utils import _get_func_signature
from fastNLP.core.utils import _get_func_signature
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np
import faiss



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

def Evaluation(ent_logits_batch,masked_lm_labels_batch,source_times_dict,score_per_times):
    ans = 0
    mrr = 0
    MAP = 0
    l = 0
    recallat5 = 0
    recallat10 = 0
    recallat30 = 0
    recallat50 = 0
    true_labels = []
    for ent_logits,masked_lm_labels in zip(ent_logits_batch,masked_lm_labels_batch):
        for i,masked_lm_label in enumerate(masked_lm_labels):
            if masked_lm_label != -1:
                ans = masked_lm_label
                rank_array = torch.argsort(ent_logits[i])
                break
        rank_array = list(rank_array)[::-1]
        if ans not in rank_array[:5000]:
            l += 1
            score_per_times[source_times_dict[ans.item()]].append(0)
            continue
        rank = list(rank_array).index(ans)+1
        mrr += 1/rank
        if rank <= 30:
            MAP += (31-rank)/(30*rank)
        if rank <= 5:
            recallat5 += 1
        if rank <= 10:
            recallat10 += 1
        if rank <= 30:
            recallat30 += 1
        if rank <= 50:
            recallat50 += 1
        l += 1
        score_per_times[source_times_dict[ans.item()]].append(1/rank)
    return MAP,mrr,recallat5,recallat10,recallat30,recallat50,l,score_per_times

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

def load_link_predictio():
    #test_pathからそれぞれのnodeをkeyとしてciteされたnodeのlistをvalueとしたdictを読み込む
    df = pd.read_csv(os.path.join(settings.citation_recommendation_dir,"test.csv"))
    ent_vocab = build_ent_vocab(os.path.join(settings.citation_recommendation_dir,"train.csv"))
    linkdict = defaultdict(set)
    for target_id,source_id in zip(df["target_id"],df["source_id"]):
        linkdict[ent_vocab[target_id]].add(ent_vocab[source_id])
    return linkdict

def link_prediction():
    #link predictionのデータを読み込む
    linkdict = load_link_prediction()
    #それぞれのnodeをkeyとしてciteされたnodeのlistをvalueとしたdictを読み込む
    #それぞれのnodeのembeddingsを読み込む
    paper_embeddings = np.load()
    #dict内のkeyごとにfaissを用いて1001までnodeを最近傍探索
    query_embeddings = []
    y_true = []
    for target_id in linkdict:
        query_embeddings.append(paper_embeddings[target_id])
        y_true.append(list(linkdict[target_id]))
    query_embeddings = np.array(query_embeddings)
    y_true = np.array(y_true)

    #prepare faiss
    d = len(query_embeddings[0])                           # dimension
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(paper_embeddings)                  # add vectors to the index
    print(index.ntotal)

    k = 1000                          # we want to see 4 nearest neighbors
    D, I = index.search(xb[:5], k) # sanity check
    print(I)
    print(D)
    D, I = index.search(xq, k)     # actual search
    #MRRを測る
    print(mrr_metrics(y_true,I,k))
    #MAPを測る
    print(map_metrics(y_true,I,k))
