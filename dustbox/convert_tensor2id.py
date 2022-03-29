import csv
import pandas as pd
#csv内のtensorをpaper idに変換する
#target_id,source_id,top5,MRR
#['1610.00479v1'],['1409.3215v1'],"[tensor(892), tensor(796), tensor(1068), tensor(1497), tensor(1438)]",1.0

def build_ent_vocab(path,dataset="AASC"):
    ent_vocab = {"UNKNOWN":0,"MASK":1}
    if dataset == "AASC":
        df = pd.read_csv(path,quotechar="'")
    else:
        df = pd.read_csv(path)
    entitylist = list(set(list(df["source_id"].values)+list(df["target_id"].values)))
    entitylist.sort()
    for i,entity in enumerate(entitylist):
        ent_vocab[entity] = i+2
    return ent_vocab

citation_recommendation_FTPR = "/home/ohagi_masaya/TransBasedCitEmb/dataset/PeerRead/train.csv"
ent_vocab = build_ent_vocab(citation_recommendation_FTPR,dataset="PeerRead")
ent_reverse_vocab = {ent_vocab[key]:key for key in ent_vocab}
df = pd.read_csv("../../results/epoch5_batchsize16_learningrate8e-05_dataPeerRead_WINDOWSIZE125_MAXLEN256_pretrainedmodelscibert_tail_linear_StructureAwareCrossEntropy.csv")
fw = open("../../results/CaseAnalysis_FullTextPeerRead.csv","w")
writer = csv.writer(fw)
writer.writerow(["target_id","source_id","top5","MRR"])
i = 0
for target_id,source_id,top5,MRR in zip(df["target_id"],df["source_id"],df["top5"],df["MRR"]):
    target_id = target_id[2:-2]
    source_id = source_id[2:-2]
    top5 = [ent_reverse_vocab[int(pred[7:-1])] for pred in top5[1:-1].split(", ")]
    i += 1
    if i < 5:
        print(target_id)
        print(source_id)
        print(top5)
    writer.writerow([target_id,source_id,top5,MRR])
