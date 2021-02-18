import re
import os
import csv
dirpath = "/works/csisv14-3/acl_anthology/AASC_v4.xhtml/xhtml"
dircitepath = "/works/csisv14-3/acl_anthology/AASC_v4.work"
fl = os.listdir(dirpath)
r1 = re.compile("<a href(.*?)anthology(.*?)[view](.*?)>")
rtext = re.compile('data-text="(.*?)"')
rid = re.compile('<p(.*?)id="(.*?)">')
rACL = re.compile("^[A-Z][0-1](.*?)")
fw = open("citationcontexts_union.csv","w")
acln = 0
basedict = {}
todict = {}
nocontextdict = {}
fu = open("citationunion.txt")
uniondict = {}
for line in fu:
    l = line.rstrip("\n")
    uniondict[l] = 1
writer = csv.writer(fw,quotechar="'")
writer.writerow(["source_id","left_citated_text","right_citated_text","target_id"])
for i,filepath in enumerate(uniondict):
    if not(os.path.exists(dirpath+"/"+filepath+".xhtml")):
        continue
    f1 = open(dirpath+"/"+filepath+".xhtml")
    if not(os.path.exists(dircitepath+"/"+filepath+".CITE")):
        continue
    acln += 1
    if acln % 500 == 0:
        print(acln)
        fw.close()
        fw = open("citationcontexts_union.csv","a")
        writer = csv.writer(fw,quotechar="'")
    basedict[filepath] = 1
    if filepath not in nocontextdict:
        nocontextdict[filepath] = {}
    fcite = open(dircitepath+"/"+filepath+".CITE")
    fbody = open(dircitepath+"/"+filepath+".body.tsv")
    rc = re.compile("CITE-(p-((0|1|2|3|4|5|6|7|8|9|-)+))")
    citedict = {}
    bodyl = []
    #本文からCITEを探す
    for i,line in enumerate(fbody):
        l = line.rstrip("\n").split("\t")[2]
        bodyl.append(l)
        if re.search(rc,l):
            citedict[i] = [cite[0] for cite in re.findall(rc,l)]
    citecontextdict = {}
    #citationcontextを作成
    for key in citedict:
        for cite in citedict[key]:
            if cite not in citecontextdict:
                citecontextdict[cite] = []
            leftcitationcontext = ""
            rightcitationcontext = ""
            if key == 0:
                citestart = bodyl[key].find(cite)-6
                citeend = bodyl[key].find(cite) + len(cite)+1
                leftcitationcontext = bodyl[key][:citestart]
                rightcitationcontext = bodyl[key][citeend:]
                for number in range(key+1,min(len(bodyl),key+4)):
                    rightcitationcontext += bodyl[number]
                citecontextdict[cite].append({"left_citated_text":leftcitationcontext,"right_citated_text":rightcitationcontext})
            elif key == len(bodyl)-1:
                citestart = bodyl[key].find(cite)-6
                citeend = bodyl[key].find(cite) + len(cite)+1
                leftcitationcontext = bodyl[key-3]+bodyl[key-2]+bodyl[key-1] + bodyl[key][:citestart]
                rightcitationcontext = bodyl[key][citeend:]
                citecontextdict[cite].append({"left_citated_text":leftcitationcontext,"right_citated_text":rightcitationcontext})
            else:
                citestart = bodyl[key].find(cite)-6
                citeend = bodyl[key].find(cite) + len(cite)+1
                leftcitationcontext = bodyl[key-3]+bodyl[key-2]+bodyl[key-1] + bodyl[key][:citestart]
                rightcitationcontext = bodyl[key][citeend:]
                for number in range(key+1,min(len(bodyl),key+4)):
                    rightcitationcontext += bodyl[number]
                citecontextdict[cite].append({"left_citated_text":leftcitationcontext,"right_citated_text":rightcitationcontext})
    iddict = {}
    lines = re.sub("\n"," ",f1.read())
    rp = re.compile("(<p(.*?)\/p>)")
    ll = [l1[0] for l1 in re.findall(rp,lines)]
    acldict = {}
    #ACLのidと紐付け
    for line in ll:
        l = line.rstrip("\n")
        if re.search(r1,l):
            if not(re.search(rid,l)):
                print(filepath)
                print(l)
            id1 = re.search(rid,l)[2]
            #ここをdictにすべき
            acldict[id1] = re.search(r1,l)[2][1:-3]
    rc = re.compile("\(CITE-(p-((0|1|2|3|4|5|6|7|8|9|-)+))\)")
    rc1 = re.compile("\(CITE-(p-((0|1|2|3|4|5|6|7|8|9|-)+))")
    rc2 = re.compile("CITE-(p-((0|1|2|3|4|5|6|7|8|9|-)+))")
    rmath = re.compile("MATH-((p|w)-((0|1|2|3|4|5|6|7|8|9|-)+))")
    for key in acldict:
        if key in citecontextdict:
            for context in citecontextdict[key]:
                left_citated_text,right_citated_text = context["left_citated_text"],context["right_citated_text"]
                left_citated_text = re.sub(rc," ",left_citated_text)
                left_citated_text = re.sub(rc1," ",left_citated_text)
                left_citated_text = re.sub(rc2," ",left_citated_text)
                left_citated_text = re.sub(rmath,"",left_citated_text)
                left_citated_text = re.sub("'",'"',left_citated_text)
                right_citated_text = re.sub(rc," ",right_citated_text)
                right_citated_text = re.sub(rc1," ",right_citated_text)
                right_citated_text = re.sub(rc2," ",right_citated_text)
                right_citated_text = re.sub(rmath,"",right_citated_text)
                right_citated_text = re.sub("'",'"',right_citated_text)
                writer.writerow([acldict[key],left_citated_text,right_citated_text,filepath])
