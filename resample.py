import io
import random
with io.open('data/train.txt', encoding="utf8") as finput:
    finput.readline()
    out=[]
    i=0
    for line in finput:
        line=line.strip().split('\t')
        if line[-1]!="happy" and random.random()>0.3:
            continue
        out.append(str(i)+'\t'+'\t'.join(line[1:]))
        i+=1
        
with io.open('data/happy.txt', "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label", "prediction"]) + '\n')
        fout.write('\n'.join(out))        

