from sentence_transformers import SentenceTransformer, util
from static_data import *
from tools import load_data

def COSINE_SIMILARITY(data_path):

    languagePatterns=[]
    labels = []
    full_languagePatterns = open('../pattern/pattern4.txt','r',encoding='utf-8').read().split('\n')

    for pattern in full_languagePatterns:
        languagePatterns.append(pattern.split('$')[0])
        labels.append(pattern.split('$')[1])

    # # Load pre-trained Sentence Transformer Model. It will be downloaded automatically
    model = SentenceTransformer("all-MiniLM-L12-v2")
        
    sentences, real_labels = load_data(data_path)
    

    sentence_embeddings = model.encode(sentences)

    prediction=[]
    i = 0
    for st_emb in sentence_embeddings:
        # i += 1
        # print(f"test case {i}")
        similarity=[]
        for pattern in languagePatterns:
            content=pattern #提取标签的具体内容
            lb_emb=model.encode(content)
            cossim = util.cos_sim(st_emb,lb_emb) #计算句向量和标签向量的余弦相似度
            similarity.append(cossim.item()) #存放在一个列表中，注意cossim函数的返回值是一个tensor张量，需要提取其数值
            
        sorted_sml=sorted(zip(labels, similarity), key=lambda x: x[1], reverse=True) #将标签按照余弦相似度排序
        prediction.append(sorted_sml[0][0])#保存预测标签

        
    num = len(sentences)
    cnt = 0
    for i in range(num):
        if real_labels[i] == prediction[i]:
            cnt += 1
            id = label_encode[prediction[i]]
            TP[id] += 1
        else:
            id = label_encode[real_labels[i]] #实际类别为 i的样本被错误预测为其他类别
            FN[id] += 1
            id = label_encode[prediction[i]]
            FP[id] += 1
        
    print(f"accuracy : {cnt / num}") 

    recalls = []
    precisions = []
    F1_scores = []


    for i in range(9):
        if TP[i] + FN[i] == 0: 
            recall = 0
        else : 
            recall = TP[i] / (TP[i] + FN[i])
        if TP[i] + FP[i] == 0:
            precision = 0
        else :
            precision = TP[i] / (TP[i] + FP[i])
        
        if recall + precision == 0:
            F1_score = 0
        else :
            F1_score = 2 * (precision * recall) / (precision + recall)       
            
        precisions.append(precision)
        recalls.append(recall)
        F1_scores.append(F1_score)
        print(f"* precision of label {label_decode[i]} : {precision} ")
        print(f"* recall of label {label_decode[i]} : {recall} ")
        print(f"* F1_score of label {label_decode[i]} : {F1_score} ")
        print("-----------------------------------")
        
    num = {}
    for label in real_labels:
        if label in num:
            num[label] += 1
        else : num[label] = 1
