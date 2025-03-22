from sentence_transformers import SentenceTransformer,util
import numpy as np
from static_data import *
from memory_profiler import profile
from tools import result_report, load_data, get_date_time
from transformers import BertModel, BertTokenizer



# @profile
def ZSL(test_data_path, result_dir, save_name):
    # Load pre-trained Sentence Transformer Model. It will be downloaded automatically
    model = SentenceTransformer("all-MiniLM-L12-v2")
    
    #建立标签和标签具体内容的映射
    hash={
        "1 1" : "all must be, all of the",
        "1 0" : "100% of ,later 100 ,out of 100 ,longer than 100 ,be avilable 100 ,exceed 100 ,more than 100 ,support 100 ,up to 100 ,at least 100 ,minimum of 100 ,support a maximum of 100 ,after 100 ,be able to handle 100 ,be capable of handling 100 ,achieve 100",
        "1 -1" : "a period of 100 ,on 100 ,be 100 ,approximately 100 ,reamain 100 ,to 100 ,every 100 ,contain the 100",
        "0 -1" : "within 100 ,decreased by 100 ,in under 100 ,less than 100 ,maximum of 100 ,at most 100",
        "-1 -1" : "in an acceptable time ,respond fast to"
    }
    sentences, real_labels = load_data(test_data_path)
    sentence_embeddings = model.encode(sentences)

    predicte_labels=[]
    i = 0
    for st_emb in sentence_embeddings:
        i += 1
        similarity=[]
        keys = []
        for key in hash:
            keys.append(key)
            content=hash[key] #提取标签的具体内容
            lb_emb=model.encode(content)
            cossim = util.cos_sim(st_emb,lb_emb) #计算句向量和标签向量的余弦相似度
            similarity.append(cossim.item()) #存放在一个列表中，注意cossim函数的返回值是一个tensor张量，需要提取其数值
            
        sorted_sml=sorted(zip(keys, similarity), key=lambda x: x[1], reverse=True) #将标签按照余弦相似度排序
        predicte_labels.append(sorted_sml[0][0])#保存预测标签

    save_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'
    result_report("ZSL", save_path, predicte_labels, real_labels)
