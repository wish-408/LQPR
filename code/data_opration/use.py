import time
import os
from memory_profiler import memory_usage
from LLM_TALK import LLM_TALK

from ML_TF_IDF_BAYES import ML_TF_IDF_BAYES
from ML_TF_IDF_KNN import ML_TF_IDF_KNN

from tools import load_data, get_date_time, write_log

import LQPR
import ML_BOW_KNN
import ML_BOW_BAYES
import LLM_CLASSIFY

from tools import languagePatterns, labels

# path = "../test_data/Shaukat_et_al.txt"
# ouput_path = "../test_data/Shaukat_et_al2.txt"

# with open(ouput_path, 'w', encoding='utf-8') as f:
#     with open(path, 'r', encoding='utf-8') as file:
#         index = 1
#         for line in file:
#             f.write(str(index)+ '@' + line.split('@')[1] + '@' + line.split('@')[2])
#             index += 1
#         file.close();
#     f.close();


# for i in range(len(languagePatterns)):
#     pt = languagePatterns[i]
#     label = labels[i]
#     s = ''
#     for j in range(1, len(pt)):
#         s = s + pt[j] + ' '
#     print((s, label))

# data_path = "./test.txt"
# data_path = "../test_data/Promise.txt"
data_path = "../test_data/Shaukat_et_al.txt"
# data_path = "../test_data/PURE.txt"

log_path = f"../logs/{get_date_time()}_log.txt"

sentences, labels = load_data(data_path)

for i in range(len(sentences)) :
    res1 = LQPR.predicte(sentences[i], True, True)
    
    if res1[0] != labels[i]:

        log_msg = [
            "index : " + str(i + 1),
            "requirement : " + sentences[i],
            "res of LQPR : " + res1[0],
            "real label : " + labels[i]
        ]
        
        write_log(log_path, log_msg)
    
# for i in range(len(sentences)) :
#     res = ML_BOW_KNN.predicte(sentences[i])
#     print(f"predicte : {res[0]}, real : {labels[i]}")


# res = ML_BOW_BAYES.predicte(sentences)
# for i in range(len(sentences)):
#     if res[i] != labels[i]:
#         print(sentences[i])
#         print(res[i])
#         print(labels[i])
#         print("----------------------")
    
# for i in range(len(sentences)) :
#     res = LLM_CLASSIFY.predicte(sentences[i])
#     print(f"predicte : {res[0]}, real : {labels[i]}")

