from tools import *
from static_data import *
import matplotlib.pyplot as plt
from memory_profiler import profile
import random
import spacy
import numpy as np

languagePatterns=[]
labels=[]
pattern_vecs = []

# 词向量表
word2vec = {}
# 加载 spaCy 的英文模型
nlp = spacy.load("en_core_web_sm" , disable=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

def cosine_similarity(vector1, vector2):
    # 对向量进行 L2 归一化
    vector1_normalized = vector1 / np.linalg.norm(vector1)
    vector2_normalized = vector2 / np.linalg.norm(vector2)
    
    # 计算归一化后向量的点积（即余弦相似度）
    cosine_sim = np.dot(vector1_normalized, vector2_normalized)
    return (cosine_sim + 1) / 2 # 将余弦相似度映射到 [0, 1] 区间
    

def sentence_vector(sentence):
    
    doc = nlp(sentence)
    vectors = [token.vector for token in doc if token.has_vector]
    if not vectors:
        # 返回一个与正常结果形状相同的默认向量
        default_vector = np.ones((96,))
        return default_vector
    return np.mean(vectors, axis=0)


# 求语义相似度
def semantic_smilarity(vecs, vec2):
    res = 0
    for vec1 in vecs:
        res = max(res, cosine_similarity(vec1, vec2))
    return res

def get_split_sentence_vec(word_list) : 
    res = []
    for substr_len in range(2, 6): # 取不同长度的子串
        for i in range(1, len(word_list)): 
            if i + substr_len > len(word_list):
                break
            sub_str = ' '.join(word_list[i : i+substr_len])
            res.append(sentence_vector(sub_str))
            # print(sub_str)
            # res.append(np.mean(vec_list[i: i+substr_len], axis=0))
    return res
        

def init_pattern_vecs():
    full_languagePatterns = open('../pattern/patterns.txt','r',encoding='utf-8').read().split('\n')
    for pattern in full_languagePatterns:
        languagePatterns.append(get_word(pattern.split('$')[0]))
        labels.append(pattern.split('$')[1])
        
    for pattern in languagePatterns:
        print(' '.join(pattern[1: ]))
        pattern_vecs.append(sentence_vector(' '.join(pattern[1: ])))

    print("pattern_vecs初始化完成")
        
#将sentence与所有的pattern进行词形和语义匹配
def matching(sentence, config): 
    use_sync = config['use_sync']
    use_semantic = config['use_semantic']
    weight = config['weight']
    possible_result = []
    num = len(languagePatterns)
    word_list = get_word(sentence)
    for i in range(num):
        pattern = languagePatterns[i]
        LCS, score = lcs(word_list,pattern)
        if len(pattern) == 3:
            if len(LCS) > 0:
                score = get_punishment_score(LCS[-1] - LCS[0] + 1, 2, score)
                
         # 只使用语义匹配        
        if not use_sync:
            lcs_str = ' '.join(word_list[index] for index in LCS)
            lsc_vec = sentence_vector(lcs_str)
            score = cosine_similarity(lsc_vec, pattern_vecs[i])
        
        # 词形相似度 + 语义相似度
        if use_semantic and use_sync:
            lcs_score = score
            lcs_str = ' '.join(word_list[index] for index in LCS)
            lsc_vec = sentence_vector(lcs_str)
            sem_score = cosine_similarity(lsc_vec, pattern_vecs[i])
            score = weight * lcs_score + (1 - weight) * sem_score
            
        possible_result.append((LCS, pattern, labels[i], score, i))
        possible_result=sorted(possible_result, key = key_function, reverse=True)

    return possible_result

#核心函数，求各种预测信息
def predicte(big_sentence, config):
    use_inv = config['use_inv']
    score = 0
    Changing_trend = ''
    matched_pattern = ''
    negative_word = 'no negative word'
    big_sentence = big_sentence.lower()
                    
    possible_result = matching(big_sentence, config) 
    
    if possible_result[0][4] == 0: # 和 100 % of 匹配
        if possible_result[0][3] == possible_result[1][3]:
            if possible_result[0][0][0] == possible_result[1][0][-1]: #对 100 % of 进行修饰
                possible_result[0] = possible_result[1] #修饰语为大

    
    result = possible_result[0]
    Changing_trend = result[2]
    negative_word = is_passive(big_sentence)
    
    if use_inv:
        if  negative_word != 'no negative word' : # 语义反转
                if result[4] != 0: # 100 % of 不需要语义反转
                    Changing_trend = label_inv[Changing_trend] 

            
    score = result[3]                 
    matched_pattern = result[1]
    
    return (Changing_trend, list2str(matched_pattern[1:]), score, negative_word)

    
# @profile
def LQPR(test_data_path, result_dir, save_name, config):
    
    use_inv = config['use_inv']
    use_sync = config['use_sync']
    use_semantic = config['use_semantic']

    if not use_inv:
        print("no label inv")
    if not use_sync:
        print("no syntactic matching")
    if not use_semantic:
        print("no semantic matching")
    
    sentences, real_labels = load_data(test_data_path)
    
    predicte_labels = []

    i = 0
    for big_sentence in sentences:
        
        Changing_trend, matched_pattern, score, negative_word = predicte(big_sentence, config)

        i += 1
        predicte_labels.append(Changing_trend)

    if not use_inv:
        save_path = result_dir + get_date_time() + '_' + 'LQPR_L' + '.xlsx'    
        result_report("LQPR-L", save_path, predicte_labels, real_labels)  
    elif not use_sync:
        save_path = result_dir + get_date_time() + '_' + 'LQPR_se' + '.xlsx'    
        result_report("LQPR-se", save_path, predicte_labels, real_labels)
    elif not use_semantic:
        save_path = result_dir + get_date_time() + '_' + 'LQPR_sy' + '.xlsx'    
        result_report("LQPR-sy", save_path, predicte_labels, real_labels)
    else:
        save_path = result_dir + get_date_time() + '_' + 'LQPR' + '.xlsx'    
        result_report("LQPR", save_path, predicte_labels, real_labels)

