from tools import *
from static_data import *
import matplotlib.pyplot as plt
from memory_profiler import profile
import random
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

pattern_vecs = []

# 加载 spaCy 的英文模型
nlp = spacy.load("en_core_web_sm" , disable=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

def get_vec_list(words): # 输入为一个单词列表
    res = []
    for doc in nlp.pipe(words):
        vector = doc[0].vector
        res.append(vector)
    return res

def sentence_vector(sentence):
    doc = nlp(sentence)
    vectors = [token.vector for token in doc if token.has_vector]
    if not vectors:
        return np.zeros(nlp.vocab.vectors.shape[1])
    return np.mean(vectors, axis=0)

def cos_sim(vec1, vec2):
    similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    return similarity

# 求语义相似度
def semantic_smilarity(vecs, vec2):
    res = 0
    for vec1 in vecs:
        res = max(res, cos_sim(vec1, vec2))
    return res

def get_split_sentence_vec(word_list) : 
    res = []
    for substr_len in range(2, 6): # 取不同长度的子串
        for i in range(1, len(word_list)): 
            if i + substr_len > len(word_list):
                break
            sub_str = ' '.join(word_list[i : i+substr_len])
            # print(sub_str)
            res.append(sentence_vector(sub_str))
    return res
        

def init_pattern_vecs():
    for pattern in languagePatterns:
        print(' '.join(pattern[1: ]))
        pattern_vecs.append(sentence_vector(' '.join(pattern[1: ])))
    print("pattern_vecs初始化完成")
        
#将sentence与所有的pattern进行词形和语义匹配
def matching(sentence, use_penalty): 
    possible_result = []
    num = len(languagePatterns)
    word_list = get_word(sentence)
    split_sentence_vecs = get_split_sentence_vec(word_list)
    for i in range(num):
        pattern = languagePatterns[i]
        LCS, score = lcs(word_list,pattern)
        if len(pattern) == 3:
            if len(LCS) > 0:
                score = get_punishment_score(LCS[-1] - LCS[0] + 1, 2, score)
        score = (score + semantic_smilarity(split_sentence_vecs, pattern_vecs[i])) / 2 # 词形相似度 + 语义相似度
        # score =  semantic_smilarity(split_sentence_vecs, pattern_vecs[i])
        possible_result.append((LCS, pattern, labels[i], score, i))
        
    if use_penalty:
        possible_result=sorted(possible_result, key = key_function, reverse=True) # 使用punishment_score
    else : 
        possible_result=sorted(possible_result, key = lambda x : x[3], reverse=True) # 不使用punishment_score
    return possible_result

#核心函数，求各种预测信息
def predicte(big_sentence, use_penalty, use_inv):
    score = 0
    Changing_trend = ''
    matched_pattern = ''
    matched_part = ''
    negative_word = 'no negative word'
    big_sentence = big_sentence.lower()
                    
    possible_result = matching(big_sentence, use_penalty) 
    
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
    matched_part = big_sentence    
    
    return (Changing_trend, list2str(matched_pattern[1:]), score, matched_part, negative_word)
    
def matched(sentence1, sentence2):
    n = len(sentence1)
    m = len(sentence2)
    for i in range(1, m) :
        j = 1
        k = i
        while(j < n and k < m ):
            if is_num(sentence1[j]) and is_num(sentence2[k]) :
                j += 1
                k += 1
            elif sentence1[j] == sentence2[k]:
                j += 1
                k += 1
            else:
                break
        if j == n : return True
        
    return False

    loc = random.randint(0, num-1) # 随机初始化一个位置

#核心函数，求各种预测信息
def predicte2(big_sentence):
        
    Changing_trend = ''
    negative_word = 'no negative word'
    big_sentence = big_sentence.lower()
    sentence = get_word(big_sentence)  
    num = len(languagePatterns)
    loc = 0
    for i in range(num): #匹配所有的pattern
        pattern = languagePatterns[i]
        if(matched(pattern, sentence)):
            loc = i
            break

    Changing_trend = labels[loc]
    negative_word = is_passive(big_sentence)
    if  negative_word != 'no negative word' : # 语义反转
            if loc != 0: # 100 % of 不需要语义反转
                Changing_trend = label_inv[Changing_trend] 
            
    return Changing_trend
    
@profile
def LQPR(test_data_path, result_dir, save_name, use_penalty, use_inv, use_lcs):

    if not use_penalty:
        print("no punishment score")
    if not use_inv:
        print("no label inv")
    if not use_lcs:
        print("no lcs matching")
    
    sentences, real_labels = load_data(test_data_path)
    
    predicte_labels = []

    i = 0
    for big_sentence in sentences:
        
        if use_lcs:
            Changing_trend, matched_pattern, score, matched_seg, negative_word = predicte(big_sentence, use_penalty, use_inv)
            log_path = f"../logs/{get_date_time()}_log.txt"
            log_msg =[
                "dataset : " + test_data_path,
                "requirement : " + big_sentence,
                "res of LQPR : " + Changing_trend,
                "real label : " + real_labels[i],
                "matched pattern : " + matched_pattern,
            ]
            if Changing_trend != real_labels[i]:
                write_log(log_path, log_msg)
        else :
            Changing_trend = predicte2(big_sentence)
            
        i += 1
        predicte_labels.append(Changing_trend)

    save_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'    
    if not use_lcs :
        result_report("LQPR-FM", save_path, predicte_labels, real_labels)
    elif not use_inv:
        result_report("LQPR-L", save_path, predicte_labels, real_labels)
    elif not use_penalty:
        result_report("LQPR-m", save_path, predicte_labels, real_labels)  
    else:
        result_report("LQPR", save_path, predicte_labels, real_labels)
