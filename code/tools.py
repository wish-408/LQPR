#本模块封装了各种工具函数

import numpy as np
from datetime import datetime
from random import randint
from static_data import *
import pandas as pd
import os
from datetime import datetime

#否定词汇列表
negative_words = open('../pattern/negative_word.txt', 'r', encoding = 'utf-8').read().split('\n')

#用于修饰的情态动词
qualifier=["shall","must","shuold","will", "be able to","have", "can"]

word_num = {
    'one':'1',
    'two':'2',
    'three':'3',
    'four':'4',
    'five':'5',
    'six':'6',
    'seven':'7',
    'eight':'8',
    'nine':'9',
    'ten':'10'
}

#划分训练集和测试集
def random_move(path,path1,path2):
    train_sentences=[]
    test_sentences=[]
    # 打开文件
    with open(path, 'r',encoding='ansi') as f:#注意txt的编码格式
        # 逐行读取文件内容
        for line in f:
            sign = randint(0,2)
            if(sign<2):
                train_sentences.append(line)

            else :
                test_sentences.append(line)

    with open(path1, 'w',encoding='ansi') as file1:
        for line in train_sentences:
            file1.write(line)  
        
    with open(path2, 'w',encoding='ansi') as file2:
        for line in test_sentences:
            file2.write(line)              

#判断一个单词是否是数字
def is_num(x):
    return x[0].isdigit()


#判断两个字符串是否相等
def equal(a,b): 
    
    # 大写变小写
    a = a.lower()
    b = b.lower()
    
    # 数字单词变数字
    if a in word_num:
        a = word_num[a]
    if b in word_num:
        b = word_num[b]
        
    if(is_num(a) and is_num(b)): #都是数字
        return True
   
    if a in qualifier and b in qualifier: # 都是情态动词
        return True
        
    if a == b :
        return True
    return False

#求最长公共子序列的具体内容
def get_lcs_content(sequence,l,r,trans,LCS):
    if(l==0 or r==0):return
    if(trans[l][r]==1):
        get_lcs_content(sequence,l-1,r-1,trans,LCS)
        LCS.append(sequence[l])
    elif(trans[l][r]==2):
        get_lcs_content(sequence,l-1,r,trans,LCS)
    else: get_lcs_content(sequence,l,r-1,trans,LCS)

#求最长公共子序列中每个单词的位置索引
def get_lcs_location(sequence,l,r,trans,LCS):
    if(l==0 or r==0):return
    if(trans[l][r]==1):
        get_lcs_location(sequence,l-1,r-1,trans,LCS)
        LCS.append(l)
    elif(trans[l][r]==2):
        get_lcs_location(sequence,l-1,r,trans,LCS)
    else: get_lcs_location(sequence,l,r-1,trans,LCS)

#求两个句子的最长公共子序列    
def lcs(list1,list2): 
    n=len(list1)
    m=len(list2)
    dp=[[0 for _ in range(m+1)] for _ in range(n+1)]
    trans=[[0 for _ in range(m+1)] for _ in range(n+1)]

    for i in range(n): dp[i][0] = 1
    for i in range(m): dp[0][i] = 1
    for i in range(1,n):
        for j in range(1,m):
            if equal(list1[i],list2[j]):
                dp[i][j]=dp[i-1][j-1]+1
                trans[i][j]=1
            else: 
                if(dp[i-1][j]>=dp[i][j-1]):
                    trans[i][j]=2
                else:
                    trans[i][j]=3
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
    x = 0 
    y = 0
    for i in range(n):
        for j in range(m):
            if dp[i][j] > dp[x][y]:
                x = i
                y = j
            
    LCS=[]
    get_lcs_location(list1,x,y,trans,LCS)
    
    return LCS,(dp[n-1][m-1] - 1)/(len(list2) - 1)


#将句子分割为单词列表
def get_word(s):
    res=[' ']
    for word in s.split(' '):
        if word=='':
            continue
        if is_num(word): # 将数字都替换成100
            if '%' in word: # 特殊处理一下数字中带百分号的情况
                res.append('100')
                res.append('percent')
            else:
                res.append('100')
            continue
        
        res.append(word)
    return res

 
#根据最长公共子序列的紧凑程度进行罚分   
def get_punishment_score(loc_delta,len_pattren,score): # 如果远距离间隔匹配会被罚得分
    if loc_delta <= len_pattren: k = 1
    else : k = len_pattren/loc_delta
    return k * score
  
#排序函数  
def key_function(result):
    LCS = result[0]
    pattern = result[1]
    if len(LCS):
        loc_delta = LCS[-1] - LCS[0] + 1
    else : loc_delta = 0
    return (result[3],get_punishment_score(loc_delta, len(pattern) - 1, result[3]),len(pattern))    
    #按照最长公共子序列得分、惩罚得分、pattern规模来排序


#找到一句需求描述的数字阈值
def get_number(sequence):
    for word in sequence:
        if word[0].isdigit():
            return word
    return "none"


#暴力匹配（可用kmp优化）一个句子中是否包含某个短语
def match(sentence1, sentence2):
    n = len(sentence1)
    m = len(sentence2)
    for i in range(1, m) :
        j = 1
        k = i
        while(j < n and k < m and equal(sentence1[j], sentence2[k])):
            j += 1
            k += 1
        if j == n : return True
        
    return False

#检测语义反转
def is_passive(text): #text是一个单词列表
    text = get_word(text)
    for keyword in negative_words :
        if match(get_word(keyword), text):
            return keyword
    return 'no negative word'

#获取当前时间
def get_formatted_time():
    now = datetime.now()

    # 格式化当前时间为字符串
    formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
    
    return formatted_time

#将一列表中的词汇拼接成字符串
def list2str(ls):
    res = ''
    for word in ls:
        res = res + ' ' + word
    return res 


def load_data(data_path):
    full_sentences = open(data_path,'r',encoding='utf-8').read().split('\n')
    sentences = []
    real_labels = []
    for sentence in full_sentences:
        try:
            sentences.append(sentence.split('@')[1])
            label = sentence.split('@')[2]
            real_labels.append(label)
        except:
            continue
    return sentences, real_labels


def write_to_excel(path, data):
    
    if not os.path.exists(path):
        new_file = pd.DataFrame()
        new_file.to_excel(path, header = True, index = False)
        
    result = []
    df = pd.read_excel(path)
    h,w = df.shape
    for i in range(h):
        row = []
        for j in range(w):
            row.append(df.iloc[i,j])
        result.append(row)
    result.append(data)
    result = pd.DataFrame(result)
    header = ["approch", "w_p", "w_r", "w_f"]
    result.to_excel(path, header=header, index=False)

def format_number(number):
    rounded_number = round(number, 3)
    formatted_number = "{:.3f}".format(rounded_number)
    return formatted_number

def result_report(approch, save_path, predicte_labels, real_labels):
    
    TP = [0 for i in range(9)] # 真正例数：实际类别为 i 的样本被正确预测为 i 的数量。
    FN = [0 for i in range(9)] # 假负例数：实际类别为 i 的样本被错误预测为其他类别的数量。
    FP = [0 for i in range(9)] # 假正例数：实际类别不为 i 但被预测成 i 的样本数量
    num = [0 for i in range(9)] # 各种样本的数量
    
    n = len(real_labels)
    
    correct = 0 # 预测正确的样本数量
    
    for i in range(n):
    
        Changing_trend = predicte_labels[i]
        if Changing_trend == real_labels[i]:
            correct += 1
            id = label_encode[Changing_trend]
            TP[id] += 1
        else: 
            id = label_encode[real_labels[i]] #实际类别为 i的样本被错误预测为其他类别
            FN[id] += 1
            id = label_encode[Changing_trend]
            FP[id] += 1
    

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
        

        # print(f"* precision of label {label_decode[i]} : {precision} ")
        # print(f"* recall of label {label_decode[i]} : {recall} ")
        # print(f"* F1_score of label {label_decode[i]} : {F1_score} ")
        
        
    d = {}
    sum = 0
    for label in real_labels:
        if label in d:
            d[label] += 1
        else : d[label] = 1
        sum += 1
    
    w_p = 0
    w_r = 0
    w_f = 0   
    for label in d:
        i = label_encode[label]
        w_p += precisions[i] * d[label] / sum
        w_r += recalls[i] * d[label] / sum
        w_f += F1_scores[i] * d[label] /sum
        
    
    maco_recall = 0
    maco_precsion = 0
    maco_f1_score = 0  
    
    label_num = 0
    for label in d:
        label_num += 1
        i = label_encode[label]
        maco_recall += recalls[i] 
        maco_precsion += precisions[i] 
        maco_f1_score += F1_scores[i] 
    
    print("label_num : ", label_num)
    maco_recall /= label_num
    maco_precsion /= label_num
    maco_f1_score /= label_num


    
    for i in range(9):
        TP[0] += TP[i]
        FN[0] += FN[i]
        FP[0] += FP[i]
        
    mico_recall = TP[0] / (TP[0] + FN[0])
    mico_precsion = TP[0] / (TP[0] + FP[0])
    mico_f1_score =  2 * (mico_precsion * mico_recall) / (mico_precsion + mico_recall)   
    
    # print(f"* accurucy : {correct / n}")
    
    print(f"* w_precision : {w_p}")
    print(f"* w_recall : {w_r}")
    print(f"* w_F1_score : {w_f}")
    print("-----------------------------------") 
    
    # print(f"* maco_precsion  : {maco_precsion} ")
    # print(f"* maco_recall : {maco_recall} ")
    # print(f"* maco_f1_score : {maco_f1_score} ")  
    # print("-----------------------------------")

    # print(f"* mico_precsion  : {mico_precsion} ")
    # print(f"* mico_recall : {mico_recall} ")
    # print(f"* mico_f1_score : {mico_f1_score} ")
    
    row_data = [approch, format_number(w_p), format_number(w_r), format_number(w_f)]
    
    write_to_excel(save_path, row_data)
    

def get_date_time():
    current_date = datetime.now()
    return current_date.strftime('%Y%m%d')

def write_log(log_path, log_msg):
    with open(log_path, 'a', encoding="utf-8") as f :

        for line in log_msg: # 逐行写日志
            f.write(line + '\n')
        f.write("---------------------------" + '\n')
        f.write("\n")
            
    f.close()
    

    

    
    
    

