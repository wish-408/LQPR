from tools import write_to_excel
import os
from tools import *

# path = "../data/exprimental.txt"
# data = []
# with open(path, 'r', encoding="utf-8") as f:
#     id = 0
#     for line in f :
#         id += 1
#         new_line = str(id) + '@' + line.split('@')[1] + '@' + line.split('@')[2]
#         data.append(new_line)
# f.close()
# with open(path,'w',encoding="utf-8") as f:
#     for line in data:
#         f.write(line)
# f.close()


# 统计句子长度分布-------------------------------------------------------------------------------
def st_words():
    data_path = "../data/exprimental.txt"
    sentences, real_labels = load_data(data_path)

    cnt_hash = {}

    for sentence in sentences: 
        word_list = get_word(sentence)
        num = len(word_list)
        if num in cnt_hash:
            cnt_hash[num] += 1
        else : 
            cnt_hash[num] = 1

    x = []
    y = []
    for key in cnt_hash:
        x.append(key)

    x = sorted(x)
    cnt = 0
    for num in x :
        if num <= 25 :
            cnt += cnt_hash[num]
        y.append(cnt_hash[num])

    for i in range(len(x)):
        print(f"({x[i]},{y[i]})")
    print(cnt / len(sentences))


#统计阈值数量分布---------------------------------------------------------
def st_threshold_num():
    data_path = "../data/exprimental.txt"
    sentences, real_labels = load_data(data_path)
    zero = 0
    one = 0
    two = 0
    for sentence in sentences: 
        word_list = get_word(sentence)
        cnt = 0
        for word in word_list :
            if is_num(word):
                cnt += 1
        if cnt == 0:
            zero += 1
        elif cnt == 1 : 
            one += 1
        else :
            print(sentence)
        
    print(f"zero : {zero}, one : {one}, two : {len(sentences) - zero - one}")

    # zero : 35, one : 177, tow : 47

#统计type分布------------------------------------------------
def st_type():
    log_path = f"../logs/{get_date_time()}_log.txt"
    data_path = "../data/exprimental.txt"
    sentences, real_labels = load_data(data_path)
    tp1 = tp2 = tp3 = tp4 = 0
    
    for i in range(len(sentences)):
        label = real_labels[i]
        if label == "0 -1":
            tp1 += 1
        elif label == "1 0":
            tp2 += 1
        elif label == "-1 -1" or label == "1 1":
            tp3 += 1
            
        else:
            log_msg = [
            "requirement : " + sentences[i],
            "real label : " + label
            ]
            write_log(log_path, log_msg)
            tp4 += 1
    
    print(f"Type-I : {tp1}, {tp1 / len(sentences)}")
    print(f"Type-II : {tp2}, {tp2 / len(sentences)}")
    print(f"Type-III : {tp3}, {tp3 / len(sentences)}")
    print(f"N/A : {tp4}, {tp4 / len(sentences)}")
    
st_type()
            

        