import math

def cohen_d(avi, si, ni, avj, sj, nj):
    up = (ni - 1) * si * si + (nj - 1) * sj * sj
    down = ni + nj - 2
    result = (avi - avj) / math.sqrt(up / down)
    return round(result, 2)  # 使用round函数将结果保留两位小数后返回

def get_av(data) : 
    s = 0.0
    for num in data :
        s += float(num)
        
    return s / len(data)

def get_s(data) :
    av = get_av(data)
    s = 0.0
    for num in data :
        s  = s + (float(num) - float(av)) * (float(num) - float(av))
    return math.sqrt(s / (len(data) - 1))

def trans(s):
    s = s.strip()
    return float(s[1:-1])


hash = {"acc" : [[], [], [], [], [], [], [], [], [], []],
        "pre" : [[], [], [], [], [], [], [], [], [], []],
        "rec" : [[], [], [], [], [], [], [], [], [], []],
        "f1" : [[], [], [], [], [], [], [], [], [], []]}

paths = ["data.txt", "data2.txt", "data3.txt"]

sign = {
        "acc" : [[0 for j in range(10)] for i in range(10)],
        "pre" : [[0 for j in range(10)] for i in range(10)],
        "rec" : [[0 for j in range(10)] for i in range(10)],
        "f1" : [[0 for j in range(10)] for i in range(10)],
}

cohen = {
        "acc" : [[0 for j in range(10)] for i in range(10)],
        "pre" : [[0 for j in range(10)] for i in range(10)],
        "rec" : [[0 for j in range(10)] for i in range(10)],
        "f1" : [[0 for j in range(10)] for i in range(10)],
}

sorted_cohen = {
        "acc" : [[0 for j in range(10)] for i in range(10)],
        "pre" : [[0 for j in range(10)] for i in range(10)],
        "rec" : [[0 for j in range(10)] for i in range(10)],
        "f1" : [[0 for j in range(10)] for i in range(10)],
}

sorted_label = {
    "acc" : [],
    "pre" : [],
    "rec" : [],
    "f1" : [],
}



group = {
    "acc" : [],
    "pre" : [],
    "rec" : [],
    "f1" : [],
} 

labels = [ 
    "LCS",
    "ZSL",
    "BOW+KNN",
    "BOW+BAYES",
    "TFIDF+KNN",
    "TFIDF+BAYES",
    "LLM+model_1",
    "LLM+model_2",
    "LLM+model_3",
    "Pretrained-LLM"]

id_str = {0 : "LCS", 1 : "ZSL", 2 : "BOW+KNN", 
          3 : "BOW+BAYES", 4 : "TFIDF+KNN", 
          5 : "TFIDF+BAYES", 6 : "LLM+model_1", 
          7 : "LLM+model_2", 8 : "LLM+model_3", 9 : "Pretrained-LLM"}

str_id = {
    "LCS": 0,
    "ZSL": 1,
    "BOW+KNN": 2,
    "BOW+BAYES": 3,
    "TFIDF+KNN": 4,
    "TFIDF+BAYES": 5,
    "LLM+model_1": 6,
    "LLM+model_2": 7,
    "LLM+model_3": 8,
    "Pretrained-LLM": 9
}

def read_data ():
    for path in paths :
        print(path)
        with open(path, 'r', encoding="utf-8") as f:
            id = 0
            for line in f:
                line_split = line.split('|')
                # print(line_split)
                hash["acc"][id].append(trans(line_split[2]))
                hash["pre"][id].append(trans(line_split[3]))
                hash["rec"][id].append(trans(line_split[4]))
                hash["f1"][id].append(trans(line_split[5]))
                id += 1
        
def do_group():
    for key in sign:
        for i in range(10): #依次考虑每个数据该分到哪一组
            success = False
            for id in range(len(group[key])):
                if success:
                    break
                
                t = True
                for j in group[key][id] :
                    if sign[key][i][j] == 1 :
                        t = False
                        break
                if t:
                    success = True
                    group[key][id].append(i)
            if not success:
                group[key].append([i])
        
        
def cal_dis():
    threshold = 0.3
    for key in hash:
        datas = hash[key]
        for i in range(10):
            for j in range(i + 1, 10):
                avi = get_av(datas[i])
                avj = get_av(datas[j])
                si = get_s(datas[i])
                sj = get_s(datas[j])
                cohen[key][i][j] = cohen[key][j][i] = cohen_d(avi, si, 3, avj, sj, 3) 
                if (abs(cohen[key][i][j])) > threshold :
                    sign[key][i][j] = 1
                    sign[key][j][i] = 1  
                    
def f(x, key): # 聚类后的组间排序函数
    av = 0.0
    for i in x:
        data = hash[key][i]
        av = av + get_av(data)
    return av / len(x)

def f2(x, key): # 标签在每个测量指标上的排序函数
    data = hash[key][x]
    return get_av(data)

def do_sort():
    for key in group:
        sorted_group = sorted(group[key], key= lambda x: f(x, key), reverse=True)
        group[key] = sorted_group
        
    
    for key in group:
        for g in group[key]:
            for i in range(len(g)):
                g[i] = id_str[g[i]]
                
def do_sort_label() :
    for key in hash:
        sorted_label[key] = sorted(labels,  key= lambda x: f2(str_id[x], key), reverse=True)
        
def do_sort_cohen():
    for key in cohen:
        sorted_label_ = sorted_label[key]
        n = len(sorted_label_)
        for i in range(n):
            for j in range(n):
                k = str_id[sorted_label_[i]]
                u = str_id[sorted_label_[j]]
                sorted_cohen[key][i][j] = cohen[key][k][u]
        
def output(table):
    for key in table :
        print(f'{key}----------------------')
        for v in table[key]:
            print(v)
        print("")     
    
read_data()
cal_dis()
do_group()  
do_sort()   
do_sort_label()
do_sort_cohen()

output(cohen)
output(group)
output(sign)
output(sorted_cohen)
output(sorted_label)

# for key in sorted_label:
for v in sorted_label["f1"]:
    print(f2(str_id[v], "f1"))


# for key in cohen :
#     print(f'{key}----------------------')
#     for v in cohen[key]:
#         print(v)
#     print("")
           

# for key in group:
#     print(f"{key} : -------------------------------------- ")
#     for g in group[key]:
#         print(g)
#     print("")


                
# for key in sign:
#     print(f"{key} : -------------------------------------- ")
#     for s in sign[key]:
#         print(s)
#     print("")
    

    
# for key in sorted_cohen :
#     print(f"{key} : -------------------------------------- ")
#     for v in sorted_cohen[key]:
#         print(v)
#     print("")  