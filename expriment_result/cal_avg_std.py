import pandas as pd
import os

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
    header = ["approch", "w_p", "w_r", "w_f", "maco_precsion", "maco_recall", "maco_f1_score", "mico_precsion", "mico_recall", "mico_f1_score"]
    result.to_excel(path, header=header, index=False)

data_dir = "./promise_split/"
# data_dir = "./PURE_split/"
# data_dir = "./LLM-GEN_split/"
for file_name in os.listdir(data_dir):
    file_path = data_dir + file_name
    df = pd.read_excel(file_path)
    h, w = df.shape
    approch = df.iloc[0, 0]
    row = [approch]
    for i in range(1, w) : # 依次计算每一列的均值方差
        column = df.iloc[1:h, i]
        mean = round(column.mean(), 2)
        std = round(column.std(), 4)
        row.append((mean, std))
    write_to_excel(data_dir + "avg_std.xlsx", row)
    
