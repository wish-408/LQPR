import pandas as pd
def calculate_mean(data):
    """
    计算给定数据列表的均值，并将结果四舍五入保留两位小数。

    :param data: 包含数值的列表
    :return: 四舍五入保留两位小数后的均值
    """
    if not data:
        return 0
    total = sum(data)
    length = len(data)
    return round(total / length, 2)


def calculate_std_var(data):
    """
    计算给定数据列表的标准差，并将结果四舍五入保留四位小数。

    :param data: 包含数值的列表
    :return: 四舍五入保留四位小数后的标准差
    """
    if len(data) <= 1:
        return 0
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return round(std_dev, 4)


data_path = "../expriment_result/LLM-GEN_split/20250213_LQPR.xlsx"
target_path = "../expriment_result/LLM-GEN_split/only_lcs/avg_std.xlsx"
res_path = "../expriment_result/LLM-GEN_split/lcs_semantic/avg_std.xlsx"
df1 = pd.read_excel(data_path)
df2 = pd.read_excel(target_path)

res = []
h, w = df2.shape
for i in range(h):
    x = []
    if df2.iloc[i, 0] == "LQPR":
        continue
    for j in range(w):
        
        x.append(df2.iloc[i, j])
    res.append(x)

row = ["LQPR"]
h, w = df1.shape
for i in range(1, w):
    x = []
    for j in range(h):
        x.append(df1.iloc[j, i])
    row.append((calculate_mean(x), calculate_std_var(x)))
    
res.append(row)
res = pd.DataFrame(res)
header = ["approch", "w_p", "w_r", "w_f", "maco_precsion", "maco_recall", "maco_f1_score", "mico_precsion", "mico_recall", "mico_f1_score"]

res.to_excel(res_path, header=header, index=False)