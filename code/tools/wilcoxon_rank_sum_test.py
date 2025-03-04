from scipy.stats import wilcoxon
import pandas as pd
import os
from scipy.stats import ranksums
# from tools import get_numbers


    
data_dir = "../expriment_result"
metrixs = ["","w_p", "w_r", "w_f", "maco_precsion", "maco_recall", "maco_f1_score", "mico_precsion", "mico_recall", "mico_f1_score"]

result = []

for file in os.listdir(data_dir):
    data_path = data_dir + '/' + file
    data_path2 = "../bert_result/" + file
    print(data_path)
    
    df = pd.read_excel(data_path)
    df2 = pd.read_excel(data_path2)
    print(df2.shape)
    p = [file.split('.')[0]]
    for i in range(1, 4):
        metrix = metrixs[i]
        sample1 = [df.iloc[0, i] for j in range(30)]
        sample2 = []
        for j in range(30):
            sample2.append(df2.iloc[j, i])
            
        statistic, p_value = ranksums(sample1, sample2)
        
        print(statistic, p_value)
#         print(sample1)
#         print(sample2)
#         # 进行Wilcoxon秩和检验
#         statistic, p_value = wilcoxon(sample1, sample2)

        print(f"Wilcoxon统计量: {statistic}")
        print(f"p - 值: {p_value}")
        p.append(p_value)
    
    
    result.append(p)

result = pd.DataFrame(result)
header = ["dataset", "w_p", "w_r", "w_f"]
result.to_excel("../wilcoxon_rank_sum_test3.xlsx", header=header, index=False)