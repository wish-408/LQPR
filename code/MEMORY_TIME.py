import time
import os
from memory_profiler import memory_usage
from LLM_TALK import LLM_TALK
from ZSL import ZSL
from LQPR import RUN

from ML_BOW_BAYES import ML_BOW_BAYES
from ML_BOW_KNN import ML_BOW_KNN
from ML_TF_IDF_BAYES import ML_TF_IDF_BAYES
from ML_TF_IDF_KNN import ML_TF_IDF_KNN

from COSINE_SIMILARITY import COSINE_SIMILARITY

from LLM_CLASSIFY import LLM_CLASSIFY

import pandas as pd


if __name__ == '__main__':
    # data_dir = "../test_data/new_test/"
    data_dir = "../test_data/"
    for data_path in os.listdir(data_dir):
        if os.path.isdir(data_dir + data_path):
            continue
        print(f"data path : {data_path}")
        RUN(data_dir + data_path, True, True, True, True)
        
        # ML_BOW_BAYES(data_dir + data_path)
        
        # time_result = [[0 for j in range(10)] for i in range(10)] # 初始化全为0的数组
        
        # time_record_path = "../expriment_result/time/" + data_path.split('.')[0] + ".xlsx"
        
        # #表头
        # header = ["LQPR", "BoW/kNN", "BoW/NB", "TF-IDF/NB", "TF-IDF/kNN", "Bert", "Gemma-27B", "Deepseek-67B", "Llama-8B", "ZSL"]
        # approch_col = {"LQPR" : 0, "BoW/kNN" : 1, "BoW/NB" : 2, "TF-IDF/NB" : 3, "TF-IDF/kNN" : 4, 
        #                "Bert" : 5, "Gemma-27B" : 6, "Deepseek-67B" : 7, "Llama-8B" : 8, "ZSL" : 9}
        
        # #如果excel不存在就新建一个
        # if not os.path.exists(time_record_path):
        #     new_file = pd.DataFrame(columns = header)
        #     new_file.to_excel(time_record_path, header = header, index = False)
        
        # #把原本的文件信息记录
        # df = pd.read_excel(time_record_path)
        # h, w = df.shape
        # for i in range(h):
        #     for j in range(w):
        #         time_result[i][j] = df.iloc[i,j]
                
        # time_used = []
        
        # test_num = 1
        # for i in range(test_num):
            
        #     print(f"test case : {i}")
        
        #     # 记录开始时间
        #     start_time = time.time()

        #     # ML_TF_IDF_KNN('../test_data/' + data_path)
        #     RUN('../test_data/' + data_path, True, True, False)

        #     # 记录结束时间
        #     end_time = time.time()

        #     # 计算时间
        #     elapsed_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
            
        #     time_used.append(elapsed_time_ms)

        #     # 输出结果
        #     print(f"运行时间: {elapsed_time_ms:.2f} 毫秒")
        #     print("")
        #     print("")
            
        # col = 0
        # for i in range(test_num):
        #     time_result[i][col] = time_used[i]
        # time_result = pd.DataFrame(time_result)
        # time_result.to_excel(time_record_path, header=header, index=False)
        
        
