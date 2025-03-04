from LQPR import LQPR, init_pattern_vecs
# from ML_BOW_BAYES import ML_BOW_BAYES
# from ML_BOW_KNN import ML_BOW_KNN
# from ML_TF_IDF_BAYES import ML_TF_IDF_BAYES
# from ML_TF_IDF_KNN import ML_TF_IDF_KNN
# from PRETRAINED_BERT import PRETRAINED_BERT
# from ZSL import ZSL
import os
from tools import load_data, get_date_time, write_to_excel
from static_data import *
import random
import time

init_pattern_vecs()
test_data_dir = "../test_data/promise_all/random_split/"
for split_dir in os.listdir(test_data_dir):
    print(split_dir)
    # 记录函数开始前的时间
    start_time = time.perf_counter()
    # result_dir = "../expriment_result/promise_split/"
    # train_data_path = test_data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
    # test_data_path = test_data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_test.txt'
    
    result_dir = "../expriment_result/PURE_split/"
    # result_dir = "../expriment_result/Shaukat_et_al_split/"
    # train_data_path = test_data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
    test_data_path = "../test_data/PURE.txt"
    # test_data_path = "../test_data/Shaukat_et_al.txt"
        
    # model_dir = "../bert_models_30/train" + split_dir + "/"
    LQPR(test_data_path, result_dir, "LQPR", True, True, True)
    # ML_BOW_BAYES(train_data_path, test_data_path, result_dir, "ML_BOW_BAYES")
    # ML_BOW_KNN(train_data_path, test_data_path, result_dir, "ML_BOW_KNN")  
    # ML_TF_IDF_BAYES(train_data_path, test_data_path, result_dir, "ML_TF_IDF_BAYES")
    # ML_TF_IDF_KNN(train_data_path, test_data_path, result_dir, "ML_TF_IDF_KNN")
    # PRETRAINED_BERT(model_dir, test_data_path, result_dir, "PRETRAINED_BERT")
    # ZSL(test_data_path, result_dir, "ZSL")
    # 记录函数结束后的时间
    end_time = time.perf_counter()
    # 计算函数执行时间，单位转换为毫秒
    execution_time = (end_time - start_time) * 1000
    print(f"函数执行时间: {execution_time:.2f} ms")
    break
    # i += 1
    # if i > 5:
    #     break
    
