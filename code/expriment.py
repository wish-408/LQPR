from LQPR import LQPR, init_pattern_vecs
from ML_BOW_BAYES import ML_BOW_BAYES
from ML_BOW_KNN import ML_BOW_KNN
from ML_TF_IDF_BAYES import ML_TF_IDF_BAYES
from ML_TF_IDF_KNN import ML_TF_IDF_KNN
from PRETRAINED_BERT import PRETRAINED_BERT
from ZSL import ZSL
import os

def RQ1():
    
    config = {  
        'use_inv' : True,
        'use_sync' : True,
        'use_semantic' : True,
        'weight' : 0.7
    }
    
    init_pattern_vecs()
    data_dir = "../dataset/Promise/random_split/" # 随机划分的训练-测试集
    for split_dir in os.listdir(data_dir):
        print(split_dir)

        model_dir = "../bert_models_30/train" + split_dir + "/"
        result_dir = "../expriment_result/Promise/RQ1/"
        train_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
        test_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_test.txt'
            
        LQPR(test_data_path, result_dir, "LQPR", config)
        # ML_BOW_BAYES(train_data_path, test_data_path, result_dir, "ML_BOW_BAYES")
        # ML_BOW_KNN(train_data_path, test_data_path, result_dir, "ML_BOW_KNN")  
        # ML_TF_IDF_BAYES(train_data_path, test_data_path, result_dir, "ML_TF_IDF_BAYES")
        # ML_TF_IDF_KNN(train_data_path, test_data_path, result_dir, "ML_TF_IDF_KNN")
        # PRETRAINED_BERT(model_dir, test_data_path, result_dir, "PRETRAINED_BERT")
        # ZSL(test_data_path, result_dir, "ZSL")
        break
        
def RQ2():
    
    init_pattern_vecs()
    data_dir = "../dataset/Promise/random_split/" # 
    for split_dir in os.listdir(data_dir):
        print(split_dir)

        result_dir = "../expriment_result/Promise/"
        train_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
        test_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_test.txt'
        
        result_dir = "../expriment_result/PURE_split/"
        # result_dir = "../expriment_result/Shaukat_et_al_split/"
        # train_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
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
        
RQ1()