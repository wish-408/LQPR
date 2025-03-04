from LQPR import LQPR, init_pattern_vecs
from ML_BOW_BAYES import ML_BOW_BAYES
from ML_BOW_KNN import ML_BOW_KNN
from ML_TF_IDF_BAYES import ML_TF_IDF_BAYES
from ML_TF_IDF_KNN import ML_TF_IDF_KNN
from PRETRAINED_BERT import PRETRAINED_BERT
from LLM_TALK import LLM_TALK
from BERT_TRAIN import BERT_TRAIN
from ZSL import ZSL
import os

# 训练30个bert模型
def bert_train(): 
    BERT_TRAIN()

# 实验1
def RQ1():
    
    config = {  
        'use_inv' : True,
        'use_sync' : True,
        'use_semantic' : True,
        'weight' : 0.7
    }
    
    init_pattern_vecs()
    data_dir = "../dataset/Promise/random_split/" # 随机划分的Promise训练-测试集
    for split_dir in os.listdir(data_dir):
        print(split_dir)

        model_dir = "../bert_models_30/train" + split_dir + "/"
        result_dir = "../expriment_result/Promise/RQ1/"
        train_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
        test_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_test.txt'
            
        LQPR(test_data_path, result_dir, "LQPR", config)
        ML_BOW_BAYES(train_data_path, test_data_path, result_dir, "ML_BOW_BAYES")
        ML_BOW_KNN(train_data_path, test_data_path, result_dir, "ML_BOW_KNN")  
        ML_TF_IDF_BAYES(train_data_path, test_data_path, result_dir, "ML_TF_IDF_BAYES")
        ML_TF_IDF_KNN(train_data_path, test_data_path, result_dir, "ML_TF_IDF_KNN")
        ZSL(test_data_path, result_dir, "ZSL")
        PRETRAINED_BERT(model_dir, test_data_path, result_dir, "PRETRAINED_BERT")
        LLM_TALK(test_data_path, result_dir, "LLM_TALK")
        
def RQ2():
    config = {  
        'use_inv' : True,
        'use_sync' : True,
        'use_semantic' : True,
        'weight' : 0.7
    }
    init_pattern_vecs()
    test_data_dirs = ["PURE", "Shaukat_et_al", "LLM-GEN"]
    for test_data_dir in test_data_dirs:
        print(test_data_dir)
        result_dir = "../expriment_result/" + test_data_dir + "/" + "RQ2/"
        test_data_path = "../dataset/" + test_data_dir + ".txt"
        
        train_data_dirs = "../dataset/Promise/random_split/" 
        for split_dir in os.listdir(train_data_dirs):
            train_data_path = train_data_dirs + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
            model_dir = "../bert_models_30/train" + split_dir + "/"
            
            LQPR(test_data_path, result_dir, "LQPR", config)
            ML_BOW_BAYES(train_data_path, test_data_path, result_dir, "ML_BOW_BAYES")
            ML_BOW_KNN(train_data_path, test_data_path, result_dir, "ML_BOW_KNN")  
            ML_TF_IDF_BAYES(train_data_path, test_data_path, result_dir, "ML_TF_IDF_BAYES")
            ML_TF_IDF_KNN(train_data_path, test_data_path, result_dir, "ML_TF_IDF_KNN")
            ZSL(test_data_path, result_dir, "ZSL")
            PRETRAINED_BERT(model_dir, test_data_path, result_dir, "PRETRAINED_BERT")
            LLM_TALK(test_data_path, result_dir, "LLM_TALK")
            
        
def RQ4():
    configs = [{  
        'use_inv' : True,
        'use_sync' : True,
        'use_semantic' : True,
        'weight' : 0.7
    },
    {  
        'use_inv' : False,
        'use_sync' : True,
        'use_semantic' : True,
        'weight' : 0.7
    },
    {  
        'use_inv' : True,
        'use_sync' : False,
        'use_semantic' : True,
        'weight' : 0.7
    },
    {  
        'use_inv' : True,
        'use_sync' : True,
        'use_semantic' : False,
        'weight' : 0.7
    }]     
    init_pattern_vecs()
    
    
    for config in configs:
        data_dir = "../dataset/Promise/random_split/" # 随机划分的Promise训练-测试集
        for split_dir in os.listdir(data_dir):
            result_dir = "../expriment_result/Promise/RQ4/"
            test_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_test.txt'
            LQPR(test_data_path, result_dir, "LQPR", config)
            
        test_data_dirs = ["PURE", "Shaukat_et_al", "LLM-GEN"]
        for test_data_dir in test_data_dirs:
            result_dir = "../expriment_result/" + test_data_dir + "/" + "RQ4/"
            test_data_path = "../dataset/" + test_data_dir + ".txt"
            LQPR(test_data_path, result_dir, "LQPR", config)

def example():
    config = {
        'use_inv' : True,
        'use_sync' : True,
        'use_semantic' : True,
        'weight' : 0.7
    }
    from LQPR import predicte, init_pattern_vecs
    init_pattern_vecs()
    sentence = "The time taken to add products to the shopping cart must not exceed 2 milliseconds."
    print(predicte(sentence, config))
    
    

if __name__ == '__main__':
    # bert_train()
    # RQ1()
    # RQ2()
    # RQ4()
    # example()
    pass