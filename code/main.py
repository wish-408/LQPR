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
import time


# Train 30 BERT models
def bert_train(): 
    BERT_TRAIN()

# Experiment 1
def RQ1():
    
    config = {  
        'use_inv': True,
        'use_sync': True,
        'use_semantic': True,
        'weight': 0.7
    }
    
    init_pattern_vecs()
    data_dir = "../dataset/Promise/random_split/"  # Randomly split Promise training-test sets
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
        'use_inv': True,
        'use_sync': True,
        'use_semantic': True,
        'weight': 0.7
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
            
        
def RQ3():
    configs = [{  
        'use_inv': True,
        'use_sync': True,
        'use_semantic': True,
        'weight': 0.7
    },
    {  
        'use_inv': False,
        'use_sync': True,
        'use_semantic': True,
        'weight': 0.7
    },
    {  
        'use_inv': True,
        'use_sync': False,
        'use_semantic': True,
        'weight': 0.7
    },
    {  
        'use_inv': True,
        'use_sync': True,
        'use_semantic': False,
        'weight': 0.7
    }]     
    init_pattern_vecs()
    
    
    for config in configs:
        data_dir = "../dataset/Promise/random_split/"  # Randomly split Promise training-test sets
        for split_dir in os.listdir(data_dir):
            result_dir = "../expriment_result/Promise/RQ3/"
            test_data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_test.txt'
            # LQPR(test_data_path, result_dir, "LQPR", config)
            
        test_data_dirs = ["PURE", "Shaukat_et_al", "LLM-GEN"]
        for test_data_dir in test_data_dirs:
            result_dir = "../expriment_result/" + test_data_dir + "/" + "RQ3/"
            test_data_path = "../dataset/" + test_data_dir + ".txt"
            LQPR(test_data_path, result_dir, "LQPR", config)

def RQ4():
    config = {  
        'use_inv': True,
        'use_sync': True,
        'use_semantic': True,
        'weight': 0.7
    }
    init_pattern_vecs()
    
    model_dir = "../bert_models_30/trainsplit_0/"
    result_dir = "../expriment_result/Promise/RQ1/"
    train_data_path = '../dataset/Promise/random_split/split_0/promise_splited_0_train.txt'
    test_data_path = '../dataset/Promise/random_split/split_0/promise_splited_0_test.txt'
        
    start_time = time.perf_counter()
    LQPR(test_data_path, result_dir, "LQPR", config)
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"LQPR execution time: {execution_time:.2f} ms")
    
    start_time = time.perf_counter()
    ML_BOW_BAYES(train_data_path, test_data_path, result_dir, "ML_BOW_BAYES")
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"ML_BOW_BAYES execution time: {execution_time:.2f} ms")
    
    start_time = time.perf_counter()
    ML_BOW_KNN(train_data_path, test_data_path, result_dir, "ML_BOW_KNN")  
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"ML_BOW_KNN execution time: {execution_time:.2f} ms")
    
    start_time = time.perf_counter()
    ML_TF_IDF_BAYES(train_data_path, test_data_path, result_dir, "ML_TF_IDF_BAYES")
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"ML_TF_IDF_BAYES execution time: {execution_time:.2f} ms")
    
    start_time = time.perf_counter()
    ML_TF_IDF_KNN(train_data_path, test_data_path, result_dir, "ML_TF_IDF_KNN")
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"ML_TF_IDF_KNN execution time: {execution_time:.2f} ms")
    
    start_time = time.perf_counter()
    ZSL(test_data_path, result_dir, "ZSL")
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"ZSL execution time: {execution_time:.2f} ms")
    
    start_time = time.perf_counter()
    PRETRAINED_BERT(model_dir, test_data_path, result_dir, "PRETRAINED_BERT")
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"PRETRAINED_BERT execution time: {execution_time:.2f} ms")
    
    start_time = time.perf_counter()
    LLM_TALK(test_data_path, result_dir, "LLM_TALK")
    # Record time after function ends
    end_time = time.perf_counter()
    # Calculate function execution time, convert to milliseconds
    execution_time = (end_time - start_time) * 1000
    print(f"LLM_TALK execution time: {execution_time:.2f} ms")

def example():
    config = {
        'use_inv': True,
        'use_sync': True,
        'use_semantic': True,
        'weight': 0.7
    }
    from LQPR import predict, init_pattern_vecs  # Note: 'predicte' changed to 'predict' in previous code
    init_pattern_vecs()
    sentence = "The time taken to add products to the shopping cart must not exceed 2 milliseconds."
    print(predict(sentence, config))
    

if __name__ == '__main__':
    # bert_train()
    # RQ1()
    # RQ2()
    # RQ3()
    # RQ4()
    # example()
    pass