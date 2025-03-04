import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from static_data import *
from tools import *
from memory_profiler import profile
import os
import time


_, real_labels = load_data("../data/new_labeled_data1.txt")
label_encoder = LabelEncoder()
label_encoder.fit_transform(real_labels)
# label_encoder.fit_transform(all_labels)

model_path = "../bert_models/train0"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()

def predicte(sentence) :
  # 设置模型为评估模式
    
    predicte_labels = []
    
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():  # 禁用梯度计算
        outputs = model(**inputs)

    logits = outputs.logits  # 获取模型的输出
    predicted_class = logits.argmax(dim=1)  # 获取预测的类别

    original_label = label_encoder.inverse_transform([predicted_class.item()])
    predicte_labels.append(original_label[0])
    
    return predicte_labels

# @profile
def LLM_CLASSIFY(data_path):
    
    file_name = data_path.split('/')[2]
    
    model_dir = "../bert_models"
    
    time_result = [[0 for j in range(10)] for i in range(10)] # 初始化全为0的数组
        
    time_record_path = "../expriment_result/time/" + file_name.split('.')[0] + ".xlsx"
    
    
    #表头
    header = ["LQPR", "BoW/kNN", "BoW/NB", "TF-IDF/NB", "TF-IDF/kNN", "Bert", "Gemma-27B", "Deepseek-67B", "Llama-8B", "ZSL"]
    approch_col = {"LQPR" : 0, "BoW/kNN" : 1, "BoW/NB" : 2, "TF-IDF/NB" : 3, "TF-IDF/kNN" : 4, 
                    "Bert" : 5, "Gemma-27B" : 6, "Deepseek-67B" : 7, "Llama-8B" : 8, "ZSL" : 9}
    
    #如果excel不存在就新建一个
    if not os.path.exists(time_record_path):
        new_file = pd.DataFrame(columns = header)
        new_file.to_excel(time_record_path, header = header, index = False)
    
    #把原本的文件信息记录
    df = pd.read_excel(time_record_path)
    h, w = df.shape
    for i in range(h):
        for j in range(w):
            time_result[i][j] = df.iloc[i,j]
            
    time_used = []
    
    cnt = 0
    for bert_model in os.listdir(model_dir):  # 测试多个模型
        cnt += 1
        # 记录开始时间
        start_time = time.time()
        model_path = model_dir + '/' + bert_model + '/'
        sentences, real_labels = load_data(data_path)

        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model.eval()  # 设置模型为评估模式
        
        predicte_labels = []
        
        i = 0
        for sentence in sentences:

            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

            with torch.no_grad():  # 禁用梯度计算
                outputs = model(**inputs)

            logits = outputs.logits  # 获取模型的输出
            predicted_class = logits.argmax(dim=1)  # 获取预测的类别

            original_label = label_encoder.inverse_transform([predicted_class.item()])
            predicte_labels.append(original_label[0])
        
        dataset_path = data_path.split('/')[2]
        dataset_path = dataset_path.split('.')[0]
        save_path = "../bert_result/" + get_date_time() + '_' + dataset_path + '.xlsx'
        result_report("Bert_" + bert_model, save_path, predicte_labels, real_labels)
        
    #     # 记录结束时间
    #     end_time = time.time()

    #     # 计算时间
    #     elapsed_time_ms = (end_time - start_time) * 1000  # 转换为毫秒
        
    #     time_used.append(elapsed_time_ms)

    #     # 输出结果
    #     print(f"运行时间: {elapsed_time_ms:.2f} 毫秒")
    #     print("")
    #     print("")
        
    #     if cnt >= 10:
    #         break
            
    # col = 5
    # for i in range(10):
    #     time_result[i][col] = time_used[i]
    # time_result = pd.DataFrame(time_result)
    # time_result.to_excel(time_record_path, header=header, index=False)