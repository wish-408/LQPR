import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from static_data import *
from tools import *
from memory_profiler import profile
import os
import time



# @profile
def PRETRAINED_BERT(model_dir, test_data_path, result_dir, save_name):
    _, all_labels = load_data("../dataset/Promise/promise_all.txt")
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(all_labels) # 固定label_encoder

    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.eval()  # 设置模型为评估模式
    
    sentences, real_labels = load_data(test_data_path)
    predicte_labels = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

        with torch.no_grad():  # 禁用梯度计算
            outputs = model(**inputs)

        logits = outputs.logits  # 获取模型的输出
        predicted_class = logits.argmax(dim=1)  # 获取预测的类别

        original_label = label_encoder.inverse_transform([predicted_class.item()])
        predicte_labels.append(original_label[0])
    
    save_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'
    result_report("Bert", save_path, predicte_labels, real_labels)
        
