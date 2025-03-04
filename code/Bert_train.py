import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
from tools import load_data

# 转换为Torch Dataset
class SentenceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
def compute_accuracy(p):
    preds, labels = p
    preds = preds.argmax(axis=1)  # 获取预测标签
    return {'accuracy': accuracy_score(labels, preds)}

data_dir = "../test_data/promise_all/random_split/"
_, all_labels = load_data("../test_data/promise_all/promise_all.txt")
label_encoder = LabelEncoder()
label_encoder.fit_transform(all_labels) # 固定label_encoder

for split_dir in os.listdir(data_dir):

    output_path = "../bert_models_30/train" + split_dir
    print("模型保存路径 : ",output_path)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    sentences = []
    real_labels = []
    data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
    print("训练数据集路径 : ", data_path)

    # 加载数据， 此时为列表
    sentences, real_labels = load_data(data_path)
    
    # 使用label_encoder转换为数字标签
    encoded_real_labels = label_encoder.transform(real_labels)
    
    # 将编码结果转换为 Long 类型
    long_encoded_real_labels = torch.tensor(encoded_real_labels, dtype=torch.long)
    train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, long_encoded_real_labels, test_size=0.2)

    # Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = SentenceDataset(train_encodings, train_labels)
    val_dataset = SentenceDataset(val_encodings, val_labels)

    # 模型训练
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=9)

    training_args = TrainingArguments(
        output_dir = output_path,
        num_train_epochs = 10,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size = 64,
        eval_strategy = 'no',
        save_strategy = 'no',  # 训练过程中不保存模型
        logging_dir = None,
        logging_steps = 1000000,
        load_best_model_at_end = False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_accuracy,
    )

    trainer.train()
    
    # 训练结束后保存最终模型
    model.save_pretrained(output_path)
    
