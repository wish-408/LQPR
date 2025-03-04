import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
from tools import load_data
from memory_profiler import profile
import psutil
import time

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

@profile
def bert_train():
    max_memory_usage = 0
    data_dir = "../dataset/promise_all/random_split/"

    for split_dir in os.listdir(data_dir):

        output_path = "../bert_models_30/train" + split_dir
        print("模型保存路径 : ",output_path)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        sentences = []
        real_labels = []
        data_path = data_dir + split_dir + "/promise_splited_" + split_dir.split('_')[1] + '_train.txt'
        print("训练数据集路径 : ", data_path)

        sentences, real_labels = load_data(data_path)

        label_encoder = LabelEncoder()
        real_labels = label_encoder.fit_transform(real_labels)

        train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, real_labels, test_size=0.2)

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
        # 开始监控内存
        process = psutil.Process(os.getpid())
        def monitor_memory():
            nonlocal max_memory_usage
            while True:
                memory_info = process.memory_info()
                current_memory_usage = memory_info.rss
                if current_memory_usage > max_memory_usage:
                    max_memory_usage = current_memory_usage
                time.sleep(0.1)  # 每0.1秒检查一次内存使用情况

        import threading
        memory_monitor = threading.Thread(target=monitor_memory)
        memory_monitor.daemon = True
        memory_monitor.start()
        trainer.train()
                # 停止监控
        memory_monitor.join(timeout=1)

        print(f"最大内存消耗: {max_memory_usage / (1024 ** 2):.2f} MB")

        # 训练结束后保存最终模型
        model.save_pretrained(output_path)
        
bert_train()