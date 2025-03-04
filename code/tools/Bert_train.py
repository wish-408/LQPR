import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

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

for i in range(1, 10):
    print(os.getcwd())
    print(f"test case : {i}")
    output_path = "./results/train" + str(i)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print(output_path)
    
    sentences = []
    real_labels = []
    data_path = '../data/new_labeled_data1.txt'

    full_sentences = open(data_path,'r',encoding='utf-8').read().split('\n')

    for sentence in full_sentences:
        sentences.append(sentence.split('@')[1])
        label = sentence.split('@')[2]
        real_labels.append(label)

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
        output_dir=output_path,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        eval_strategy='epoch',  # 每个 epoch 结束时进行评估
        save_strategy='epoch',  # 保存策略
        logging_dir=None,  # 不记录日志文件
        logging_steps=1000000,  # 设置一个很大的值，避免在控制台打印日志
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_accuracy,
    )

    trainer.train()
