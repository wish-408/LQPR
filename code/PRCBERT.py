import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from tools import load_data, result_report, get_date_time
from memory_profiler import profile

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class RobertaClassifier(nn.Module):
    def __init__(self, model_name, num_label, pooler='first'):
        super(RobertaClassifier, self).__init__()
        self.num_label = num_label
        self.model = AutoModel.from_pretrained('checkpoints/' + model_name)
        
        # Key modification 1: Adapt output layer dimension to actual number of classes
        self.linear = nn.Linear(self.model.config.hidden_size, num_label)  # 2 → num_label
        
        self.softmax = nn.Softmax(dim=-1)
        self.pooler = pooler
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        outputs = self.model(**X)
        if self.pooler == 'first':
            cls_emb = outputs.pooler_output 
        elif self.pooler == 'mean':
            mask = X['attention_mask']
            seq_len = mask.sum(-1, keepdim=True)
            attn_mask = mask.unsqueeze(2).repeat(1,1,self.model.config.hidden_size)
            last_hidden_state = outputs.last_hidden_state
            cls_emb = (last_hidden_state*attn_mask).sum(dim=1)/seq_len
        
        # Add dropout layer
        cls_emb = self.dropout(cls_emb)
        logits = self.linear(cls_emb)
        return logits

    def predict(self, loader, tokenizer):
        self.eval()
        metrics = torch.zeros([self.num_label, self.num_label])
        with torch.no_grad():
            for batch_sample, batch_label in tqdm(loader, desc='infering'):
                batch_sample = list(batch_sample)
                
                # Key modification 2: Add tokenizer parameter to avoid warnings
                X = tokenizer(
                    batch_sample, 
                    return_tensors='pt', 
                    padding='longest',
                    max_length=512,
                    truncation=True,
                    clean_up_tokenization_spaces=False  # Explicitly set parameter
                ).to(device)
                
                logits = self.forward(X)
                
                # Key modification 3: Correct way to get prediction results
                _, preds = torch.max(logits, dim=1)  # Directly get index of maximum probability
                
                # Process true labels (assuming batch_label is one-hot encoded)
                true_labels = torch.argmax(torch.LongTensor(batch_label), dim=1)
                
                # Update confusion matrix
                for t, p in zip(true_labels, preds):
                    metrics[t, p] += 1
                    
        self.train()
        return metrics

# Training function
def train_model(model, label_encoder, train_loader, test_loader, num_epochs=3):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training phase
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    return model

# Evaluation function
def evaluate_model(model, label_encoder, test_loader, result_path):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].numpy()
            
            outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
            _, preds = torch.max(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels)
    
    # Convert back to original labels
    pred_labels = label_encoder.inverse_transform(predictions)
    true_labels = label_encoder.inverse_transform(true_labels)
    
    result_report("PRCBERT", result_path, pred_labels, true_labels)
    # Generate classification report
    report = classification_report(true_labels, pred_labels, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "detailed_report": report
    }

# Model saving
def save_checkpoint(model, label_encoder, filename):
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_classes": label_encoder.classes_
    }, filename)
    print(f"Model saved to {filename}")
    
def load_checkpoint(filename):
    checkpoint = torch.load(filename, map_location=device, weights_only=False)
    
    # Reconstruct label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = checkpoint['label_classes']
    
    # Initialize model structure (must be identical to training)
    model = RobertaClassifier(
        model_name='roberta-large',
        num_label=len(label_encoder.classes_),
        pooler='mean'  # Assuming mean pooling is used
    )
    
    # Load parameters
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, label_encoder

# @profile
def PRCBERT_train():
    for i in range(30):
        data_path = f"./dataset/Promise/random_split/split_{i}/promise_splited_{i}_train.txt"
        test_data_path = f"./dataset/Promise/random_split/split_{i}/promise_splited_{i}_test.txt"
        sentences, real_labels = load_data(data_path)
        test_sentences, test_real_labels = load_data(test_data_path)
        # Label encoding
        label_encoder = LabelEncoder()
        label_encoder.fit(real_labels + test_real_labels)  # Ensure all possible labels are included
        encoded_train_labels = label_encoder.transform(real_labels)
        encoded_test_labels = label_encoder.transform(test_real_labels)
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained("checkpoints/roberta-large")
        
        # Create dataset
        train_dataset = TextClassificationDataset(
            texts=sentences,
            labels=encoded_train_labels,
            tokenizer=tokenizer
        )
        test_dataset = TextClassificationDataset(
            texts=test_sentences,
            labels=encoded_test_labels,
            tokenizer=tokenizer
        )
        
        # Create data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=3,
            shuffle=False,
            num_workers=2
        )
        
        print("load data success")
        
        
        # Initialize model
        model = RobertaClassifier(
            model_name="roberta-large",
            num_label=len(label_encoder.classes_),
            pooler="mean"
        ).to(device)
        
        try:
            loaded_state_dict = torch.load(
                "checkpoints/state_dict_finetuned_on_promise",
                map_location=device,
                weights_only=True
            )
        except TypeError:  # Handle backward compatibility
            loaded_state_dict = torch.load(
                "checkpoints/state_dict_finetuned_on_promise",
                map_location=device
            )
        # Parameter filtering
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in loaded_state_dict.items() 
                        if k in model_dict and v.shape == model_dict[k].shape}
        # Load parameters (allow partial missing)
        model.load_state_dict(filtered_dict, strict=False)
        print("load model success")
        # Train model
        trained_model = train_model(model, label_encoder, train_loader, test_loader, num_epochs=1)   
        if not os.path.exists(f"../PRCBERT_model/trainsplit_{i}/"):
            os.makedirs(f"../PRCBERT_model/trainsplit_{i}/")
        save_path = f"../PRCBERT_model/trainsplit_{i}/model.pth"
        save_checkpoint(trained_model, label_encoder, save_path)
        

def preprocess_text(text):
    tokenizer = AutoTokenizer.from_pretrained('../PRCBERT_model/roberta-large')
    return tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    ).to(device)
    
def predict_single(text):
    inputs = preprocess_text(text)
    model, label_encoder = load_checkpoint("final_model.pth")
    model.eval()  # Switch to prediction mode
    print("load model success")
    with torch.no_grad():
        outputs = model(inputs)
        pred_idx = torch.argmax(outputs).item()
    return label_encoder.inverse_transform([pred_idx])[0]

# @profile
def PRCBERT(model_path, test_data_path, result_dir, save_name, tokenizer_path = "../PRCBERT_model/roberta-large"):

    model, label_encoder = load_checkpoint(model_path)
    model.eval()  # Switch to prediction mode
    print("load model success")
    
    test_sentences, test_real_labels = load_data(test_data_path)
    encoded_test_labels = label_encoder.transform(test_real_labels)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    test_dataset = TextClassificationDataset(
        texts=test_sentences,
        labels=encoded_test_labels,
        tokenizer=tokenizer
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=3,
        shuffle=False,
        num_workers=2
    )
    print("load data success")
    
    # Evaluation phase
    result_path = result_dir + get_date_time() + '_' + save_name + '.xlsx'
    eval_results = evaluate_model(model, label_encoder, test_loader, result_path)
    print(f"Evaluation Results:\n{eval_results}")