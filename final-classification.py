import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
import json
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ==================== 文本清洗预处理 ====================
def clean_legal_text(text, keep_short_forms=True):
    legal_symbols = {'§', '¶', '©', '®', '™', '°'}
    text = text.lower()

    if keep_short_forms:
        short_forms = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so on',
            's.': 'section',
            'art.': 'article'
        }
        for short, full in short_forms.items():
            text = text.replace(short, full)

    translator = str.maketrans('', '', ''.join(set(string.punctuation) - legal_symbols))
    text = text.translate(translator)

    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words]
    text = ' '.join(words)

    text = ' '.join(text.split())
    return text.strip()




# ==================== 数据加载与处理 ====================
def load_and_process_data(file_path):
    """加载并处理原始数据"""


    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    processed_data = []
    for entry in raw_data:
        if not isinstance(entry, dict):
            continue

        for fact in entry.get('Facts', []):
            if isinstance(fact, dict):
                for key in ['1', '2', '3', '4', '5']:
                    if key in fact and (text := fact[key].strip()):
                        cleaned_text = clean_legal_text(text)
                        processed_data.append({
                            'original_text': text,
                            'cleaned_text': cleaned_text,
                            'original_label': key,
                        })

        for reasoning in entry.get('Reasoning', []):
            if isinstance(reasoning, dict):
                for key in ['6', '7', '8']:
                    if key in reasoning and (text := reasoning[key].strip()):
                        cleaned_text = clean_legal_text(text)
                        processed_data.append({
                            'original_text': text,
                            'cleaned_text': cleaned_text,
                            'original_label': key,
                        })

        for result in entry.get('Results', []):
            if isinstance(result, dict):
                for key in ['9', '10', '11']:
                    if key in result and (text := result[key].strip()):
                        cleaned_text = clean_legal_text(text)
                        processed_data.append({
                            'original_text': text,
                            'cleaned_text': cleaned_text,
                            'original_label': key,
                        })

    return pd.DataFrame(processed_data)


# ==================== 数据集类 ====================
class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ==================== 训练函数 ====================
def train_model(df, num_epochs=8):
    #df['final_label'] = df['rule_label'].fillna(df['original_label'])
    df['final_label'] = df['original_label']
    df = df.dropna(subset=['cleaned_text', 'final_label'])

    # 编码标签
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['final_label'])

    # 数据集划分：70%训练集，15%验证集，15%测试集
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # 初始化模型
    model_name = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_)
    )

    # 创建数据集
    train_dataset = LegalDataset(
        train_df['cleaned_text'],
        train_df['label_encoded'],
        tokenizer
    )
    val_dataset = LegalDataset(
        val_df['cleaned_text'],
        val_df['label_encoded'],
        tokenizer
    )
    test_dataset = LegalDataset(
        test_df['cleaned_text'],
        test_df['label_encoded'],
        tokenizer
    )

    # 数据集加载
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss, epoch_train_correct = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            epoch_train_correct += (preds == batch['labels']).sum().item()

        model.eval()
        epoch_val_loss, epoch_val_correct = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                epoch_val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                epoch_val_correct += (preds == batch['labels']).sum().item()

        train_loss = epoch_train_loss / len(train_loader)
        train_acc = epoch_train_correct / len(train_dataset)
        val_loss = epoch_val_loss / len(val_loader)
        val_acc = epoch_val_correct / len(val_dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'b-o', label='Train')
    plt.plot(range(1, num_epochs + 1), val_losses, 'r-o', label='Val')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, 'b-o', label='Train')
    plt.plot(range(1, num_epochs + 1), val_accs, 'r-o', label='Val')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('Legal-BERT_uncased.png')
    plt.close()

    return model, tokenizer, label_encoder, test_dataset  # 返回测试集


# ==================== 测试模型 ====================
def evaluate_model(model, test_loader, label_encoder):
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            preds = torch.argmax(outputs.logits, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch['labels'].cpu().numpy())

    # 计算准确率和分类报告
    accuracy = accuracy_score(test_labels, test_preds)
    report = classification_report(test_labels, test_preds, target_names=label_encoder.classes_)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("Loading and processing data...")
    df = load_and_process_data("number_data.json")

    print("\nTraining model...")
    model, tokenizer, label_encoder, test_dataset = train_model(df, num_epochs=5)

    # 评估模型在测试集上的表现
    test_loader = DataLoader(test_dataset, batch_size=8)
    evaluate_model(model, test_loader, label_encoder)

    output_dir = "Legal-BERT_uncased_output"
    os.makedirs(output_dir, exist_ok=True)

    # 修改后的保存方式 - 使用Hugging Face标准格式
    model.save_pretrained(output_dir)  # 保存模型(config.json + pytorch_model.bin)
    tokenizer.save_pretrained(output_dir)  # 保存tokenizer
    
    # 保存标签编码器
    with open(os.path.join(output_dir, 'label_encoder.json'), 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)

    print("训练完成，模型和其他文件已保存。")