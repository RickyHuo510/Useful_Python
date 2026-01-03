import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import pickle  # 用来保存词表
import os
import pandas as pd

# ==========================================
# 0. GPU 设备配置 (核心步骤)
# ==========================================
# 检查是否有 NVIDIA 显卡，没有则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 当前使用的计算设备: {device}")
if device.type == 'cuda':
    print(f"   显卡名称: {torch.cuda.get_device_name(0)}")

# ==========================================
# 1. 配置与模型定义 (保持不变)
# ==========================================
class Config:
    vocab_size = 5000
    embed_dim = 100
    filter_sizes = [3, 4, 5]
    num_filters = 100
    num_classes = 2
    dropout = 0.5
    batch_size = 2
    lr = 0.001
    epochs = 20
    max_len = 20
    model_save_path = "textcnn_model.pth"   # 模型权重保存路径
    vocab_save_path = "vocab.pkl"           # 词表保存路径

# 数据预处理工具
def tokenizer(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    return text.split()

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, num_filters, num_classes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs) 
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conved = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        logits = self.fc(cat)
        return logits

# Dataset 定义
class TextDataset(Dataset):
    def __init__(self, data, word2idx, max_len):
        self.data = data
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, label = self.data[index]
        tokens = tokenizer(text)
        token_ids = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        if len(token_ids) < self.max_len:
            token_ids += [0] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]
        return torch.tensor(token_ids), torch.tensor(label)

# ==========================================
# 2. 训练流程 (包含 GPU 操作)
# ==========================================
def train():
    # 模拟数据
    df=pd.read_csv("IMDBDataset.csv")
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    raw_data = list(zip(df['review'][:1000], df['sentiment'][:1000]))

    # 构建词表
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text, label in raw_data:
        for word in tokenizer(text):
            if word not in word2idx:
                word2idx[word] = idx
                idx += 1
    
    # 准备 DataLoader
    dataset = TextDataset(raw_data, word2idx, Config.max_len)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    # 初始化模型，并搬运到 GPU !!!
    model = TextCNN(len(word2idx), Config.embed_dim, Config.filter_sizes, 
                    Config.num_filters, Config.num_classes, Config.dropout)
    model = model.to(device)  # <--- 关键步骤：模型搬家

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)

    print("\n--- 开始训练 ---")
    model.train()
    for epoch in range(Config.epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # 数据搬运到 GPU !!!
            batch_x = batch_x.to(device)  # <--- 关键步骤：输入搬家
            batch_y = batch_y.to(device)  # <--- 关键步骤：标签搬家
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{Config.epochs}], Loss: {total_loss/len(dataloader):.4f}")

    # ==========================================
    # 3. 保存模型与词表
    # ==========================================
    print("\n--- 保存模型 ---")
    # 1. 保存模型参数 (state_dict)
    torch.save(model.state_dict(), Config.model_save_path)
    print(f"✅ 模型参数已保存至: {Config.model_save_path}")

    # 2. 保存词表 (word2idx)
    # 这步至关重要，没有词表，模型就是废铁
    with open(Config.vocab_save_path, 'wb') as f:
        pickle.dump(word2idx, f)
    print(f"✅ 词表已保存至: {Config.vocab_save_path}")

# ==========================================
# 4. 加载与推理 (模拟生产环境调用)
# ==========================================
class SentimentPredictor:
    def __init__(self):
        # 1. 加载词表
        if not os.path.exists(Config.vocab_save_path):
            raise FileNotFoundError("找不到词表文件，请先运行训练！")
            
        with open(Config.vocab_save_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        
        # 2. 初始化模型结构 (参数必须与训练时完全一致)
        self.model = TextCNN(len(self.word2idx), Config.embed_dim, Config.filter_sizes, 
                             Config.num_filters, Config.num_classes, Config.dropout)
        
        # 3. 加载权重
        # map_location确保在没有GPU的机器上也能加载GPU训练的模型
        self.model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
        
        # 4. 搬运到 GPU
        self.model = self.model.to(device)
        self.model.eval() # 开启评估模式 (关闭Dropout)
        
        print("🎉 模型加载成功，随时待命！")

    def predict(self, text):
        # 数据预处理
        tokens = tokenizer(text)
        token_ids = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        
        # Padding
        if len(token_ids) < Config.max_len:
            token_ids += [0] * (Config.max_len - len(token_ids))
        else:
            token_ids = token_ids[:Config.max_len]
        
        # 转 Tensor 并搬运到 GPU
        tensor_input = torch.tensor(token_ids).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.model(tensor_input)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            
        return pred_idx, probs[0][pred_idx].item()

# ==========================================
# 主程序入口
# ==========================================
if __name__ == '__main__':
    # 第一次运行：训练并保存
    train()
    
    # 模拟：重启程序后，直接加载模型进行预测
    print("\n--- 模拟重新加载模型 ---")
    predictor = SentimentPredictor()
    
    # 测试
    test_sentences = [
        "This is the best movie I have seen",
        "Absolutely garbage, do not watch"
    ]
    
    for sent in test_sentences:
        label, conf = predictor.predict(sent)
        res_str = "积极 😊" if label == 1 else "消极 😡"
        print(f"语句: {sent}\n预测: {res_str} (置信度: {conf*100:.2f}%)")
        print("-" * 30)