import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. 参数设置 (使用列表指定每一层)
# ==========================================
INPUT_SIZE = 10       # 输入特征维度
# 【核心修改】这里改为列表。
# 例如 [10, 5] 表示：
#   第1个隐藏层有10个神经元
#   第2个隐藏层有5个神经元
# 总深度 = len(HIDDEN_SIZE)
HIDDEN_SIZE = [2, 2]  
OUTPUT_SIZE = 1      # 输出维度
LEARNING_RATE = 0.005 
EPOCHS = 100
ACTIVATION = 'relu'  # 可选: 'sigmoid', 'relu', 'tanh'

# ==========================================
# 2. 构建数据集
# ==========================================
torch.manual_seed(time.time())
X = torch.randn(20, INPUT_SIZE)
# 构建一个非线性标签: x1和x2符号相同 且 x3大于0
Y = (((X[:, 0] * X[:, 1]) > 0) & (X[:, 2] > 0)).float().unsqueeze(1)

print(f"输入数据形状: {X.shape}")
print(f"标签数据形状: {Y.shape}")
print(X)
print(Y)

# ==========================================
# 3. 动态构建变宽模型
# ==========================================
class CustomLayerNN(nn.Module):
    def __init__(self, input_dim, hidden_sizes_list, output_dim, act_type='relu'):
        super(CustomLayerNN, self).__init__()
        
        # 1. 选择激活函数
        if act_type == 'sigmoid':
            self.act_func = nn.Sigmoid()
        elif act_type == 'relu':
            self.act_func = nn.ReLU()
        elif act_type == 'tanh':
            self.act_func = nn.Tanh()
        else:
            raise ValueError("不支持的激活函数类型")

        # 2. 动态构建层
        layers = []
        
        # 记录上一层的输出维度，初始化为输入数据的维度
        prev_dim = input_dim
        
        # --- 循环构建隐藏层 ---
        # 这里的 h_dim 会依次取出列表中的值，例如先取10，再取5，再取4
        for h_dim in hidden_sizes_list:
            # 添加线性层: 上一层维度 -> 当前层维度
            layers.append(nn.Linear(prev_dim, h_dim))
            # 添加激活函数
            layers.append(self.act_func)
            # 更新 prev_dim，为下一层做准备
            prev_dim = h_dim
            
        # --- 构建输出层 ---
        # 从 最后一个隐藏层的维度 -> 输出维度
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # --- 输出层激活 (二分类固定用 Sigmoid) ---
        layers.append(nn.Sigmoid())
        
        # 打包模型
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 初始化模型
# 注意：这里直接传入列表 HIDDEN_SIZE
model = CustomLayerNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, ACTIVATION)

print(f"\n当前网络结构 (隐藏层列表={HIDDEN_SIZE}):")
print(model)

# ==========================================
# 4. 训练过程
# ==========================================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_history = []

print("\n开始训练...")
for epoch in range(EPOCHS):
    predictions = model(X)
    loss = criterion(predictions, Y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 2 == 0:
        predicted_classes = (predictions > 0.5).float()
        accuracy = (predicted_classes == Y).float().mean()
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Acc: {accuracy.item():.2f}')

# ==========================================
# 5. 可视化
# ==========================================
plt.plot(loss_history)
plt.title(f'Loss (Hidden Structure={HIDDEN_SIZE})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()