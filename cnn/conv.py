import torch
import torch.nn.functional as F

# PyTorch 要求输入维度必须是 4 维: (Batch_Size, Channels, Height, Width)
# 假设我们有 1 张图片，1 个通道，5x5 大小
input_tensor = torch.tensor([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 变为 (1, 1, 5, 5)

# 定义卷积核 (Out_Channels, In_Channels, k_h, k_w)
kernel_tensor = torch.tensor([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0) # 变为 (1, 1, 3, 3)

# 执行卷积
output = F.conv2d(input_tensor, kernel_tensor)

print("PyTorch 计算结果:\n", output.squeeze()) # squeeze 去掉多余的维度以便查看