import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 任务(1): 绘制点分布图

# 样本集 S
S = np.array([(1, 2), (4, 3), (2, 9), (5, 7), (12, 6)])

# 地标点 l1, l2, l3
landmarks = np.array([(4, 3), (2, 9), (5, 7)])

# --- 修正部分：为了方便检查，将地标点转换为元组集合 ---
landmark_set = set(map(tuple, landmarks))

# 提取 x 和 y 坐标用于绘图
x_coords = S[:, 0]
y_coords = S[:, 1]
lx_coords = landmarks[:, 0]
ly_coords = landmarks[:, 1]

# 创建图形和坐标轴
plt.figure(figsize=(10, 8))
# 绘制样本点 S
plt.scatter(x_coords, y_coords, s=100, label='样本点 S', c='blue', marker='o', zorder=2)
# 绘制地标点
#plt.scatter(lx_coords, ly_coords, s=150, label='地标点 L', c='red', marker='X', zorder=3)

# --- 修正部分：有选择地为点添加坐标文本 ---
# 1. 只为那些不是地标的样本点添加标签
for x, y in S:
    if (x, y) not in landmark_set:
        plt.text(x + 0.2, y + 0.2, f'({x}, {y})', zorder=4)

# 2. 为所有地标点添加标签（这将覆盖是样本点又是地标点的标签）
for i, (x, y) in enumerate(landmarks):
    plt.text(x + 0.2, y + 0.2, f'L{i+1} ({x}, {y})', color='red', zorder=4)

# 设置图表属性
plt.title('样本集S的点分布图')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True, zorder=1)
plt.legend()
plt.axis([0, 14, 0, 10]) # 设置坐标轴范围
plt.show()