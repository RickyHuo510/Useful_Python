import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from matplotlib.ticker import MultipleLocator

# --- 数据准备 ---
# 这里我们定义了三组离散数据 (x, y)
# 您可以根据需要替换成您自己的数据

# 数据组 1
x1 = np.array([0.911,0.793,0.684,0.575,0.498,0.332,0.114])
y1 = np.array([1416,1431,1444,1458,1469,1487,1520])

# 数据组 2
x2 = np.array([0.11,0.247,0.340,0.467,0.594,0.694,0.802])
y2 = np.array([1285,1266,1247,1236,1221,1207,1194])

# 数据组 3
x3 = np.array([1.001,0.886,0.779,0.681,0.576,0.450,0.197,0.102])
y3 = np.array([1107,1156,1195,1239,1284,1330,1428,1468])

# 将所有数据组放入一个列表中，方便循环处理
datasets = [
    {'x': x1, 'y': y1, 'label': '固有机械特性'},
    {'x': x2, 'y': y2, 'label': '降低电枢电压的人为机械特性'},
    {'x': x3, 'y': y3, 'label': '串联电阻的人为机械特性'}
]

colors = ['#1f77b4','#d62728', '#2ca02c','#ff7f0e', '#9467bd', '#8c564b']

# --- 创建绘图窗口 ---
plt.figure(figsize=(10, 6))

# --- 对每组数据进行插值和绘图 ---
for i, data in enumerate(datasets):
    x_original = data['x']
    y_original = data['y']
    label = data['label']

    # ==========================================================
    # 在这里添加排序代码
    # 1. 获取对 x 进行排序的索引
    sort_indices = np.argsort(x_original)
    
    # 2. 使用这些索引来排序 x 和 y，以保持对应关系
    x_sorted = x_original[sort_indices]
    y_sorted = y_original[sort_indices]
    # ==========================================================

    # 1. 创建三次样条插值函数
    cs = CubicSpline(x_sorted, y_sorted)

    # 2. 创建更密集的x轴数据点，用于绘制平滑曲线
    x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 300)

    # 3. 计算插值后的y轴数据
    y_smooth = cs(x_smooth)

    # 4. 绘制原始数据点（散点图）
    plt.scatter(x_sorted, y_sorted, label=f'{label}', color=colors[i])

    # 5. 绘制三次样条插值曲线
    plt.plot(x_smooth, y_smooth, label=f'{label}', color=colors[i])

# --- 图表美化 ---
# 设置中文字体，以防显示乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 添加标题和坐标轴标签
plt.title('他励电动机的三种机械特性')
plt.xlabel('转矩 (N * m)')
plt.ylabel('转速 (r / min)')

# 显示图例
plt.legend()

# 显示网格
#plt.grid(True)
ax = plt.gca()
# 设置 x 轴的次刻度，每 0.25 个单位一个
ax.xaxis.set_minor_locator(MultipleLocator(0.05))
# 设置 y 轴的次刻度，每 0.25 个单位一个
ax.yaxis.set_minor_locator(MultipleLocator(25))
# 启用并自定义主网格
plt.grid(which='major', linestyle='-', linewidth='1.0', color='gray')
# 启用并自定义次网格
plt.grid(which='minor', linestyle=':', linewidth='0.8', color='gray')
plt.xlim(0, 1.1)
plt.ylim(1000, 1550)
# --- 显示图表 ---
plt.show()