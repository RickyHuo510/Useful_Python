import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time

# --- 在这里输入数据和标签 ---
x_data = np.array([105,96,81,65,49,42,35,33,18])
#y_data = np.array([159.4,145,125.4,104,84,74.6,66.6,63.6,47])
#y_data=np.array([0.318,0.289,0.26,0.231,0.202,0.188,0.176,0.170,0.151])
#y_data=np.array([65.87,66.21,64.59,62.50,58.33,56.3,52.55,51.89,38.3])
#y_data=np.array([0.073,0.065,0.056,0.045,0.033,0.027,0.023,0.023,0.025,0.017])
y_data=np.array([0.781,0.778,0.744,0.692,0.636,0.607,0.576,0.572,0.473])
#y_data=(x_data*55*1.4*0.8)/(x_data*55*1.4*0.8+1.54+x_data*x_data*3.87)*100
#np.random.seed(int(time.time())) # 固定随机种子以确保结果可复现
#num_points = 50 # 数据点的数量
# 生成 0 到 10 之间的随机 x 值，并排序以方便绘图
#x_data = np.sort(np.random.rand(num_points) * 10)
# 生成 -5 到 5 之间的完全随机的 y 值
#y_data = np.random.uniform(-10, 10, num_points)

x_variable_name = r"输出功率 $P_2$"
y_variable_name = r"功率因数 $cos\Phi_1$"
plot_title = r"功率因数曲线"


sort_indices = np.argsort(x_data) 

x_sorted = x_data[sort_indices]
y_sorted = y_data[sort_indices]

# 1. 创建一个三次样条插值器对象
# CubicSpline 会计算出连接每个数据点所需的一系列三次多项式
# 这可以理解为“学习”如何在点与点之间进行平滑过渡
cs = CubicSpline(x_sorted, y_sorted)

# 2. 生成用于绘制平滑曲线的密集x轴坐标
# 我们需要很多点来展示样条函数计算出的平滑曲线
x_smooth = np.linspace(x_sorted.min(), x_sorted.max(), 500)

# 3. 使用样条插值器计算平滑曲线的y轴坐标
# 这里的 cs() 就像一个函数，它内部包含了所有分段多项式的信息
# 它会根据 x_smooth 的每个值，自动选择对应的多项式段进行计算
y_smooth = cs(x_smooth)



# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(16, 8))
plt.minorticks_on()

# 绘制原始数据点（现在曲线将精确穿过它们）
plt.plot(x_sorted, y_sorted, 'o', color='blue', markersize=8, label='原始数据点')

# 绘制三次样条插值曲线
plt.plot(x_smooth, y_smooth, label='三次样条插值曲线', color='red', linewidth=2.5)

plt.xlabel(f"{x_variable_name}", fontsize=20)
plt.ylabel(f"{y_variable_name}", fontsize=20)
plt.title(plot_title, fontsize=32)
plt.legend(loc='best', fontsize=16)
#plt.grid(True, linestyle='--', alpha=0.7)
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#plt.ylim(85,100)    #设置y轴值域
plt.tight_layout()
plt.show()