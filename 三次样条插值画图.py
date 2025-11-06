import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time

# --- 在这里输入数据和标签 ---
#x_data = np.array([0.2,0.4,0.6,0.8,1.0,1.2])
#y_data = np.array([0.4937,0.4827,0.4865,0.4837,0.4819,0.4841])
#y_data=(x_data*55*1.4*0.8)/(x_data*55*1.4*0.8+1.54+x_data*x_data*3.87)*100
#print(y_data)
np.random.seed(int(time.time())) # 固定随机种子以确保结果可复现
num_points = 50 # 数据点的数量

# 生成 0 到 10 之间的随机 x 值，并排序以方便绘图
x_data = np.sort(np.random.rand(num_points) * 10)

# 生成 -5 到 5 之间的完全随机的 y 值
y_data = np.random.uniform(-10, 10, num_points)


x_variable_name = r"电流标幺值 $I^*_2$"
y_variable_name = r"效率 $\eta$%"
plot_title = r"效率曲线"


# 1. 创建一个三次样条插值器对象
# CubicSpline 会计算出连接每个数据点所需的一系列三次多项式
# 这可以理解为“学习”如何在点与点之间进行平滑过渡
cs = CubicSpline(x_data, y_data)

# 2. 生成用于绘制平滑曲线的密集x轴坐标
# 我们需要很多点来展示样条函数计算出的平滑曲线
x_smooth = np.linspace(x_data.min(), x_data.max(), 500)

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
plt.plot(x_data, y_data, 'o', color='blue', markersize=8, label='原始数据点')

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