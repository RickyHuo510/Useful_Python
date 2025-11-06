import numpy as np
import matplotlib.pyplot as plt
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# 设置参数
frequency = 50  # 频率 (Hz)
max_voltage = 220  # 幅值电压 (V)
t = np.linspace(0, 0.04, 1000)  # 时间轴，显示两个周期

# 计算三相电压
# 相位 A
voltage_a = max_voltage * np.sin(2 * np.pi * frequency * t)
# 相位 B, 滞后 120 度 (2*pi/3)
voltage_b = max_voltage * np.sin(2 * np.pi * frequency * t - 2 * np.pi / 3)
# 相位 C, 超前 120 度 (2*pi/3)
voltage_c = max_voltage * np.sin(2 * np.pi * frequency * t + 2 * np.pi / 3)

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制三相波形
plt.plot(t, voltage_a, label='相位 A')
plt.plot(t, voltage_b, label='相位 B')
plt.plot(t, voltage_c, label='相位 C')

# 添加图表元素
plt.title('三相交流电波形图')
plt.xlabel('时间 (s)')
plt.ylabel('电压 (V)')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5) # 添加 x 轴

# --- 修改部分 ---
# 将图例放置在右上角
plt.legend(loc='upper right')

# 显示图表
plt.show()