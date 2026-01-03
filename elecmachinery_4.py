import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ================= 数据录入 (完全保持不变) =================

# 1. 空载特性数据 (Table 4-1)
If_occ_raw = np.array([0.91, 0.82, 0.78, 0.76, 0.71, 0.62, 0.54, 0.46, 0.39, 0.32, 0.25, 0.18, 0.09, 0.0])
U0_occ_raw = np.array([242, 230, 224, 220, 212, 191, 172, 151, 131, 110, 90,  67,   40,   12])

# 排序
sort_idx_occ = np.argsort(If_occ_raw)
If_occ = If_occ_raw[sort_idx_occ]
U0_occ = U0_occ_raw[sort_idx_occ]

# 2. 短路特性数据 (Table 4-2)
If_scc_raw = np.array([0.65, 0.60, 0.57, 0.54, 0.49, 0.40, 0.31, 0.22, 0.13, 0.08, 0.0])
Ik_scc_raw = np.array([0.389, 0.360, 0.350, 0.332, 0.300, 0.250, 0.20, 0.150, 0.100, 0.07, 0.022]) 

# 排序
sort_idx_scc = np.argsort(If_scc_raw)
If_scc = If_scc_raw[sort_idx_scc]
Ik_scc = Ik_scc_raw[sort_idx_scc]

# 3. 外特性数据 (Table 4-3)
I_ext_raw = np.array([0.354, 0.307, 0.27, 0.227, 0.179, 0.119, 0.08, 0.074])
U_ext_raw = np.array([222, 234, 242, 248, 253, 257, 259, 260])

# 排序
sort_idx_ext = np.argsort(I_ext_raw)
I_ext = I_ext_raw[sort_idx_ext]
U_ext = U_ext_raw[sort_idx_ext]

# ================= 三次样条插值函数 (保持不变) =================
def smooth_curve(x, y, num_points=300):
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

# ================= 绘图 (修改部分：独立窗口 + 加密网格) =================
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# --- 图1：空载特性曲线 ---
plt.figure(1, figsize=(8, 6)) # 创建第1个独立窗口
x1, y1 = smooth_curve(If_occ, U0_occ)
plt.scatter(If_occ, U0_occ, color='red', label='实验数据')
plt.plot(x1, y1, color='blue', label='拟合曲线')
plt.title('空载特性曲线 $U_0 = f(I_f)$')
plt.xlabel('励磁电流 $I_f$ (A)')
plt.ylabel('空载电压 $U_0$ (V)')
plt.legend()
# 网格加密设置
plt.minorticks_on() # 开启次刻度
plt.grid(which='major', linestyle='-', linewidth=0.75, alpha=0.8) # 主网格
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)  # 次网格（加密）

# --- 图2：短路特性曲线 ---
plt.figure(2, figsize=(8, 6)) # 创建第2个独立窗口
x2, y2 = smooth_curve(If_scc, Ik_scc)
plt.scatter(If_scc, Ik_scc, color='red', label='实验数据')
plt.plot(x2, y2, color='green', label='拟合曲线')
plt.title('短路特性曲线 $I_K = f(I_f)$')
plt.xlabel('励磁电流 $I_f$ (A)')
plt.ylabel('短路电流 $I_K$ (A)')
plt.legend()
# 网格加密设置
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.75, alpha=0.8)
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

# --- 图3：外特性曲线 ---
plt.figure(3, figsize=(8, 6)) # 创建第3个独立窗口
x3, y3 = smooth_curve(I_ext, U_ext)
plt.scatter(I_ext, U_ext, color='red', label='实验数据')
plt.plot(x3, y3, color='purple', label='拟合曲线')
plt.title('外特性曲线 (纯电阻) $U = f(I)$')
plt.xlabel('负载电流 $I$ (A)')
plt.ylabel('端电压 $U$ (V)')
plt.legend()
# 网格加密设置
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth=0.75, alpha=0.8)
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

# 显示所有窗口
plt.show()