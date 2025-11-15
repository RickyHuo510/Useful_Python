import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.fft import fft, fftfreq
import warnings
import time

x_data = np.array([0.014,0.021,0.031,0.038,0.047,0.062,0.091,0.111])
y_data = np.array([16.6,49.7,27.7,60.5,55.2,44.2,66.2,38.5])


# 2. 输入物理量名称和单位
x_variable_name = r'电流$I_0$'
x_unit = ""

y_variable_name = "电压 U_0"
y_unit = ""

# 3. 设置图表标题
plot_title = "无关联随机数据的拟合测试"


# --- 定义所有拟合函数 ---
def linear_func(x, a, b): return a * x + b
def quadratic_func(x, a, b, c): return a * x**2 + b * x + c
def cubic_func(x, a, b, c, d): return a * x**3 + b * x**2 + c * x + d
def exponential_func(x, a, b): return a * np.exp(np.clip(b * x, -100, 100))
def log_func(x, a, b): return a * np.log(x) + b
def power_func(x, a, b): return a * np.power(x, b)
def gaussian_func(x, a, b, c): return a * np.exp(-(x - b)**2 / (2 * c**2))
def logistic_func(x, L, k, x0): return L / (1 + np.exp(-k * (x - x0)))
def sinusoidal_func(x, a, b, c, d): return a * np.sin(b * x + c) + d
def inverse_func(x, a, b): return a / x + b

# --- 新增的阻尼振荡函数 ---
def damped_sinusoidal_func(x, a, k, b, c, d):
    """ a: 初始振幅, k: 衰减系数, b: 角频率, c: 相位, d: 垂直偏移 """
    return a * np.exp(-k * x) * np.sin(b * x + c) + d


# --- 核心改进：智能猜测初始参数 ---
def estimate_sinusoidal_params(x, y):
    N = len(x)
    if N < 2: return [1, 1, 0, np.mean(y)]
    d_guess = np.mean(y)
    a_guess = (np.max(y) - np.min(y)) / 2.0
    y_centered = y - d_guess
    sampling_spacing = (x[-1] - x[0]) / (N - 1)
    yf = fft(y_centered)
    xf = fftfreq(N, sampling_spacing)[:N//2]
    try:
        idx = np.argmax(np.abs(yf[1:N//2])) + 1 
        freq_guess = xf[idx]
        b_guess = 2 * np.pi * freq_guess
    except (IndexError, ValueError):
        b_guess = 1.0
    c_guess = 0
    return [a_guess, b_guess, c_guess, d_guess]

# --- 主代码 ---
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_r_squared(y_true, y_pred):
    if y_pred is None or np.std(y_pred) < 1e-9: return -np.inf
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: return 1.0 if ss_res == 0 else 0.0
    return 1 - (ss_res / ss_tot)

# 将所有模型（包括新模型）放入字典
models = {
    '线性拟合': [linear_func, 2], '二次多项式拟合': [quadratic_func, 3],
    '三次多项式拟合': [cubic_func, 4], '指数拟合': [exponential_func, 2],
    '高斯拟合': [gaussian_func, 3], 'S型曲线拟合': [logistic_func, 3],
    '正弦拟合': [sinusoidal_func, 4],
    '阻尼振荡拟合': [damped_sinusoidal_func, 5] # 添加新模型
}
if np.all(x_data > 0):
    models['对数拟合'] = [log_func, 2]; models['幂函数拟合'] = [power_func, 2]
if np.all(x_data != 0):
     models['反比例拟合'] = [inverse_func, 2]

best_model, best_r2, best_params, best_func_name = None, -np.inf, None, ""

print("正在尝试不同的模型进行拟合...")
for name, (func, num_params) in models.items():
    try:
        p0 = None
        # 为复杂模型提供初始猜测值
        if name == '高斯拟合': p0 = [np.max(y_data), x_data[np.argmax(y_data)], 1]
        elif name == 'S型曲线拟合': p0 = [np.max(y_data), 1, np.median(x_data)]
        elif name == '正弦拟合':
            p0_sin = estimate_sinusoidal_params(x_data, y_data)
            p0 = p0_sin
            print(f"  - 正弦拟合的初始猜测 (a,b,c,d): [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}, {p0[3]:.2f}]")
        elif name == '阻尼振荡拟合':
            # 复用FFT的结果来猜测公共参数
            p0_sin = estimate_sinusoidal_params(x_data, y_data)
            a_guess, b_guess, c_guess, d_guess = p0_sin
            # 为衰减系数 k 提供一个合理的初始猜测
            k_guess = 0.1 
            p0 = [a_guess, k_guess, b_guess, c_guess, d_guess]
            print(f"  - 阻尼振荡的初始猜测 (a,k,b,c,d): [{p0[0]:.2f}, {p0[1]:.2f}, {p0[2]:.2f}, {p0[3]:.2f}, {p0[4]:.2f}]")
        
        #params, _ = curve_fit(func, x_data, y_data, p0=p0, maxfev=50000)
        params, _ = curve_fit(
            func,
            x_data,
            y_data,
            p0=p0,
            maxfev=2000,      # 大幅增加最大迭代次数
            ftol=1e-14,         # 提高函数值收敛精度
            xtol=1e-14,         # 提高参数值收敛精度
            gtol=1e-14          # 提高梯度收敛精度
        )
        y_pred = func(x_data, *params)
        r2 = calculate_r_squared(y_data, y_pred)
        print(f"  - {name}: R² = {r2:.6f}")
        if r2 > best_r2: best_r2, best_model, best_params, best_func_name = r2, func, params, name
    except (RuntimeError, ValueError) as e:
        print(f"  - {name}: 拟合失败，已跳过。")
        continue

if best_model is None:
    print("\n所有模型都无法成功拟合您的数据。")
else:
    print(f"\n自动选择的最佳模型是: {best_func_name} (R² = {best_r2:.6f})")
    plt.figure(figsize=(16, 8)) # 增大图像尺寸以获得更好视野
    plt.scatter(x_data, y_data, label='原始数据点', color='blue', zorder=5, alpha=0.6)
    x_min, x_max = np.min(x_data), np.max(x_data)
    plot_margin = (x_max - x_min) * 0.05
    x_smooth = np.linspace(x_min - plot_margin, x_max + plot_margin, 500)
    y_smooth = best_model(x_smooth, *best_params)
    label_text = f"最佳拟合: {best_func_name}\n$R^2 = {best_r2:.4f}$"
    plt.plot(x_smooth, y_smooth, label=label_text, color='red', linewidth=2.5)
    plt.xlabel(f"{x_variable_name} ({x_unit})", fontsize=14)
    plt.ylabel(f"{y_variable_name} ({y_unit})", fontsize=14)
    plt.title(plot_title, fontsize=18)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

warnings.resetwarnings()