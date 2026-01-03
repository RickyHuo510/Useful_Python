import numpy as np
import matplotlib.pyplot as plt
import control as ct
import sympy as sp

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
plt.style.use('default')
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = ':'

def parse_poly_str(expr_str):
    """解析多项式字符串"""
    s = sp.symbols('s')
    try:
        expr = sp.sympify(expr_str)
        poly = sp.Poly(expr, s)
        return [float(c) for c in poly.all_coeffs()]
    except Exception as e:
        print(f"解析错误: {e}")
        return None

def get_bode_parameters(sys):
    """
    计算绘制渐近线所需的关键参数：
    1. Bode增益 K
    2. 系统型别 v
    3. 转折频率列表 (包含频率值和对应的斜率变化量)
    """
    zeros = sys.zeros()
    poles = sys.poles()
    
    # 1. 计算系统型别 v (原点极点个数 - 原点零点个数)
    num_zeros_origin = sum(1 for z in zeros if abs(z) < 1e-10)
    num_poles_origin = sum(1 for p in poles if abs(p) < 1e-10)
    v = num_poles_origin - num_zeros_origin
    
    # 2. 计算 Bode 增益 K
    # 传递函数化为时间常数形式后，分子分母常数项之比
    # 方法：将 s 替换为一个极小值 (模拟 s->0)，除去 s^v 的影响
    # G(s) * s^v | s->0 = K
    # 使用 control 库的 minreal 消除对消零极点，虽然不是必须但更严谨
    sys_min = sys.minreal()
    # 计算 DC Gain 需要去除积分环节
    # 这里我们用系数法更直接
    num = sys.num[0][0]
    den = sys.den[0][0]
    
    # 找到分子分母中倒数第 (origin_count + 1) 个系数
    # 例如 s^2+2s，倒数第一个是0，倒数第二个是2
    def get_lowest_coef(arr):
        for val in reversed(arr):
            if abs(val) > 1e-10: return val
        return 1.0
        
    K = get_lowest_coef(num) / get_lowest_coef(den)
    
    # 3. 提取转折频率及其斜率变化
    # 规则：实根变化 20dB/dec，复根变化 40dB/dec（复根成对出现，每个算20即可）
    # 零点 +20，极点 -20
    corners = []
    
    # 处理零点
    for z in zeros:
        if abs(z) > 1e-10: # 忽略原点
            corners.append({'w': abs(z), 'slope': 20})
            
    # 处理极点
    for p in poles:
        if abs(p) > 1e-10:
            corners.append({'w': abs(p), 'slope': -20})
            
    # 按频率排序
    corners.sort(key=lambda x: x['w'])
    
    return K, v, corners

def generate_asymptotic_line(omega, K, v, corners):
    """生成渐近线数据"""
    mag_asymp = []
    
    # 初始斜率 (低频段)
    current_slope = -20 * v
    
    # 初始基准点计算：利用低频渐近线方程 L(w) = 20lgK - 20v lgw
    # 取频率数组第一个点作为起点
    w_start = omega[0]
    current_val = 20 * np.log10(abs(K)) - 20 * v * np.log10(w_start)
    
    # 转折点索引
    c_idx = 0
    
    for i, w in enumerate(omega):
        if i > 0:
            # 线性递推：值 = 上一点值 + 斜率 * log(频率比)
            d_log_w = np.log10(w) - np.log10(omega[i-1])
            current_val += current_slope * d_log_w
        
        mag_asymp.append(current_val)
        
        # 检查是否经过转折频率
        # 处理可能重合的转折频率（例如二阶重根）
        while c_idx < len(corners) and w >= corners[c_idx]['w']:
            current_slope += corners[c_idx]['slope']
            c_idx += 1
            
    return np.array(mag_asymp)

def auto_scale_nyquist(sys, ax):
    """模拟 MATLAB 的视野缩放，聚焦原点，忽略无穷大"""
    omega = np.logspace(-3, 3, 2000)
    resp = sys(1j * omega)
    r, i = np.real(resp), np.imag(resp)
    
    # 过滤掉极大的值
    mask = (np.abs(r) < 50) & (np.abs(i) < 50)
    if np.any(mask):
        limit = max(np.max(np.abs(r[mask])), np.max(np.abs(i[mask]))) * 1.2
        limit = max(1.5, limit) # 至少显示 -1.5 到 1.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

def main():
    
    num_str = '100'
    den_str = '(s)*(0.1*s+1)*(0.05*s+1)'

    num_coeffs = parse_poly_str(num_str)
    den_coeffs = parse_poly_str(den_str)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False


    if not num_coeffs or not den_coeffs: return

    # 创建系统
    sys = ct.TransferFunction(num_coeffs, den_coeffs)
    print(f"\n系统传递函数:\n{sys}")

    # 计算参数
    K_bode, v, raw_corners = get_bode_parameters(sys)
    
    # 整理唯一的转折频率用于绘图范围和标注
    # set去重，因为共轭复根会有两个相同的频率
    unique_corners = sorted(list(set([c['w'] for c in raw_corners])))
    
    print(f"Bode增益 K = {K_bode:.4f} (20lgK = {20*np.log10(abs(K_bode)):.2f} dB)")
    print(f"转折频率: {[round(w, 4) for w in unique_corners]} rad/s")

    # 定义频率范围 (覆盖所有转折点)
    if unique_corners:
        w_min = min(unique_corners) / 10
        w_max = max(unique_corners) * 10
    else:
        w_min, w_max = 0.1, 100
    
    # 生成绘图频率点 (对数分布)
    omega = np.logspace(np.log10(w_min), np.log10(w_max), 2000)

    # ==========================================
    # 窗口 1: 标准 Bode 图 (带裕度)
    # ==========================================
    plt.figure(1, figsize=(10, 8))
    ct.bode_plot(sys, omega, dB=True, deg=True, margins=True, grid=True)
    plt.gcf().suptitle("Figure 1: Standard Bode Plot (MATLAB style)")
    plt.tight_layout()

    # ==========================================
    # 窗口 2: Nyquist 图 (带坐标系)
    # ==========================================
    plt.figure(2, figsize=(8, 8))
    ct.nyquist_plot(sys, plot=True, label_freq=False)
    ax_nyq = plt.gca()
    
    # --- 增加实轴和虚轴 ---
    ax_nyq.axhline(0, color='black', linewidth=1.2) # 实轴
    ax_nyq.axvline(0, color='black', linewidth=1.2) # 虚轴
    
    # 标记 (-1, j0)
    ax_nyq.plot(-1, 0, 'r+', markersize=12, markeredgewidth=2, label='Critical Point (-1, 0)')
    
    # 自动缩放视野
    auto_scale_nyquist(sys, ax_nyq)
    
    ax_nyq.set_title("Figure 2: Nyquist Plot")
    ax_nyq.legend()
    plt.tight_layout()

    # ==========================================
    # 窗口 3: 渐近线幅频特性图 (PPT要求)
    # ==========================================
    plt.figure(3, figsize=(10, 6))
    
    # 1. 画精确曲线 (作为参考)
    mag_exact, _, _ = ct.bode(sys, omega, plot=False)
    mag_db_exact = 20 * np.log10(mag_exact)
    plt.semilogx(omega, mag_db_exact, color='blue', alpha=0.5, linewidth=2, label='精确曲线')
    
    # 2. 画渐近线 (折线)
    mag_asymp = generate_asymptotic_line(omega, K_bode, v, raw_corners)
    plt.semilogx(omega, mag_asymp, color='red', linewidth=2, label='渐近线 (PPT方法)')
    
    # 3. 标出转折点坐标
    # 我们需要在 mag_asymp 数组中找到转折频率对应的 dB 值
    for w_c in unique_corners:
        # 找到频率数组中最近的索引
        idx = (np.abs(omega - w_c)).argmin()
        val_db = mag_asymp[idx]
        
        # 画点
        plt.scatter(w_c, val_db, color='green', s=50, zorder=5)
        # 画垂线
        plt.axvline(x=w_c, color='green', linestyle=':', alpha=0.5)
        # 标数值
        label_text = f"({w_c:.2f}, {val_db:.1f}dB)"
        plt.annotate(label_text, (w_c, val_db), 
                     textcoords="offset points", xytext=(0, 10), ha='center', 
                     color='green', fontweight='bold')

    plt.title("Figure 3: Asymptotic Bode Magnitude Plot")
    plt.xlabel("Frequency (rad/s)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.grid(True, which="both", ls=':')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()