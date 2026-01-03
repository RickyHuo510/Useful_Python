import numpy as np
import matplotlib.pyplot as plt
import control as ct
import sympy as sp

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False 
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
    """计算绘制渐近线所需的关键参数"""
    zeros = sys.zeros()
    poles = sys.poles()
    
    num_zeros_origin = sum(1 for z in zeros if abs(z) < 1e-10)
    num_poles_origin = sum(1 for p in poles if abs(p) < 1e-10)
    v = num_poles_origin - num_zeros_origin
    
    def get_lowest_coef(arr):
        for val in reversed(arr):
            if abs(val) > 1e-10: return val
        return 1.0
        
    num = sys.num[0][0]
    den = sys.den[0][0]
    K = get_lowest_coef(num) / get_lowest_coef(den)
    
    corners = []
    for z in zeros:
        if abs(z) > 1e-10: corners.append({'w': abs(z), 'slope': 20})
    for p in poles:
        if abs(p) > 1e-10: corners.append({'w': abs(p), 'slope': -20})
    corners.sort(key=lambda x: x['w'])
    
    return K, v, corners

def generate_asymptotic_line(omega, K, v, corners):
    """生成渐近线数据"""
    mag_asymp = []
    current_slope = -20 * v
    w_start = omega[0]
    current_val = 20 * np.log10(abs(K)) - 20 * v * np.log10(w_start)
    c_idx = 0
    
    for i, w in enumerate(omega):
        if i > 0:
            d_log_w = np.log10(w) - np.log10(omega[i-1])
            current_val += current_slope * d_log_w
        
        mag_asymp.append(current_val)
        
        while c_idx < len(corners) and w >= corners[c_idx]['w']:
            current_slope += corners[c_idx]['slope']
            c_idx += 1
            
    return np.array(mag_asymp)

def auto_scale_nyquist(sys, ax):
    """自动缩放Nyquist图"""
    omega = np.logspace(-3, 3, 2000)
    resp = sys(1j * omega)
    r, i = np.real(resp), np.imag(resp)
    mask = (np.abs(r) < 50) & (np.abs(i) < 50)
    if np.any(mask):
        limit = max(np.max(np.abs(r[mask])), np.max(np.abs(i[mask]))) * 1.2
        limit = max(1.5, limit)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

def calculate_and_print_margins(sys):
    """
    计算并打印幅值裕度和相位裕度
    """
    # ct.margin 返回值: gm (绝对值), pm (度), wg (相位穿越频率), wp (截止频率/幅值穿越频率)
    gm, pm, wg, wp = ct.margin(sys)
    
    # 将幅值裕度转换为 dB
    gm_db = 20 * np.log10(gm) if gm > 0 else float('inf')
    
    print("\n" + "="*40)
    print("       稳定裕度计算结果 (Stability Margins)")
    print("="*40)
    
    # 1. 截止频率与相位裕度
    print(f"1. 截止频率 (Gain Crossover Freq, Wc/Wcp):")
    print(f"   {wp:.4f} rad/s")
    print(f"   -> 在此频率处，幅值 L(w) = 0 dB")
    
    print(f"\n2. 相位裕度 (Phase Margin, PM):")
    print(f"   {pm:.2f} 度 (deg)")
    print(f"   -> 物理意义: γ = 180° + φ(wc)")

    print("-" * 40)

    # 2. 相位穿越频率与幅值裕度
    print(f"3. 相位穿越频率 (Phase Crossover Freq, Wg/Wcg):")
    if wg is not None and not np.isnan(wg):
        print(f"   {wg:.4f} rad/s")
        print(f"   -> 在此频率处，相位 φ(w) = -180°")
    else:
        print(f"   NaN (相位曲线未穿越 -180°)")

    print(f"\n4. 幅值裕度 (Gain Margin, GM):")
    if not np.isinf(gm_db):
        print(f"   {gm_db:.2f} dB  (绝对倍数: {gm:.4f})")
        print(f"   -> 物理意义: h = -L(wg)")
    else:
        print(f"   Inf (无穷大)")

    print("-" * 40)
    
    # 简单的稳定性判定 (仅适用于最小相位系统)
    is_stable = (gm_db > 0 or np.isinf(gm_db)) and pm > 0
    print(f"稳定性判定 (基于最小相位系统判据): \n   {'[ 稳定 ]' if is_stable else '[ 不稳定 ]'}")
    print("="*40 + "\n")
    
    return gm, pm, wg, wp

def main():
    # --- 输入传递函数 ---
    num_str = '3.6*(0.5*s+1)*3.1'
    den_str = '(0.5*s+1)*(0.05*s+1)*s'

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial'] # 用来正常显示中文标签
    num_coeffs = parse_poly_str(num_str)
    den_coeffs = parse_poly_str(den_str)

    if not num_coeffs or not den_coeffs: return

    # 创建系统
    sys = ct.TransferFunction(num_coeffs, den_coeffs)
    print(f"\n系统传递函数:\n{sys}")

    # --- 计算基础Bode参数 ---
    K_bode, v, raw_corners = get_bode_parameters(sys)
    unique_corners = sorted(list(set([c['w'] for c in raw_corners])))
    
    print(f"Bode增益 K = {K_bode:.4f} (20lgK = {20*np.log10(abs(K_bode)):.2f} dB)")
    print(f"转折频率: {[round(w, 4) for w in unique_corners]} rad/s")

    # --- 【新增】计算裕度 ---
    calculate_and_print_margins(sys)

    # 定义频率范围
    if unique_corners:
        w_min = min(unique_corners) / 10
        w_max = max(unique_corners) * 10
    else:
        w_min, w_max = 0.1, 100
    omega = np.logspace(np.log10(w_min), np.log10(w_max), 2000)

    # ==========================================
    # 窗口 1: 标准 Bode 图 (带裕度标注)
    # ==========================================
    plt.figure(1, figsize=(10, 8))
    # margins=True 会自动在图上画出裕度线
    ct.bode_plot(sys, omega, dB=True, deg=True, margins=True, grid=True)
    plt.gcf().suptitle("Figure 1: Standard Bode Plot with Margins")
    plt.tight_layout()

    # ==========================================
    # 窗口 2: Nyquist 图
    # ==========================================
    plt.figure(2, figsize=(8, 8))
    ct.nyquist_plot(sys, plot=True, label_freq=False)
    ax_nyq = plt.gca()
    ax_nyq.axhline(0, color='black', linewidth=1.2)
    ax_nyq.axvline(0, color='black', linewidth=1.2)
    ax_nyq.plot(-1, 0, 'r+', markersize=12, markeredgewidth=2, label='Critical Point (-1, 0)')
    auto_scale_nyquist(sys, ax_nyq)
    ax_nyq.set_title("Figure 2: Nyquist Plot")
    ax_nyq.legend()
    plt.tight_layout()

    # ==========================================
    # 窗口 3: 渐近线幅频特性图
    # ==========================================
    plt.figure(3, figsize=(10, 6))
    mag_exact, _, _ = ct.bode(sys, omega, plot=False)
    mag_db_exact = 20 * np.log10(mag_exact)
    plt.semilogx(omega, mag_db_exact, color='blue', alpha=0.5, linewidth=2, label='精确曲线')
    
    mag_asymp = generate_asymptotic_line(omega, K_bode, v, raw_corners)
    plt.semilogx(omega, mag_asymp, color='red', linewidth=2, label='渐近线')
    
    for w_c in unique_corners:
        idx = (np.abs(omega - w_c)).argmin()
        val_db = mag_asymp[idx]
        plt.scatter(w_c, val_db, color='green', s=50, zorder=5)
        plt.axvline(x=w_c, color='green', linestyle=':', alpha=0.5)
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