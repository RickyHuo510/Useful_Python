import control as ct
import matplotlib.pyplot as plt
import sympy
import numpy as np

# --------------------------------------------------------------------------
# 函数定义区域
# --------------------------------------------------------------------------

def parse_polynomial_string(poly_string):
    """使用SymPy将多项式字符串解析为系数列表。"""
    if not poly_string.strip():
        print("错误：输入的多项式字符串为空。")
        return None
    s = sympy.Symbol('s')
    try:
        expr = sympy.sympify(poly_string, locals={'s': s})
    except (sympy.SympifyError, SyntaxError) as e:
        print(f"\n错误：多项式字符串 '{poly_string}' 格式无法解析。")
        print("请检查：是否忘记了乘法符号 '*' (例如应为 2*s 而不是 2s) 或使用了错误的幂次符号 '^' (应为 s**2)。")
        return None
    expanded_expr = sympy.expand(expr)
    poly = sympy.Poly(expanded_expr, s)
    coeffs = [float(c) for c in poly.all_coeffs()]
    return coeffs

def calculate_asymptotes(poles, zeros):
    """计算并返回根轨迹的渐近线信息。"""
    n = len(poles)
    m = len(zeros)
    
    if n > m:
        # 计算实轴交点（质心）
        sigma_a = (np.sum(np.real(poles)) - np.sum(np.real(zeros))) / (n - m)
        
        # 计算渐近线角度
        angles_deg = [(2 * k + 1) * 180 / (n - m) for k in range(n - m)]
        angles_rad = np.deg2rad(angles_deg)
        
        return sigma_a, angles_deg, angles_rad
    return None, None, None

def find_breakaway_points(G):
    """使用SymPy精确计算分离点。"""
    s = sympy.Symbol('s')
    
    # 获取分子和分母的SymPy多项式
    num_poly = sympy.Poly(G.num[0][0], s)
    den_poly = sympy.Poly(G.den[0][0], s)
    
    # 计算导数: N(s)D'(s) - N'(s)D(s) = 0
    d_num = sympy.diff(num_poly, s)
    d_den = sympy.diff(den_poly, s)
    
    # ******** 修正点在这里 ********
    # 移除了不必要且错误的 sympy.expand() 调用
    # Poly对象间的运算结果已经是Poly，不需要再展开
    equation = num_poly * d_den - d_num * den_poly
    
    # 求解方程，得到候选点
    candidates = sympy.solve(equation, s)
    
    # 过滤出真实的分离点（必须是实数且增益K>=0）
    valid_points = []
    for p in candidates:
        try:
            p_float = complex(p) # 转换为复数
            # 必须是实数点（忽略非常小的虚部）
            if abs(p_float.imag) < 1e-6:
                # 计算该点的增益 K = -1/G(s)
                gain = -1 / ct.evalfr(G, p_float)
                # 增益必须为正实数
                if abs(gain.imag) < 1e-6 and gain.real >= 0:
                    valid_points.append(p_float.real)
        except TypeError:
            continue # 忽略无法转换为浮点数的解
            
    return valid_points

def calculate_angles_of_departure_arrival(poles, zeros):
    """计算复数极点的起始角和复数零点的终止角。"""
    departure_angles = {}
    arrival_angles = {}

    # 计算起始角（从复数极点出发）
    for p_i in poles:
        if np.imag(p_i) != 0:
            angle_sum = 0
            # 来自其他极点的角度
            for p_j in poles:
                if p_i != p_j:
                    angle_sum += np.angle(p_i - p_j, deg=True)
            # 来自零点的角度
            for z_k in zeros:
                angle_sum -= np.angle(p_i - z_k, deg=True)
            
            departure_angle = 180 - angle_sum
            # 将角度规范化到 (-180, 180]
            departure_angles[p_i] = (departure_angle + 180) % 360 - 180

    # 计算终止角（到达复数零点）
    for z_i in zeros:
        if np.imag(z_i) != 0:
            angle_sum = 0
            # 来自极点的角度
            for p_j in poles:
                angle_sum -= np.angle(z_i - p_j, deg=True)
            # 来自其他零点的角度
            for z_k in zeros:
                if z_i != z_k:
                    angle_sum += np.angle(z_i - z_k, deg=True)

            arrival_angle = 180 + angle_sum
            # 将角度规范化到 (-180, 180]
            arrival_angles[z_i] = (arrival_angle + 180) % 360 - 180
            
    return departure_angles, arrival_angles

def find_imaginary_axis_crossings(G):
    """通过劳斯判据或特征方程计算与虚轴的交点。"""
    s, K = sympy.symbols('s K', real=True)
    num_poly_expr = sympy.Poly(G.num[0][0], s).as_expr()
    den_poly_expr = sympy.Poly(G.den[0][0], s).as_expr()
    
    char_poly_expr = den_poly_expr + K * num_poly_expr
    
    w = sympy.symbols('w', real=True, positive=True)
    char_poly_jw = char_poly_expr.subs(s, sympy.I * w)
    
    expanded_poly = sympy.expand(char_poly_jw)
    real_part = sympy.re(expanded_poly)
    imag_part = sympy.im(expanded_poly)
    
    try:
        imag_eq = sympy.Eq(imag_part, 0)
        w_solutions = sympy.solve(imag_eq, w)
        
        crossings = []
        for w_sol in w_solutions:
            real_eq_at_w = sympy.Eq(real_part.subs(w, w_sol), 0)
            K_solutions = sympy.solve(real_eq_at_w, K)
            
            for k_sol in K_solutions:
                if k_sol >= 0:
                    crossings.append({'gain': float(k_sol), 'omega': float(w_sol)})
        return crossings
    except Exception:
        return []


if __name__ == "__main__":
    
    #num_str = '(s+6)'
    #den_str = '(s+20)*(s+10)*(s**2)'目前仍然有问题的数据,发生了除零错误
    
    num_str = '(1)'
    den_str = '(s)*(s**2+6*s+10)'#无法标记出复数分离点

    numerator_coeffs = parse_polynomial_string(num_str)
    denominator_coeffs = parse_polynomial_string(den_str)

    if numerator_coeffs is not None and denominator_coeffs is not None:
        try:
            G = ct.TransferFunction(numerator_coeffs, denominator_coeffs)
            poles = G.poles()
            zeros = G.zeros()
            
            #print("\n" + "="*50)
            #print("开环系统分析结果:")
            #print("="*50)
            print(f"开环零点: {np.round(zeros, 3) if len(zeros) > 0 else '无'}")
            print(f"开环极点: {np.round(poles, 3)}")

            sigma, angles_d, angles_r = calculate_asymptotes(poles, zeros)
            if sigma is not None:
                #print("\n--- 渐近线 ---")
                print(f"渐近线与实轴交点: {sigma:.3f}")
                print(f"渐近线角度: {[round(a, 2) for a in angles_d]}")
            
            break_points = find_breakaway_points(G)
            if break_points:
                print("\n--- 分离点 ---")
                print(f"实轴上的分离点: {[round(p, 3) for p in break_points]}")

            dep_angles, arr_angles = calculate_angles_of_departure_arrival(poles, zeros)
            if dep_angles:
                print("\n--- 起始角  ---")
                for p, a in dep_angles.items():
                    print(f"从极点 {np.round(p, 3)} 出发的起始角: {a:.2f} 度")
            if arr_angles:
                print("\n--- 终止角  ---")
                for z, a in arr_angles.items():
                    print(f"到达零点 {np.round(z, 3)} 的终止角: {a:.2f} 度")

            crossings = find_imaginary_axis_crossings(G)
            if crossings:
                print("\n--- 与虚轴交点 ---")
                for cross in crossings:
                    print(f"频率: ±{cross['omega']:.3f} j, 此时增益 K = {cross['gain']:.3f}")

            print("\n正在生成根轨迹图...")
            
            plt.figure(figsize=(10, 8))
            ct.root_locus(G, plot=True)
            
            if sigma is not None:
                x_lim = plt.xlim()
                line_length = max(abs(x_lim[0]), abs(x_lim[1])) * 2
                for angle in angles_r:
                    plt.plot([sigma, sigma + line_length * np.cos(angle)],
                             [0, line_length * np.sin(angle)], 'g--', label='Asymptotes')
            
            if break_points:
                plt.plot(break_points, np.zeros_like(break_points), 'b^', markersize=10, label='Breakaway Points')
            
            if crossings:
                for cross in crossings:
                    plt.plot([0, 0], [cross['omega'], -cross['omega']], 'ro', markersize=8, label='Imaginary Axis Crossing')

            plt.title('Root Locus Analysis')
            plt.xlabel('Real Axis')
            plt.ylabel('Imaginary Axis')
            plt.grid(True)
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                plt.legend(by_label.values(), by_label.keys())
                
            plt.show()

        except Exception as e:
            print(f"\n在处理或绘图过程中发生错误: {e}")