import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 标准梯度下降函数 (带详细打印) ---
def compute_cost(X, y, theta):
    """计算标准代价函数 J(θ)"""
    m = len(y)
    predictions = X @ theta
    sq_errors = (predictions - y) ** 2
    return (1 / (2 * m)) * np.sum(sq_errors)

def gradient_descent(X, y, theta_start, alpha, num_iters):
    """执行标准梯度下降，并显示每次迭代的详细信息"""
    m = len(y)
    theta = theta_start.copy()
    cost_history = []
    theta_history = []

    print("--- 开始标准梯度下降 ---")
    
    # 打印初始状态
    initial_cost = compute_cost(X, y, theta)
    print(f"初始状态 (迭代 0):")
    print(f"  参数: θ₀ = {theta[0,0]:.4f}, θ₁ = {theta[1,0]:.4f}, θ₂ = {theta[2,0]:.4f}")
    print(f"  代价函数 J(θ) = {initial_cost:.4f}\n")
    cost_history.append(initial_cost)
    theta_history.append(theta.flatten())

    for i in range(num_iters):
        # 计算当前预测和误差
        predictions = X @ theta
        errors = predictions - y

        # --- 新增的详细信息打印 ---
        print(f"--- 第 {i+1} 次迭代 ---")
        print(f"  当前参数: θ₀={theta[0,0]:.4f}, θ₁={theta[1,0]:.4f}, θ₂={theta[2,0]:.4f}")
        for j in range(m):
            print(f"    样本 {j} (x₁={int(X[j,1])}, x₂={int(X[j,2])}): 预测值={predictions[j,0]:.3f}, 实际值={y[j,0]}, 误差={errors[j,0]:.3f}")
        print("-" * 40)
        # --- 打印结束 ---

        # 计算梯度并更新theta
        gradient = (1 / m) * (X.T @ errors)
        theta = theta - alpha * gradient
        
        # 记录历史并打印总结
        new_cost = compute_cost(X, y, theta)
        cost_history.append(new_cost)
        theta_history.append(theta.flatten())
        print(f"第 {i+1} 次迭代总结:")
        print(f"  更新后参数: θ₀ = {theta[0,0]:.4f}, θ₁ = {theta[1,0]:.4f}, θ₂ = {theta[2,0]:.4f}")
        print(f"  代价函数 J(θ) = {new_cost:.4f}\n")
        
    return cost_history, np.array(theta_history)

# --- 2. 正则化梯度下降函数 (带详细打印) ---
def compute_cost_regularized(X, y, theta, lambda_val):
    """计算正则化的代价函数"""
    m = len(y)
    standard_cost = compute_cost(X, y, theta)
    reg_term = (lambda_val / (2 * m)) * np.sum(theta[1:]**2)
    return standard_cost + reg_term

def gradient_descent_regularized(X, y, theta_start, alpha, num_iters, lambda_val):
    """执行正则化的梯度下降，并显示每次迭代的详细信息"""
    m = len(y)
    theta = theta_start.copy()
    cost_history = []
    theta_history = []

    print("\n--- 开始正则化梯度下降 ---")

    # 打印初始状态
    initial_cost = compute_cost_regularized(X, y, theta, lambda_val)
    print(f"初始状态 (迭代 0):")
    print(f"  参数: θ₀ = {theta[0,0]:.4f}, θ₁ = {theta[1,0]:.4f}, θ₂ = {theta[2,0]:.4f}")
    print(f"  正则化代价 J(θ) = {initial_cost:.4f}\n")
    cost_history.append(initial_cost)
    theta_history.append(theta.flatten())

    for i in range(num_iters):
        # 计算当前预测和误差
        predictions = X @ theta
        errors = predictions - y
        
        # --- 新增的详细信息打印 ---
        print(f"--- 第 {i+1} 次迭代 ---")
        print(f"  当前参数: θ₀={theta[0,0]:.4f}, θ₁={theta[1,0]:.4f}, θ₂={theta[2,0]:.4f}")
        for j in range(m):
            print(f"    样本 {j} (x₁={int(X[j,1])}, x₂={int(X[j,2])}): 预测值={predictions[j,0]:.3f}, 实际值={y[j,0]}, 误差={errors[j,0]:.3f}")
        print("-" * 40)
        # --- 打印结束 ---

        # 计算正则化梯度并更新theta
        theta_reg = theta.copy()
        theta_reg[0] = 0 
        gradient = (1 / m) * (X.T @ errors) + (lambda_val / m) * theta_reg
        theta = theta - alpha * gradient

        # 记录历史并打印总结
        new_cost = compute_cost_regularized(X, y, theta, lambda_val)
        cost_history.append(new_cost)
        theta_history.append(theta.flatten())
        print(f"第 {i+1} 次迭代总结:")
        print(f"  更新后参数: θ₀ = {theta[0,0]:.4f}, θ₁ = {theta[1,0]:.4f}, θ₂ = {theta[2,0]:.4f}")
        print(f"  正则化代价 J(θ) = {new_cost:.4f}\n")
            
    return cost_history, np.array(theta_history)

# --- 3. 绘图函数 ---
def plot_cost_history(cost_history, title):
    """绘制代价函数随迭代次数的变化"""
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(cost_history)), cost_history, marker='o')
    plt.title(title)
    plt.xlabel("迭代次数 (Iteration)")
    plt.ylabel("代价函数 J(θ)")
    plt.xticks(range(len(cost_history)))
    plt.grid(True)
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 设置数据和初始参数
    X_features = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([[13.5], [17.5], [23.5], [27.5], [33.5]])
    m = len(y)
    X = np.hstack([np.ones((m, 1)), X_features])
    
    initial_theta = np.array([[4.5], [1.5], [2.5]])
    alpha = 0.02
    lambda_param = 16
    iterations = 3

    # --- 标准线性回归 ---
    cost_hist, theta_hist = gradient_descent(X, y, initial_theta, alpha, iterations)
    plot_cost_history(cost_hist, "标准回归: 代价函数 J(θ) 随迭代次数变化")

    # --- 正则化线性回归 ---
    reg_cost_hist, reg_theta_hist = gradient_descent_regularized(
        X, y, initial_theta, alpha, iterations, lambda_param
    )
    # plot_cost_history(reg_cost_hist, "正则化回归: 代价函数 J(θ) 随迭代次数变化")