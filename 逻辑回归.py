import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 核心函数 ---

def sigmoid(z):
    """计算 Sigmoid 函数"""
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(X, y, theta):
    """计算逻辑回归的代价函数 J(θ)"""
    m = len(y)
    h = sigmoid(X @ theta)
    
    # 防止 log(0) 出现，添加一个极小值 epsilon
    epsilon = 1e-5
    
    term1 = y * np.log(h + epsilon)
    term2 = (1 - y) * np.log(1 - h + epsilon)
    
    cost = - (1 / m) * np.sum(term1 + term2)
    return cost

def gradient_descent_logistic(X, y, theta_start, alpha, num_iters):
    """
    执行逻辑回归的梯度下降，并显示每次迭代的详细信息。
    """
    m = len(y)
    theta = theta_start.copy()
    cost_history = []
    
    print("--- 开始逻辑回归梯度下降 ---")

    # 打印初始状态
    initial_cost = compute_cost_logistic(X, y, theta)
    print(f"初始状态 (迭代 0):")
    print(f"  参数: θ₀ = {theta[0,0]:.4f}, θ₁ = {theta[1,0]:.4f}, θ₂ = {theta[2,0]:.4f}")
    print(f"  代价函数 J(θ) = {initial_cost:.4f}\n")
    cost_history.append(initial_cost)

    for i in range(num_iters):
        # 1. 计算线性组合 z
        z = X @ theta
        
        # 2. 通过 Sigmoid 函数计算预测概率 h(x)
        h = sigmoid(z)
        
        # 3. 计算预测概率与实际值之间的误差
        errors = h - y

        # --- 新增的详细信息打印 ---
        print(f"--- 第 {i+1} 次迭代 ---")
        print(f"  当前参数: θ₀={theta[0,0]:.4f}, θ₁={theta[1,0]:.4f}, θ₂={theta[2,0]:.4f}")
        for j in range(m):
            print(f"    样本 {j} (x₁={X[j,1]}, x₂={X[j,2]}):")
            print(f"      - 线性组合 z = {z[j,0]:.4f}")
            print(f"      - S型函数 h(x) (预测概率) = {h[j,0]:.4f}")
            print(f"      - 实际值 y = {y[j,0]}")
            print(f"      - 误差 (h-y) = {errors[j,0]:.4f}")
        print("-" * 50)
        # --- 打印结束 ---

        # 4. 计算梯度
        gradient = (1 / m) * (X.T @ errors)
        
        # 5. 更新参数 theta
        theta = theta - alpha * gradient
        
        # 6. 记录历史并打印总结
        new_cost = compute_cost_logistic(X, y, theta)
        cost_history.append(new_cost)
        print(f"第 {i+1} 次迭代总结:")
        print(f"  更新后参数: θ₀ = {theta[0,0]:.4f}, θ₁ = {theta[1,0]:.4f}, θ₂ = {theta[2,0]:.4f}")
        print(f"  代价函数 J(θ) = {new_cost:.4f}\n")
        
    return cost_history, theta

# --- 2. 绘图函数 ---
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
    # 特征 [Exam1, Exam2]
    X_features = np.array([[1, 2], [2, 1], [3, 1.5], [3, 4], [4, 3], [2, 4]])
    # 目标 [Accepted] (0 or 1)
    y = np.array([[0], [0], [0], [1], [1], [1]])
    
    # 为X添加截距项 (一列1)
    m = len(y)
    X = np.hstack([np.ones((m, 1)), X_features])
    
    # 初始参数
    initial_theta = np.array([[-4.5], [1.5], [0.5]])
    alpha = 1.2
    # 为了演示，我们迭代几次，比如5次
    iterations = 5

    # 2. 运行梯度下降算法
    cost_hist, final_theta = gradient_descent_logistic(X, y, initial_theta, alpha, iterations)

    # 3. 绘制代价函数曲线
    plot_cost_history(cost_hist, "逻辑回归: 代价函数 J(θ) 随迭代次数变化")
    
    print("--- 最终结果 ---")
    print(f"经过 {iterations} 次迭代后的最终参数为:")
    print(f"θ₀ = {final_theta[0,0]:.4f}, θ₁ = {final_theta[1,0]:.4f}, θ₂ = {final_theta[2,0]:.4f}")