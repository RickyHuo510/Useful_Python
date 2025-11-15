import numpy as np
import matplotlib.pyplot as plt
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def linear_regression_gradient_descent(x, y, theta_0_initial, theta_1_initial, learning_rate, iterations):
    """
    使用梯度下降法执行单变量线性回归，并显示每次迭代的详细信息。

    参数:
    x (np.array): 输入特征 (人口)
    y (np.array): 输出变量 (利润)
    theta_0_initial (float): θ₀ 的初始值
    theta_1_initial (float): θ₁ 的初始值
    learning_rate (float): 学习率 α
    iterations (int): 迭代次数

    返回:
    tuple: 包含代价函数历史、θ₀ 历史和 θ₁ 历史的元组
    """
    m = len(y)  # 样本数量
    theta_0 = theta_0_initial
    theta_1 = theta_1_initial

    # 用于存储每次迭代的结果
    cost_history = []
    theta_0_history = []
    theta_1_history = []

    print("--- 开始梯度下降 ---")
    
    # 打印初始状态
    initial_cost = calculate_cost(x, y, theta_0, theta_1)
    print(f"初始状态 (迭代 0):")
    print(f"  参数: θ₀ = {theta_0}, θ₁ = {theta_1}")
    print(f"  代价函数 J(θ) = {initial_cost:.4f}\n")
    cost_history.append(initial_cost)
    theta_0_history.append(theta_0)
    theta_1_history.append(theta_1)


    for i in range(iterations):
        # 1. 计算当前参数下的预测值 h(x)
        predictions = theta_0 + theta_1 * x

        # 2. 计算预测值与真实值之间的误差
        errors = predictions - y

        # --- 新增的详细信息打印 ---
        print(f"--- 第 {i+1} 次迭代 ---")
        print(f"  当前参数: θ₀ = {theta_0:.4f}, θ₁ = {theta_1:.4f}")
        # 打印每个样本的预测值和误差
        for j in range(m):
            print(f"    样本 {j} (x={x[j]}): 预测值={predictions[j]:.3f}, 实际值={y[j]}, 误差={errors[j]:.3f}")
        print("-" * 20)
        # --- 打印结束 ---

        # 3. 计算梯度 (代价函数对 θ₀ 和 θ₁ 的偏导数)
        gradient_0 = (1 / m) * np.sum(errors)
        gradient_1 = (1 / m) * np.sum(errors * x)

        # 4. 同步更新 θ₀ 和 θ₁
        theta_0 = theta_0 - learning_rate * gradient_0
        theta_1 = theta_1 - learning_rate * gradient_1

        # 5. 计算并记录新的代价函数值和更新后的 theta
        new_cost = calculate_cost(x, y, theta_0, theta_1)
        cost_history.append(new_cost)
        theta_0_history.append(theta_0)
        theta_1_history.append(theta_1)

        # 打印本次迭代的总结结果
        print(f"第 {i+1} 次迭代总结:")
        print(f"  更新后参数: θ₀ = {theta_0:.4f}, θ₁ = {theta_1:.4f}")
        print(f"  代价函数 J(θ) = {new_cost:.4f}\n")
        

    # 移除第一次记录的初始值，因为循环外已经加了
    return cost_history, theta_0_history, theta_1_history

def calculate_cost(x, y, theta_0, theta_1):
    """计算代价函数 J(θ) 的值"""
    m = len(y)
    predictions = theta_0 + theta_1 * x
    squared_errors = (predictions - y) ** 2
    cost = (1 / (2 * m)) * np.sum(squared_errors)
    return cost

def plot_results(x, y, cost_history, theta_0_hist, theta_1_hist):
    """可视化结果"""
    iterations = len(cost_history) -1
    
    # 创建一个图和两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 子图1: 代价函数随迭代次数的变化
    ax1.plot(range(iterations + 1), cost_history, marker='o')
    ax1.set_title('代价函数 J(θ) 随迭代次数的变化')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('代价函数 J(θ)')
    ax1.set_xticks(range(iterations + 1))
    ax1.grid(True)

    # 子图2: 数据散点图和最终的回归线
    ax2.scatter(x, y, color='red', marker='x', label='训练数据')
    # 生成回归线的点
    line_x = np.linspace(x.min(), x.max(), 100)
    final_theta_0 = theta_0_hist[-1]
    final_theta_1 = theta_1_hist[-1]
    line_y = final_theta_0 + final_theta_1 * line_x
    ax2.plot(line_x, line_y, color='blue', label=f'回归线: y={final_theta_0:.2f}+{final_theta_1:.2f}x')
    ax2.set_title('数据与线性回归拟合')
    ax2.set_xlabel('人口')
    ax2.set_ylabel('利润')
    ax2.legend()
    ax2.grid(True)

    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 根据题目提供的数据和参数
    population = np.array([30, 50, 60, 85])
    profit = np.array([42.4, 64.6, 71.9, 88.2])
    
    # 已知初始参数
    initial_theta_0 = 3
    initial_theta_1 = 4
    learning_rate_alpha = 0.01
    num_iterations = 1

    # 2. 运行梯度下降算法
    cost_hist, theta_0_hist, theta_1_hist = linear_regression_gradient_descent(
        population,
        profit,
        initial_theta_0,
        initial_theta_1,
        learning_rate_alpha,
        num_iterations
    )

    # 3. 绘制结果图表
    plot_results(population, profit, cost_hist, theta_0_hist, theta_1_hist)