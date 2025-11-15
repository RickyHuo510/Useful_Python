import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import time

def calculate_wcss(X, labels, centroids):
    """
    计算优化目标函数：簇内平方和 (WCSS)
    """
    wcss = 0
    for i in range(len(centroids)):
        # 找出属于当前簇 i 的所有点
        points_in_cluster = X[labels == i]
        # 计算这些点到其中心点的距离平方和
        if len(points_in_cluster) > 0:
            wcss += np.sum((points_in_cluster - centroids[i])**2)
    return wcss

def k_means(X, n_clusters, n_init=10, max_iter=300, random_state=None):
    """
    完整的K-Means算法实现，支持多次初始化。

    参数:
    X: numpy数组, shape (n_samples, n_features)，待聚类的数据。
    n_clusters: int, 聚类的目标数量 (k)。
    n_init: int, 使用不同初始质心运行算法的次数。
    max_iter: int, 单次运行的最大迭代次数。
    random_state: int, 用于生成可复现的随机结果。

    返回:
    best_centroids: 最优一次运行的最终质心。
    best_labels: 最优一次运行的最终分类标签。
    best_wcss: 最优一次运行的最小WCSS值。
    """
    # 设置随机数种子
    np.random.seed(random_state)
    
    best_wcss = float('inf')
    best_centroids = None
    best_labels = None

    # --- 多次运行K-Means以获得更好的结果 ---
    for run in range(n_init):
        print(f"\n================ K-Means Run #{run + 1}/{n_init} ================")
        
        # 1. 初始化: 随机从数据点中选择 k 个作为初始质心
        initial_indices = np.random.choice(X.shape[0], n_clusters, replace=False)
        centroids = X[initial_indices]
        
        # --- 单次运行的迭代过程 ---
        for i in range(max_iter):
            print(f"\n--- Run {run + 1}, Iteration {i + 1} ---")
            
            # 保存旧的质心用于比较
            old_centroids = np.copy(centroids)

            # 2. 分配步骤: 将每个点分配到最近的质心
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # 打印当前的分类情况 (为了简洁，只显示前10个点的分类)
            print(f"Current classifications (first 10 points): {labels[:10]}")

            # 3. 更新步骤: 重新计算每个簇的质心
            for j in range(n_clusters):
                points_in_cluster = X[labels == j]
                # 如果一个簇变空了，则保持其质心不变（也可以选择重新初始化）
                if len(points_in_cluster) > 0:
                    centroids[j] = points_in_cluster.mean(axis=0)

            # 计算并打印优化目标函数的值
            wcss = calculate_wcss(X, labels, centroids)
            print(f"Objective Function (WCSS): {wcss:.4f}")
            
            # 4. 检查终止条件: 质心是否不再变化
            if np.allclose(old_centroids, centroids):
                print(f"Convergence reached at iteration {i + 1}. Centroids are stable.")
                break
        
        # 保存本次运行的最佳结果
        if wcss < best_wcss:
            print(f"\nFound new best result in Run #{run + 1} with WCSS = {wcss:.4f}")
            best_wcss = wcss
            best_centroids = centroids
            best_labels = labels
            
    return best_centroids, best_labels, best_wcss

def visualize_clusters(X, labels, centroids, title):
    """
    可视化聚类结果
    """
    plt.figure(figsize=(8, 6))
    # 使用不同的颜色绘制每个簇的数据点
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('viridis', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        plt.scatter(X[labels == label, 0], X[labels == label, 1], 
                    color=colors(i), label=f'Cluster {label + 1}', alpha=0.7)
    
    # 用一个明显的标记绘制最终的质心
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', 
                marker='X', label='Centroids', edgecolors='black')
    
    plt.title(title, fontsize=15)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 生成随机数据
    # 创建一个包含4个明显簇的2D数据集
    n_samples = 300
    n_features = 2
    n_clusters_true = 4 # 我们期望算法能找到4个簇
    
    X, y_true = make_blobs(n_samples=n_samples, 
                           centers=n_clusters_true, 
                           n_features=n_features,
                           cluster_std=0.8, 
                           random_state=int(time.time()))

    # 2. 运行 K-Means 算法
    # 我们设置 n_init=3 来运行3次，以选择最好的结果
    k = 4
    final_centroids, final_labels, final_wcss = k_means(
        X, n_clusters=k, n_init=5, random_state=42
    )

    # 3. 打印最终的最优结果
    print("\n\n================== FINAL OPTIMAL RESULT ==================")
    print(f"Optimal WCSS found: {final_wcss:.4f}")
    print("Final Centroids:\n", final_centroids)
    
    # 4. 可视化最终的聚类结果
    visualize_clusters(
        X, final_labels, final_centroids, 
        f'K-Means Clustering Final Result (k={k})'
    )