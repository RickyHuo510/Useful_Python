import numpy as np
import numpy as np
import matplotlib.pyplot as plt

# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def kmeans(X, initial_centroids, max_iters=10):
    """
    执行 K-Means 聚类算法。
    
    参数:
    X (np.array): 数据集, shape=(m, n)
    initial_centroids (np.array): 初始聚类中心, shape=(K, n)
    max_iters (int): 最大迭代次数
    
    返回:
    tuple: (最终的聚类中心, 每个数据点的分配)
    """
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids.copy()
    
    print("--- 开始 K-Means 迭代 ---\n")
    
    for i in range(max_iters):
        print(f"--- 第 {i+1} 次迭代 ---")
        
        # 1. 分配步骤 (Assignment)
        distances = np.zeros((m, K))
        for k in range(K):
            # 计算所有点到第k个聚类中心的欧氏距离
            distances[:, k] = np.linalg.norm(X - centroids[k], axis=1)
        
        # 找到每个点最近的聚类中心索引
        assignments = np.argmin(distances, axis=1)
        
        # 打印分配结果
        for k in range(K):
            points_in_cluster = np.where(assignments == k)[0] + 1
            print(f"  与 μ{k+1} {centroids[k]} 对应的点: {list(points_in_cluster)}")

        # 2. 更新步骤 (Update)
        new_centroids = np.zeros((K, n))
        for k in range(K):
            # 找到属于当前簇的所有点
            points_in_cluster = X[assignments == k]
            # 计算这些点的均值作为新的聚类中心
            if len(points_in_cluster) > 0:
                new_centroids[k] = np.mean(points_in_cluster, axis=0)

        print("\n  计算出的新聚类中心:")
        for k in range(K):
            print(f"    新 μ{k+1}: {np.round(new_centroids[k], 2)}")
        print("-" * 30)

        # 3. 检查是否收敛
        if np.allclose(centroids, new_centroids):
            print(f"\n在第 {i+1} 次迭代后，聚类中心不再变化，算法收敛。")
            break
            
        centroids = new_centroids

    return centroids, assignments

def plot_clusters(X, assignments, centroids, point_labels):
    """可视化聚类结果"""
    K = centroids.shape[0]
    colors = ['#1f77b4', '#ff7f0e'] # 蓝色和橙色
    
    plt.figure(figsize=(10, 8))
    
    for k in range(K):
        # 绘制属于第k簇的点
        cluster_points = X[assignments == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    c=colors[k], s=100, label=f'聚类 {k+1}')
    
    # 标注数据点编号和坐标
    for i, label in enumerate(point_labels):
        plt.text(X[i, 0] + 0.1, X[i, 1] + 0.1, label, fontsize=12)

    # 绘制最终的聚类中心
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                c='red', s=200, marker='X', label='最终聚类中心')
    # 标注聚类中心
    for k in range(K):
         plt.text(centroids[k, 0] + 0.1, centroids[k, 1] + 0.1, 
                  f'μ{k+1} ({centroids[k,0]:.1f},{centroids[k,1]:.1f})', 
                  color='red', fontsize=14)

    plt.title('K-Means 聚类结果')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.grid(True)
    plt.legend()
    plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    # 题目给定的数据
    data_points = np.array([
        [1, 3], [3, 3], [5, 4], [6, 9], [10, 7], [12, 8]
    ])
    point_labels = [f"{i+1}({int(p[0])},{int(p[1])})" for i, p in enumerate(data_points)]

    initial_centroids = np.array([
        [4, 4], [6, 5]
    ])

    # 执行K-Means算法
    final_centroids, final_assignments = kmeans(data_points, initial_centroids)

    # 打印最终结果
    print("\n" + "="*30)
    print("最终聚类结果:")
    for k in range(final_centroids.shape[0]):
        points_in_cluster = np.where(final_assignments == k)[0] + 1
        print(f"  聚类中心 {k+1}: {np.round(final_centroids[k], 2)}, "
              f"包含数据点: {list(points_in_cluster)}")
    print("="*30)
    
    # 可视化结果
    plot_clusters(data_points, final_assignments, final_centroids, point_labels)