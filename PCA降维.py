import numpy as np

def perform_pca(X, n_components=1):
    """
    对给定的数据集 X 执行 PCA 降维。
    
    参数:
    X (np.array): 原始数据矩阵，每一列是一个样本 (n_features, m_samples)
    n_components (int): 希望降到的维度
    
    返回:
    tuple: (降维后的数据, 用于降维的特征向量, 协方差矩阵的特征值)
    """
    
    print("--- 原始数据矩阵 X ---")
    print(X)
    print("\n" + "="*50)
    
    # --- 步骤 1: 对矩阵按行进行去均值化 ---
    print("步骤 1: 对矩阵按行进行去均值化")
    
    # 计算每一行 (每一个维度) 的均值
    row_means = np.mean(X, axis=1, keepdims=True)
    print("  - 计算得到每行的均值:\n", row_means)
    
    # 从每一行的元素中减去该行的均值
    X_centered = X - row_means
    print("  - 去均值化后的矩阵 X_centered:\n", X_centered)
    # 验证: 打印去均值化后每行的均值，应接近于0
    print("  - 验证: 新矩阵的行均值:\n", np.mean(X_centered, axis=1))
    print("\n" + "="*50)

    # --- 步骤 2: 计算协方差矩阵及其特征值和特征向量 ---
    print("步骤 2: 计算协方差矩阵及其特征值和特征向量")
    
    # 获取样本数量 m
    m = X_centered.shape[1]
    
    # 计算协方差矩阵 C = (1/m) * X_centered * X_centered.T
    # 注意：这里使用 (m-1) 作为分母是计算“样本协方差”的无偏估计，
    # 在很多库（如np.cov）中是默认行为。使用 m 也是可以的，最终方向向量不变。
    # 我们按照 np.cov 的标准来计算。
    cov_matrix = (1 / (m)) * (X_centered @ X_centered.T)
    # 或者直接使用 numpy 的 cov 函数
    #cov_matrix = np.cov(X_centered)
    
    print("  - 计算得到协方差矩阵 C:\n", cov_matrix)
    
    # 计算协方差矩阵的特征值和特征向量
    # np.linalg.eig 返回的特征值不一定按大小排序
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    print("\n  - 计算得到特征值 (Eigenvalues):\n", eigenvalues)
    print("  - 计算得到特征向量 (Eigenvectors):\n", eigenvectors)
    
    # 对特征值进行降序排序，并获取排序后的索引
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # 根据排序后的索引，重新排列特征值和特征向量
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    print("\n  - 按特征值大小降序排列后:")
    print("    - 排序后的特征值:\n", sorted_eigenvalues)
    print("    - 对应的特征向量:\n", sorted_eigenvectors)
    print("\n" + "="*50)
    
    # --- 步骤 3: 使用特征向量将原2维矩阵降维1维向量 ---
    print(f"步骤 3: 选择前 {n_components} 个主成分进行降维")
    
    # 选取前 n_components 个最大的特征值对应的特征向量
    projection_matrix = sorted_eigenvectors[:, :n_components]
    print(f"  - 选择的主成分 (最重要的特征向量):\n", projection_matrix)
    
    # 将去均值化后的数据投影到选定的主成分上
    # Z = U.T * X_centered
    Z_reduced = projection_matrix.T @ X_centered
    
    print("\n  - 将去均值化后的数据 X_centered 投影到主成分上:")
    print("  - 计算 Z = U.T * X_centered")
    print("  - 得到降维后的1维数据 Z:\n", Z_reduced)
    print("\n" + "="*50)
    
    return Z_reduced, projection_matrix, sorted_eigenvalues

# --- 主程序 ---
if __name__ == "__main__":
    # 根据题目提供的数据创建矩阵
    # 每一列是一个样本
    X_data = np.array([
        [5, 8, 10, 1, 15],
        [8, 6, 6, 7, 9]
    ])

    # 调用 PCA 函数，降维到1维
    Z_final, U_final, lambdas_final = perform_pca(X_data, n_components=1)

    print("--- PCA 降维完成 ---")
    print("最终降维结果 Z:\n", Z_final)