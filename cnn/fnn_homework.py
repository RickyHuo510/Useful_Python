import numpy as np

class BPNeuralNetwork:
    def __init__(self, layers):
        """
        初始化神经网络
        layers: 一个列表，表示每层的神经元数量（不包括偏置），例如 [3, 2, 2, 1]
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.thetas = {} # 存储权重矩阵
        self.activations = {} # 存储每一层的输出 a
        self.z_values = {} # 存储每一层的线性输入 z
        self.deltas = {} # 存储误差项 delta
        self.gradients = {} # 存储梯度
        
        # 随机初始化权重
        for i in range(self.num_layers - 1):
            self.thetas[i+1] = np.random.rand(layers[i+1], layers[i] + 1)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, a):
        return a * (1 - a)

    def set_weights(self, layer_idx, weights):
        self.thetas[layer_idx] = weights

    def forward_propagation(self, x):
        """前向传播"""
        a = np.array(x).reshape(-1, 1)
        # 输入层加偏置
        a_with_bias = np.vstack(([1.0], a))
        self.activations[1] = a_with_bias
        
        for l in range(1, self.num_layers):
            theta = self.thetas[l]
            prev_a = self.activations[l]
            
            z = np.dot(theta, prev_a)
            self.z_values[l+1] = z
            
            a = self.sigmoid(z)
            
            # 隐藏层加偏置，输出层不加
            if l < self.num_layers - 1:
                a = np.vstack(([1.0], a))
            
            self.activations[l+1] = a
            
        return self.activations[self.num_layers]

    def back_propagation(self, y):
        """反向传播"""
        L = self.num_layers
        a_last = self.activations[L]
        
        # 1. 计算输出层误差
        delta_last = a_last - y
        self.deltas[L] = delta_last
        
        # 2. 计算隐藏层误差 (从 L-1 到 2)
        for l in range(L-1, 1, -1):
            delta_next = self.deltas[l+1]
            
            # 【修复关键点】：如果下一层是隐藏层，delta包含了偏置项误差。
            # 偏置项不接受来自上一层的输入（它是常量），所以它的误差不反向传播。
            # 我们只需要下一层“真实神经元”的误差。
            # 输出层(Layer L)没有加偏置，所以不需要切片；隐藏层(Layer < L)加了偏置，需要切片[1:]
            if l + 1 < L:
                delta_next = delta_next[1:]
                
            theta = self.thetas[l]
            a_curr = self.activations[l]
            
            # theta.T (4x2) * delta_next (2x1) -> (4x1)
            error_prop = np.dot(theta.T, delta_next)
            sig_prime = self.sigmoid_derivative(a_curr)
            
            delta_curr = error_prop * sig_prime
            self.deltas[l] = delta_curr

    def calculate_gradients(self):
        """计算梯度"""
        L = self.num_layers
        for l in range(1, self.num_layers):
            delta_next = self.deltas[l+1]
            
            # 【修复关键点】：同样，计算梯度时，只关注下一层真实神经元的误差
            if l + 1 < L:
                delta_next = delta_next[1:]
            
            a_curr = self.activations[l]
            
            # delta (n_out, 1) * a_curr.T (1, n_in+1) -> (n_out, n_in+1)
            grad = np.dot(delta_next, a_curr.T)
            self.gradients[l] = grad

    def update_weights(self, alpha):
        for l in range(1, self.num_layers):
            self.thetas[l] = self.thetas[l] - alpha * self.gradients[l]

    def train_step_verbose(self, x, y, alpha):
        print("="*20 + " 开始计算 " + "="*20)
        
        # 1. 前向传播
        output = self.forward_propagation(x)
        print(f"\n【前向传播】")
        for l in range(1, self.num_layers + 1):
            print(f"Layer {l} Activations (a^({l})): \n{self.activations[l].flatten()}")
            # 修复之前的 KeyError: 只打印存在的 z 值
            if (l + 1) in self.z_values:
                print(f"Layer {l} -> {l+1} (z^({l+1})): \n{self.z_values[l+1].flatten()}")
        
        print(f"\n预测输出 (y_hat): {output.flatten()[0]:.6f}")
        print(f"真实标签 (y): {y}")
        
        # 2. 反向传播
        self.back_propagation(y)
        print(f"\n【反向传播 (误差项 Delta)】")
        # 注意：这里的 Delta 包含了偏置项的计算结果，虽然在传播时被切除，但在公式中是存在的
        print(f"Delta^({self.num_layers}): \n{self.deltas[self.num_layers].flatten()}")
        for l in range(self.num_layers - 1, 1, -1):
            print(f"Delta^({l}): \n{self.deltas[l].flatten()}")
            
        # 3. 计算梯度
        self.calculate_gradients()
        print(f"\n【梯度 (∂J/∂Θ)】")
        for l in range(1, self.num_layers):
            print(f"Gradient Theta{l}: \n{self.gradients[l]}")

        # 4. 更新权重
        self.update_weights(alpha)
        print(f"\n【更新后的权重 (New Theta)】")
        for l in range(1, self.num_layers):
            print(f"Theta{l}: \n{self.thetas[l]}")
            
        print("="*50)

# ==========================================
# 主程序
# ==========================================

nn = BPNeuralNetwork([3, 2, 2, 1])

# Theta1: 2x4
theta1_init = np.array([
    [-0.2000, 0.4000, 0.5000, 0.1000],
    [-0.3500, 0.4000, 0.1000, -0.2000]
])

# Theta2: 2x3
theta2_init = np.array([
    [0.7000, -0.5000, 0.2000],
    [-0.3000, 0.4000, 0.6000]
])

# Theta3: 1x3
theta3_init = np.array([
    [0.5000, -0.2000, 0.3000]
])

nn.set_weights(1, theta1_init)
nn.set_weights(2, theta2_init)
nn.set_weights(3, theta3_init)

x_input = [0.6000, 0.8000, 0.4000]
y_target = 1
alpha = 2.5000

nn.train_step_verbose(x_input, y_target, alpha)