import tarfile
import pickle
import numpy as np
import os
# [PPT P29-32] 梯度下降与优化器需要用到 TensorFlow 框架
import tensorflow.keras as keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# 环境配置与数据加载部分
# ==========================================
# 解决 Windows 下找不到编译器的问题
conda_base = r"C:\Users\RickyHuo\miniconda3\envs\tensorflow" 
conda_lib_bin = os.path.join(conda_base, "Library", "bin")
os.environ['PATH'] = conda_lib_bin + os.pathsep + os.environ['PATH']

LOCAL_CIFAR_PATH = 'cnn/cifar-100-python.tar.gz' 
MODEL_PATH='cnn/cifar100_cnn_model.h5'

def load_cifar100_with_labels(filepath):
    """
    读取数据，同时提取类别名称
    [PPT P17] 输入层构建：将图像数据转换为矩阵形式
    """
    print(f"正在读取本地文件: {filepath} ...")
    with tarfile.open(filepath, 'r:gz') as tar:
        def unpickle(member_name):
            f = tar.extractfile(member_name)
            if f is None: raise ValueError(f"找不到文件: {member_name}")
            return pickle.load(f, encoding='bytes')

        train_data = unpickle('cifar-100-python/train')
        test_data = unpickle('cifar-100-python/test')
        meta_data = unpickle('cifar-100-python/meta')
        label_names = [x.decode('utf-8') for x in meta_data[b'fine_label_names']]

    x_train = train_data[b'data']
    y_train = np.array(train_data[b'fine_labels'])
    x_test = test_data[b'data']
    y_test = np.array(test_data[b'fine_labels'])

    # [PPT P66] 卷积层的输入必须是三维张量 (高度 H x 宽度 W x 深度 D)
    # 这里的 D=3 (RGB通道)，对应 PPT P65 "如果是彩色图像，分别有RGB三个颜色通道"
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # 归一化 (0-1)，利于梯度下降收敛 [PPT P29]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    return (x_train, y_train), (x_test, y_test), label_names

# ==========================================
# === 核心部分：CNN 模型定义 (结合 PPT 讲解) ===
# ==========================================
# [PPT P72] 卷积网络的整体结构：由卷积层、汇聚层、全连接层交叉堆叠而成。
model = models.Sequential()

# --- 第一组卷积块 (特征提取) ---

# [PPT P37-38] 卷积层：模拟生物学上的"感受野"机制，只接受局部区域的信号。
# [PPT P41-42] 二维卷积定义：输入信息 X 和 滤波器 W 进行运算。
# [PPT P48] 零填充 (Zero Padding)：padding='same' 对应 PPT 中的补零，保证输出图像大小不变。
# filters=64: 对应 [PPT P65] "可以在每一层使用多个不同的特征映射"，这里我们用了64个卷积核提取特征。
model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False, input_shape=(32, 32, 3))) 

# [PPT P48] 再次卷积，加深网络。深度增加有助于提取更高级特征 (如从边缘到形状)。
model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False)) 

# BatchNormalization (批归一化)：虽然 PPT 未详细展开，但在 [PPT P87] Inception v3 部分提到了"批量归一化"。
# 它的作用是加速训练，让梯度下降更稳定。
model.add(layers.BatchNormalization()) 

# [PPT P67] 激活函数：一般用 ReLU 函数。
# 对应公式 Y = f(Z)，引入非线性，让神经网络能逼近复杂函数 (PPT P88 残差网络也强调了非线性单元)。
model.add(layers.Activation('relu'))
    
model.add(layers.Conv2D(64, (3, 3), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))

# [PPT P70-71] 汇聚层 (Pooling Layer): 也叫子采样层。
# 作用：特征选择，降低特征数量，减少参数。
# (2, 2) 代表将 2x2 的区域合并成一个值，这里默认是 Max Pooling (PPT P71 最大汇聚)。
model.add(layers.MaxPooling2D((2, 2))) 

# [PPT P83] Dropout: 在 AlexNet 中首次引入。
# 作用：防止过拟合。随机让一部分神经元"失活"，对应 PPT P38 提到的"减少参数"思想的延伸。
model.add(layers.Dropout(0.2))         

# --- 第二组卷积块 (提取形状特征) ---
# [PPT P64] 权重共享：这里使用了 128 个卷积核，比上一层更多。
# 随着网络加深，特征图(Feature Map)变小(经过了Pooling)，但深度(Depth)增加。
model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
    
model.add(layers.Conv2D(128, (3, 3), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
    
# 再次下采样 [PPT P70]
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Dropout(0.3))         

# --- 第三组卷积块 (提取语义特征) ---
# 卷积核增加到 256 个
model.add(layers.Conv2D(256, (3, 3), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
    
model.add(layers.Conv2D(256, (3, 3), padding='same', use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
    
# 再次下采样
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Dropout(0.4))

# --- 分类头 (全连接层) ---

# [PPT P18] 输入层与输出层之间的转换
# Flatten 将三维的特征图 (高度x宽度x深度) 展平成一维向量，
# 准备输入到全连接层 (Fully Connected Layer)。
model.add(layers.Flatten())
    
# [PPT P15] 完全连接前馈网络：典型的多层神经网络结构。
# 这里有 512 个神经元。
model.add(layers.Dense(512, use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5)) 

# [PPT P17] 输出层：
# "每一维都代表了一个数字的置信度"。
# 这里有 100 个神经元，对应 CIFAR-100 的 100 个分类。
# activation='softmax' [PPT P83 AlexNet]：将输出转化为概率分布。
model.add(layers.Dense(100, activation='softmax'))

# [PPT P25-26] 损失函数与优化：
# loss='sparse_categorical_crossentropy': 对应 PPT P25 的"交叉熵"，用于分类问题。
# optimizer='adam': 对应 PPT P29-33 的"梯度下降法"的高级变种。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ==========================================
# 训练与交互逻辑
# ==========================================

# [PPT P26] "从函数集中寻找使得 L 最小的函数" -> 这一步就是 model.fit 训练过程
if os.path.exists(MODEL_PATH):
    model = models.load_model(MODEL_PATH)
    (x_train, y_train), (x_test, y_test), label_names = load_cifar100_with_labels(LOCAL_CIFAR_PATH)
else:
    (x_train, y_train), (x_test, y_test), label_names = load_cifar100_with_labels(LOCAL_CIFAR_PATH)
    print(f"数据加载完成。类别示例: {label_names[:5]}...")
    print("开始训练...")
    # [PPT P22] 深度神经网络的训练方法：通过多轮 (epochs) 迭代，
    # 不断进行前向传播和反向传播 [PPT P74-76] 来更新参数。
    model.fit(x_train, y_train, epochs=50, batch_size=256, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def show_random_image(nonblocking=True):
    # [PPT P4] 回顾：学习基本框架 - 测试
    # 这里模拟了 PPT P4 的测试过程：输入 x' -> f*(x') -> y' (预测标签)
    random_idx = np.random.randint(0, len(x_test))
    sample_image = x_test[random_idx]
    true_label_idx = y_test[random_idx]
    true_label_name = label_names[true_label_idx]

    img_for_model = np.expand_dims(sample_image, axis=0)
    predictions = model.predict(img_for_model) # 前向传播计算预测值
    predicted_idx = np.argmax(predictions)
    predicted_name = label_names[predicted_idx]
    confidence = np.max(predictions) * 100

    plt.figure(figsize=(4, 4))
    plt.imshow(sample_image)
    plt.axis('off')
    title_color = 'green' if predicted_idx == true_label_idx else 'red'
    plt.title(f"真实: {true_label_name}\n预测: {predicted_name} ({confidence:.1f}%)", 
              color=title_color, fontsize=14)

    if nonblocking:
        plt.show(block=False)
        plt.pause(0.1)
    else:
        plt.show()

def interactive_predict_loop():
    print("按 Enter 显示下一张图片，输入 'q' 然后回车退出。")
    show_random_image(nonblocking=True)

    while True:
        try:
            s = input()
        except (EOFError, KeyboardInterrupt):
            plt.close('all')
            print("退出。")
            break

        if s.strip().lower() in ('q', 'quit', 'exit'):
            plt.close('all')
            print("退出。")
            break

        plt.close('all')
        show_random_image(nonblocking=True)

interactive_predict_loop()