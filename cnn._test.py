import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as keras
from tensorflow.keras import layers, models, datasets
from PIL import Image, ImageDraw
import os

# ==========================================
# 第一部分：CNN 模型定义与训练/加载
# ==========================================
MODEL_PATH = 'mnist_cnn_model.h5'

def create_model():
    """创建一个简单的卷积神经网络 (CNN)"""
    model = models.Sequential([
        # 卷积层：提取特征（如边缘、线条）
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)), # 池化层：降低维度，保留主要特征
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),            # 展平：将二维图像转为一维向量
        layers.Dense(64, activation='relu'), # 全连接层
        layers.Dense(10, activation='softmax') # 输出层：10个数字的概率
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_or_train_model():
    """如果模型文件存在则加载，否则训练新模型"""
    if os.path.exists(MODEL_PATH):
        print("正在加载已保存的模型...")
        model = models.load_model(MODEL_PATH)
    else:
        print("未发现模型，开始训练 (这可能需要一两分钟)...")
        # 加载 MNIST 数据集
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        
        # 归一化 (0-255 -> 0-1) 并调整形状以适应 CNN (N, 28, 28, 1)
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

        model = create_model()
        # 训练模型 (Epochs设为5即可达到很高精度)
        model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)
        
        # 保存模型
        model.save(MODEL_PATH)
        print("模型训练完成并保存。")
    
    return model

# ==========================================
# 第二部分：手写板 GUI 程序
# ==========================================
class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("神经网络手写数字识别 (CNN)")
        self.model = model
        
        # 画布设置
        self.canvas_width = 280
        self.canvas_height = 280
        self.bg_color = 'black'
        self.paint_color = 'white'
        self.brush_size = 15  # 笔刷要够粗，因为后面会缩小图片
        
        # 创建 Tkinter 画布用于显示
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, 
                                bg=self.bg_color, cursor="cross")
        self.canvas.pack(pady=10)
        
        # 创建 PIL Image 对象用于后台处理 (与画布同步)
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0) # 'L' 模式为灰度，0为黑
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)  # 鼠标拖动
        self.canvas.bind("<ButtonRelease-1>", self.predict) # 鼠标松开时预测
        
        # 结果标签
        self.label_result = tk.Label(root, text="请在上方书写数字", font=("Helvetica", 24))
        self.label_result.pack(pady=10)
        
        # 按钮区域
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="清除", command=self.clear_canvas, font=("Helvetica", 14)).pack()

    def paint(self, event):
        """处理绘图逻辑"""
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        
        # 在 GUI 画布上画
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline=self.paint_color)
        
        # 在内存中的 PIL 图片上画
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)
        
        # 实时预测 (可选：如果电脑慢，可以将这行注释掉，只在松开鼠标时预测)
        self.predict(event)

    def clear_canvas(self):
        """清空画布"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="请在上方书写数字")

    def predict(self, event):
        """处理图像并进行预测"""
        # 1. 调整大小：将 280x280 的画布压缩到 28x28 (MNIST 标准尺寸)
        img_resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # 2. 转换为 numpy 数组
        img_array = np.array(img_resized)
        
        # 3. 预处理：归一化并reshape (1, 28, 28, 1)
        img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        # 4. 预测
        prediction = self.model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        # 5. 更新显示
        self.label_result.config(text=f"识别结果: {predicted_digit} (置信度: {confidence:.2f})")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # 第一次运行时会自动下载数据集并训练，大概需要1-2分钟
    cnn_model = load_or_train_model()
    
    # 启动 GUI
    root = tk.Tk()
    app = DigitRecognizerApp(root, cnn_model)
    root.mainloop()