import tensorflow as tf

print("TensorFlow 版本:", tf.__version__)

# 列出所有的物理设备
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"成功！发现 {len(gpus)} 个 GPU:")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("失败：未发现 GPU，将使用 CPU 运行。")