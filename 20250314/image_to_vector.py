from PIL import Image
import numpy as np

def image_to_vector(image_path):
    # 打开图像
    image = Image.open(image_path)
    # 将图像转换为 numpy 数组
    image_array = np.array(image)
    # 将数组转换为一维向量
    vector = image_array.flatten()
    return vector

# 示例用法
image_path = 'example.jpg'  # 替换为你的图像文件路径
vector = image_to_vector(image_path)
print("一维向量的长度:", len(vector))
print("一维向量的前 10 个元素:", vector[:10])
    