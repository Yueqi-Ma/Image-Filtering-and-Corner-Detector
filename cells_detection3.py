import cv2
import numpy as np
import os

def preprocess_image(image):
    # 在这里可以进行图像预处理，例如降噪、平滑和增强对比度等
    # 返回预处理后的图像
    return image

def detect_cells(image):
    # 生成尺度空间
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    scale_space = cv2.dft(blurred.astype(np.float32), flags=cv2.DFT_COMPLEX_OUTPUT)

    # 寻找局部极大值
    local_maxima = cv2.dilate(scale_space, None)
    local_maxima = (scale_space == local_maxima)

    # 设定阈值
    threshold = 0.8 # 根据实际情况调整阈值
    local_maxima = local_maxima.astype(np.float32)
    local_maxima = cv2.multiply(local_maxima, threshold)

    # 细胞计数
    num_cells = int(np.sum(local_maxima))

    # 标记细胞
    keypoints = cv2.KeyPoint_convert(np.argwhere(local_maxima == 1))  # 转换为KeyPoint对象
    image_with_markers = image.copy()
    for kp in keypoints:
        x, y = kp
        cv2.circle(image_with_markers, (int(x), int(y)), 5, (0, 0, 255), -1)

    return num_cells, image_with_markers

# 创建保存图像的文件夹
output_folder = "cell_detection"
os.makedirs(output_folder, exist_ok=True)

# 读取图像
image_paths = ['cells/001cell.png', 'cells/008cell.png', 'cells/014cell.png', 'cells/020cell.png']
images = []
for path in image_paths:
    image = cv2.imread(path)
    images.append(image)

# 处理和分析每个图像
results = []
for i, image in enumerate(images):
    preprocessed_image = preprocess_image(image)
    num_cells, result = detect_cells(preprocessed_image)
    results.append((num_cells, result))

    # 保存图像
    output_path = os.path.join(output_folder, f"cell_detection_{i+1}.png")
    cv2.imwrite(output_path, result)

    print(f"图像 {image_paths[i]} 中检测到的细胞数量：{num_cells}，结果已保存为 {output_path}")

