import cv2
import numpy as np

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

    # 可视化结果
    keypoints = cv2.KeyPoint_convert(np.argwhere(local_maxima == 1))  # 转换为KeyPoint对象
    result = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return num_cells, result

# 读取图像
image_paths = ['cells/001cell.png', 'cells/008cell.png', 'cells/014cell.png', 'cells/020cell.png']
images = []
for path in image_paths:
    image = cv2.imread(path)
    images.append(image)

# 处理和分析每个图像
results = []
for image in images:
    preprocessed_image = preprocess_image(image)
    num_cells, result = detect_cells(preprocessed_image)
    results.append((num_cells, result))

# 打印细胞计数结果和可视化图像
for i, (num_cells, result) in enumerate(results):
    image_path = image_paths[i]
    print(f"图像 {image_path} 中检测到的细胞数量：{num_cells}")
    cv2.imshow(f"Cell Detection - {image_path}", result)

cv2.waitKey(0)
cv2.destroyAllWindows()