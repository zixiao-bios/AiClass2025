from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import numpy as np

# 获取当前源代码文件所在路径
current_dir_path = Path(__file__).resolve().parent

# 输入图片的路径
img_path = current_dir_path / 'imgs' / 'traffic.jpg'

# yolo 模型权重的路径
yolo_weight_dir = '/home/qrobo/yolo/weights/yolov8l.pt'

def main():
    # 实例化YOLO模型，加载权重
    model = YOLO(yolo_weight_dir)

    # 读取本地图片，转为 RGB 模式
    input_img = Image.open(img_path).convert('RGB')

    # 将图片转为ndarray形式（矩阵）
    img_array = np.array(input_img)

    # 调用模型进行目标检测，测到结果（长度为1的数组，因此取索引为0的元素）
    result = model.predict(input_img)[0]

    # model.names 中保存了类别索引->名称的映射
    print('======================model.names======================')
    print(model.names)

    # 从结果中取出边界框
    boxes = result.boxes
    print('======================result.boxes======================')
    print(boxes)

    # xyxy 是一个二维 Tensor，每一行是一个目标框
    xyxy = boxes.xyxy
    print('======================xyxy======================')
    print(xyxy)
    n = xyxy.shape[0]

    # 每种类别目标的计数
    count = {}

    # 创建文件夹，用以保存检测结果
    Path('results').mkdir(exist_ok=True)

    # 处理检测到的每一个目标框
    for i in range(n):
        # 取第i个边界框，并转为int格式数据
        box = xyxy[i].int()

        # 取第i个边界框的类别名称
        label = model.names[boxes.cls[i].item()]

        # 类别计数
        if label not in count:
            count[label] = 0
        count[label] += 1

        print(box)
        print(label)
        print()

        # 用切片的方式取出边界框区域的图像
        obj_img = img_array[box[1].item():box[3].item(), box[0].item():box[2].item(), :]

        # 创建图像实例，并保存
        im = Image.fromarray(obj_img)
        im.save(f'results/{label}_{count[label]}.jpg')

    # 将检测结果渲染出来，注意这里输出的通道顺序为 BGR
    im_array = result.plot()  # plot a BGR numpy array of predictions

    # 创建图像实例，需要调整通道顺序
    im = Image.fromarray(im_array[:, :, ::-1])  # RGB PIL image
    im.save('results/result.jpg')  # save image

if __name__ == '__main__':
    main()
