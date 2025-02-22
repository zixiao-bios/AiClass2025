#! /usr/bin/env python
from ultralytics import YOLO
import cv2
import numpy as np
import copy
import threading

import rospy
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge


# 由于图片获取速度与目标检测速度不一致，采取异步方法
# 最新获取到的RGB图像，以及该变量的互斥锁
last_img = None
img_mutex = threading.Lock()

# 最新获取到的深度图，以及该变量的互斥锁
last_depth = None
depth_mutex = threading.Lock()

# YOLO模型
model = None

# ROS图像格式与ndarray的转换器
bridge = CvBridge()


def get_K(fov, img_w, img_h):
    """计算内参矩阵K

    Args:
        fov (number): 水平FOV
        img_w (number): 图像宽度
        img_h (number): 图像高度

    Returns:
        ndarray: K矩阵
    """
    # 计算焦距
    f_x = img_w / (2 * np.tan(np.deg2rad(fov) / 2))
    f_y = f_x * img_h / img_w  # 像素为正方形

    # 计算光心
    c_x = img_w / 2
    c_y = img_h / 2

    # 构造内参矩阵K
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])
    return K


def get_rgb_image(data, img_size):
    """订阅的RGB图像 Topic的回调函数

    Args:
        data: 接收到的Topic数据
        img_size (tuple): 图像的期望分辨率，用于数据校验
    """

    # 请求互斥锁
    img_mutex.acquire()

    # 将接收的图像消息转为ndarray，格式为'bgr8'，表示通道顺序为BGR，每个通道8位数据（即0-255）
    global last_img
    last_img = bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

    # 检查图像是否满足分辨率要求
    assert last_img.shape[0] == img_size[1] and last_img.shape[1] == img_size[0], f'Get rgb_img shape:{last_img.shape} not match target img_size:{img_size}!'

    # 释放互斥锁
    img_mutex.release()


def get_depth_image(data, img_size):
    """订阅的Depth图像 Topic的回调函数

    Args:
        data: 接收到的Topic数据
        img_size (tuple): 图像的期望分辨率，用于数据校验
    """
    depth_mutex.acquire()
    global last_depth
    last_depth = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    assert last_depth.shape[0] == img_size[1] and last_depth.shape[1] == img_size[0], f'Get depth_img shape:{last_depth.shape} not match target img_size:{img_size}!'
    depth_mutex.release()


def detect(img):
    """进行目标检测

    Args:
        img (ndarray): 待检测的图片，形状为(H, W, C)，通道顺序为BGR

    Returns:
        tuple: 检测结果，前三个分别为n个边界框的顶点、类别id、置信度，最后一个为渲染图
    """

    # 用YOLO模型做一次预测，取第一个结果（由于输入图片只有一张，因此返回值为长度为1的List）
    global model
    result = model.predict(img)[0]

    # 提取预测结果并返回
    return result.boxes.xyxy, result.boxes.cls, result.plot()


def get_depthes(depth_img, xyxys):
    """结合深度图，计算每个边界框的深度

    Args:
        depth_img (ndarray): 深度图，形状为(H, W)，只有1个通道，数值代表该像素的深度（米）
        xyxys (ndarray): n个边界框的顶点坐标（左上和右下顶点）

    Returns:
        List: n个边界框对应的深度值
    """

    # 存放深度值的列表
    depthes = []

    # xyxys形状为(n, 4)
    n = xyxys.shape[0]

    for i in range(n):
        # xyxy形状为(4)
        # 将顶点坐标转为int型，才能用于索引
        xyxy = xyxys[i].int()

        # 将边界框围住的区域通过切片提取出来
        obj_depth = depth_img[xyxy[1].item():xyxy[3].item(), xyxy[0].item():xyxy[2].item()]

        # 计算中位数，作为边界框的深度，由于深度图中包含nan数据，要使用nanmedian函数
        depth = np.nanmedian(obj_depth)

        depthes.append(depth)

    return depthes


def add_distance_text(img, xyxys, depthes):
    """将每个边界框的深度值画到渲染图上

    Args:
        img (ndarray): 渲染图
        xyxys (ndarray): n个边界框的顶点
        depthes (List): n个边界框的深度值

    Returns:
        ndarray: 修改后的渲染图
    """
    n = xyxys.shape[0]
    for i in range(n):
        xyxy = xyxys[i].int()

        # 向图片中添加文字，参数依次为：图片、文字内容、文字位置、字体、字号、颜色、粗细
        cv2.putText(img, f'{depthes[i]:.2f}', (xyxy[0].item(), xyxy[1].item() - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img


def get_positions(xyxys, depthes, K):
    """结合相机内参，与目标框的像素坐标、深度，得到目标在相机坐标系中的位置

    Args:
        xyxys (ndarray): n个边界框的顶点
        depthes (List): n个边界框的深度
        K (ndarray): 内参矩阵

    Returns:
        ndarray: n个目标的相机坐标系下的三维坐标
    """
    n = xyxys.shape[0]

    # 保存n个三维坐标的ndarray
    positions = np.zeros((n, 3))

    for i in range(n):
        xyxy = xyxys[i]

        # 像素坐标
        cx = 0.5 * (xyxy[0] + xyxy[2])
        cy = 0.5 * (xyxy[1] + xyxy[3])

        # 像素的齐次坐标矩阵
        cp = np.array([[cx.cpu()], 
                       [cy.cpu()], 
                       [1]])
        
        # 按照公式，计算相机坐标系下的坐标
        pos = depthes[i] * np.linalg.inv(K) @ cp

        # 将当前目标的三维坐标，添加到positions中，flatten()把矩阵展平为一维向量
        positions[i, :] = pos.flatten()

    return positions


def pub_obj_points(positions, pub, time):
    """发布检测到的目标点的Topic数据

    Args:
        positions (ndarray): n个目标的相机坐标系下的坐标
        pub (Publisher): Topic的发布者
        time (Time): 此次检测对应的时间戳
    """
    n = positions.shape[0]

    # 发布的消息
    msg = PointStamped()

    for i in range(n):
        pos = positions[i]

        # 坐标赋值
        msg.point.x = pos[0]
        msg.point.y = pos[1]
        msg.point.z = pos[2]

        # 坐标所在的坐标系（相机坐标系）
        msg.header.frame_id = 'kinect_frame_optical'

        # 坐标对应的时间点
        msg.header.stamp = time

        pub.publish(msg)



def main():
    rospy.init_node('detection')

    # 从参数服务器获取参数
    yolo_weight_path = rospy.get_param('~yolo_weight_path')
    fov = rospy.get_param('~fov')
    img_w = rospy.get_param('~img_w')
    img_h = rospy.get_param('~img_h')

    # 订阅kinect相机的RGB图像、深度图
    rospy.Subscriber('kinect/rgb/image_raw', RosImage, get_rgb_image, (img_w, img_h))
    rospy.Subscriber('kinect/depth/image_raw', RosImage, get_depth_image, (img_w, img_h))

    # 发布渲染效果图、目标点的Publisher
    img_pub = rospy.Publisher('detection/image', RosImage)
    point_pub = rospy.Publisher('detection/objs', PointStamped)

    # 实例化YOLO模型
    global model
    model = YOLO(yolo_weight_path)

    # 计算内参矩阵K
    K = get_K(fov, img_w, img_h)

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        # 请求互斥锁
        img_mutex.acquire()
        depth_mutex.acquire()

        # 如果还没有接收到图片，则跳过
        if last_img is None or last_depth is None:
            # 不要忘记释放互斥锁，否则会导致死锁
            img_mutex.release()
            depth_mutex.release()
            continue
        
        # 将获取的图像深拷贝过来，防止处理时的竞争
        input_img = copy.deepcopy(last_img)
        input_depth = copy.deepcopy(last_depth)

        # 释放互斥锁
        img_mutex.release()
        depth_mutex.release()

        # 获取当前时间，作为此次检测的时间
        time = rospy.Time.now()

        # 1. 目标检测
        xyxys, labels, show_img = detect(input_img)

        # 2. 用深度图和边界框，计算目标距离
        depthes = get_depthes(input_depth, xyxys)

        # 3. 把目标距离添加到渲染图上
        show_img = add_distance_text(show_img, xyxys, depthes)

        # 4. 通过目标的像素坐标、深度、内参，计算相机坐标系下的坐标
        positions = get_positions(xyxys, depthes, K)

        # 发布相机坐标系下的目标点
        pub_obj_points(positions, point_pub, time)

        # 将ndarray格式的渲染图转化为ros消息格式
        msg = bridge.cv2_to_imgmsg(show_img, encoding="bgr8")
        img_pub.publish(msg)

        rate.sleep()


if __name__ == '__main__':
    main()
