import utils


# 自己构建的数据集路径
img_dir = './my_data'

def main():
    # 把指定路径下的图片全部读入，转为 28x28 的灰度图，返回 ndarray
    img_data = utils.read_img_from_dir(img_dir, img_size=(28, 28), gray=True)
    print(img_data.shape)
    print(img_data.dtype)
    print(img_data[0])
    utils.draw_imgs(img_data)
    
    # 归一化
    img_data = img_data / 255.0 
    
    # 反转灰度（如果图片是白底黑字，使用反转灰度）
    img_data = 1 - img_data
    utils.draw_imgs(img_data)
    
    # 加载模型
    # model = YourModelClass()
    # model.load_state_dict(torch.load('your_model_name.pt'))
    
    # 预测
    pass
    
    # 用 tensorboard 记录预测结果
    # 假设 y_pred 中每一行是预测的标签（数字0～9）；data 是对应的图片，形状是 (n, c, h, w)
    for i in range(10):
        # mask 是一个布尔向量，表示 y_pred 的值等于 i 的位置，即预测为数字 i 的位置
        mask = (y_pred.view(-1) == i)
        # 仅当存在预测为数字 i 的图片时才记录
        if mask.sum() > 0:
            # 把预测为数字 i 的图片记录到 tensorboard
            writer.add_images(f'num={i}', data[mask])


if __name__ == '__main__':
    main()
