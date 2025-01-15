import torch
import numpy as np


# ================ 创建 ================
# 手动创建
tensor_manual = torch.tensor([[1, 2], [3, 4]])
print("手动创建tensor:\n", tensor_manual)

# 全零、全一和随机tensor
tensor_zeros = torch.zeros((2, 3))
tensor_ones = torch.ones((2, 3))
tensor_rand = torch.rand((2, 3))
print("全零tensor:\n", tensor_zeros)
print("全一tensor:\n", tensor_ones)
print("随机tensor:\n", tensor_rand)


# ================ 基础属性 ================
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

print("形状:", tensor.shape)
print("数据类型:", tensor.dtype)
print("设备:", tensor.device)


# ================ 运算 ================
# 基本加法运算
tensor1 = torch.tensor([1.0, 2.0, 3.0])
tensor2 = torch.tensor([4.0, 5.0, 6.0])
print("加法:", tensor1 + tensor2)

# 广播机制
tensor3 = torch.tensor([[1, 2], [3, 4]])
tensor_broadcast = tensor3 + torch.tensor([1, 2])
print("广播加法:\n", tensor_broadcast)

# 将tensor移动到GPU
if torch.cuda.is_available():
    tensor_gpu = tensor1.to("cuda")
    print("在GPU上的tensor:", tensor_gpu)


# ================ 改变形状 ================
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 重塑
reshaped_tensor = tensor.reshape(3, 2)
print("重塑后的tensor:\n", reshaped_tensor)

# 转置
transposed_tensor = tensor.transpose(0, 1)
print("转置后的tensor:\n", transposed_tensor)


# ================ 自动微分 ================
# 创建需要梯度的tensor
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x * x  # 操作
y_sum = y.sum()  # 求和
y_sum.backward()  # 计算梯度，只能对标量调用backward()
print("x的梯度:", x.grad)  # 输出 tensor([4., 6.])


# ================ 与 ndarray 的转换 ================
# 从ndarray转换为tensor
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)
print("从ndarray转换为tensor:", tensor_from_numpy)

# 从tensor转换为ndarray
tensor = torch.tensor([4.0, 5.0, 6.0])
numpy_from_tensor = tensor.numpy()
print("从tensor转换为ndarray:", numpy_from_tensor)
