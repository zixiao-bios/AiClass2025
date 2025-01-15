import numpy as np


# ================= 创建 =================
# 从列表创建
arr_from_list = np.array([1, 2, 3, 4])
print("从列表创建:\n", arr_from_list)

# 创建全零数组
zeros_array = np.zeros((2, 3))
print("全零数组:\n", zeros_array)

# 创建全一数组
ones_array = np.ones((3, 2))
print("全一数组:\n", ones_array)

# 创建随机数组
random_array = np.random.rand(2, 2)
print("随机数组:\n", random_array)


# ================= 基本属性 =================
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("数组形状:", arr.shape)
print("数据类型:", arr.dtype)
print("数组大小:", arr.size)
print("数组维度:", arr.ndim)


# ================= 索引 =================
arr = np.array([10, 20, 30, 40, 50])

# 获取单个元素
print("获取第二个元素:", arr[1])

# 获取子数组
print("获取第2到第4个元素:", arr[1:4])

# 多维索引
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("获取第二行第三列的元素:", arr_2d[1, 2])
print("获取第一列的所有元素:", arr_2d[:, 0])


# ================= 基本运算 =================
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 加法运算
print("加法:", arr1 + arr2)  # 输出 [5 7 9]

# 乘法运算
print("乘法:", arr1 * arr2)  # 输出 [4 10 18]

# 标量运算
print("乘以标量:", arr1 * 2)  # 输出 [2 4 6]

# 广播机制
arr3 = np.array([[1, 2, 3], [4, 5, 6]])
print("广播加法:\n", arr3 + np.array([1, 2, 3]))  # 每行分别加上 [1, 2, 3]


# ================= 改变形状 =================
arr = np.array([[1, 2, 3], [4, 5, 6]])

# 重塑数组
reshaped_arr = arr.reshape(3, 2)
print("重塑后的数组:\n", reshaped_arr)

print("转为列向量:\n", arr.reshape(-1, 1))

# 展平数组
flattened_arr = arr.flatten()
print("展平后的数组:", flattened_arr)


# ================= 统计 =================
arr = np.array([[1, 2, 3], [4, 5, 6]])

print("总和:", np.sum(arr))
print("按列求和:", np.sum(arr, axis=0))
print("均值:", np.mean(arr))
print("标准差:", np.std(arr))
print("最大值:", np.max(arr))
print("最小值:", np.min(arr))


# ================= 与 list 对比 =================
import time

# 使用NumPy计算
arr_np = np.arange(int(1e8))
start = time.time()
arr_np_sum = np.sum(arr_np)
print("NumPy求和时间:", time.time() - start)

# 使用Python列表计算
arr_list = list(range(int(1e8)))
start = time.time()
arr_list_sum = sum(arr_list)
print("Python列表求和时间:", time.time() - start)
