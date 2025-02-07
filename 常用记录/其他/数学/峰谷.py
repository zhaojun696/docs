#%%
# 简单曲线
import matplotlib.pyplot as plt
def peak_and_valley(data):
    peaks = []
    valleys = []
    for i in range(1, len(data)-1):
        if data[i]>=data[i-1] and data[i]>=data[i+1]:
            peaks.append(i)
        elif data[i]<=data[i-1] and data[i]<=data[i+1]:
            valleys.append(i)
    return peaks, valleys



data = [10, 15, 20, 18, 13, 8, 4, 6, 9, 12, 16, 21, 19, 14, 9, 5, 3]

peaks, valleys = peak_and_valley(data)

plt.plot(data)
plt.plot(peaks, [data[i] for i in peaks], 'ro')
plt.plot(valleys, [data[i] for i in valleys], 'go')
plt.show()

#%%
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# 生成一个随机的复杂曲线
x = np.linspace(0, 4*np.pi, 1000)
y = np.sin(x) + 0.5*np.sin(10*x) + np.cos(5*x) + np.random.normal(0, 0.1, x.shape)





# x：一维曲线的X轴或数组。如果没有指定，则默认为arange(N)，其中N是Y的长度。
# height：要找到的峰或谷的最小高度。默认值为None，允许找到任何高度的峰或谷。
# threshold：峰或谷的最小距离（以X轴为单位）。默认值为None，允许相邻峰或谷之间没有距离门限。
# distance：相邻峰或谷之间的最小距离（以X轴为单位）。默认为1，意味着峰或谷之间必须相隔至少一个数据点。
# prominence：峰或谷的最小腰宽。默认情况下，不应用任何宽度限制，并且没有最小宽度要求。
# width：峰或谷的最小宽度。默认情况下，不应用任何宽度限制，并且没有最小宽度要求。
# wlen：确定峰或谷的局部宽度的左右宽度（以X轴单位计算）。默认值为None，此时局部宽度采用默认值，即sqrt(distance)。
# rel_height：峰或谷高度的相对值。默认值为0.5，意思是要从整个数据集中找到50％高度以上的峰值或谷值。
# plateau_size：控制要考虑的平原大小。默认值为None，因此不考虑任何平原。

# 在使用find_peaks函数时，可以根据需要设定不同的参数，以查找特定类型的极值。
# 例如，如果想要查找复杂信号中较高的波峰和波谷，可以设置height参数，
# 因为较低的峰谷通常与噪声相关联。
# 如果想要查找特定距离范围内的峰值，则可以设置distance参数；
# 如果只想保留高度较高的峰值，则可以设置prominance参数。

# 使用find_peaks函数查找波峰和波谷
distance=10
rel_height=0.9
threshold=0.1
peaks, _ = find_peaks(y,rel_height=rel_height,distance=distance,threshold=threshold)
valleys, _ = find_peaks(y * -1,rel_height=rel_height,distance=distance,threshold=threshold)

# 绘制原始曲线
plt.plot(x, y)

# 将波峰和波谷标记为红色圆点
plt.plot(x[peaks], y[peaks], 'ro')
plt.plot(x[valleys], y[valleys], 'ro')

plt.show()