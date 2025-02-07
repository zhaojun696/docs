#%%
# kde交点
import numpy as np
from scipy import stats
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# 生成两个随机样本
sample1 = np.random.normal(loc=0, scale=1, size=1000)
sample2 = np.random.normal(loc=2, scale=1, size=1000)

# 计算两个样本的kde分布
kde1 = stats.gaussian_kde(sample1)
kde2 = stats.gaussian_kde(sample2)

# 画出两个kde分布
x = np.linspace(-5, 8, 200)
plt.plot(x, kde1(x))
plt.plot(x, kde2(x))

# 找到两个kde交点处的x值
root_func = lambda x: kde1(x) - kde2(x)
root1 = brentq(root_func, -5, 8)
# 在图中添加交点标记
plt.plot([root1], [kde1(root1)], 'ro')

# 显示图形
plt.show()
root1


#%%
# 所有交点
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 创建两个正态分布的样本，用于计算KDE
np.random.seed(0)
x1 = np.random.normal(loc=-1, scale=1, size=100)
x2 = np.random.normal(loc=1, scale=1, size=100)

# 计算KDE的密度函数
kde1 = gaussian_kde(x1)
kde2 = gaussian_kde(x2)

# 生成密度函数对应的x和y值
x = np.linspace(-5, 5, num=200)
y1 = kde1(x)
y2 = kde2(x)

# 找到KDE密度函数相交的点
diff = y1 - y2
cross_points = []
for i in range(len(diff) - 1):
    if diff[i] * diff[i+1] < 0:
        cross_points.append((x[i] + x[i+1])/2)
        
# 绘制KDE密度函数
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y1, label='Sample 1')
ax.plot(x, y2, label='Sample 2')
ax.fill_between(x, y1, y2, where=y1>y2, interpolate=True, alpha=0.2)
ax.fill_between(x, y1, y2, where=y1<y2, interpolate=True, alpha=0.92)
ax.legend()

# 在图像中标注KDE相交点
for point in cross_points:
    ax.axvline(x=point, linestyle='--', color='gray')
    
plt.show()