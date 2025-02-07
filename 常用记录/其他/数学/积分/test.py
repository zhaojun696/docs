# 积分（integration）是微积分中的一种基本运算，
# 是求解一个函数在一定区间上面的面积，其本质是将一个连续函数在给定区间上分成无数个微小的矩形，
# 然后将这些微小的矩形的面积相加，从而得到整个区间上的面积大小。
# 积分在数学中有广泛的应用，包括在物理学、工程学、经济学、统计学等领域中的应用。



# 积分可以被视为是函数在给定区域上的面积，
# 但是并不是所有的积分都是代表面积。
# 例如，在本例中，因为函数 $f(x, y) = x \cdot y$ 在积分域上是对称的，积分结果为0。
# 这意味着函数在积分域上所产生的正面积与负面积的总和为0，
# 即正面积的面积等于负面积的面积。
# 因此，虽然图像上的积分区域看起来不为0，但在函数上的积分值确实为0。
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
# nquad 函数用于多维积分，它可以对一个 N 维函数在 N 维矩形域上进行积分。
# 它需要指定被积函数和每一维的积分上下限。它的返回值是积分的数值近似和估计的误差。


def f(x, y):
    return x * y

a1, b1, a2, b2 = -1, 1, -1, 1

# 绘制函数图像
x = np.linspace(a1, b1, 100)
y = np.linspace(a2, b2, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z, cmap="plasma")
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(X, Y, Z, cmap='viridis')
ax.set_title('Integration Area')
plt.show()

# 进行多重积分计算
result, error = integrate.nquad(f, [(a1, b1), (a2, b2)])
print(f"The result is {result} with error {error}.")

def fs(x, y):
    return abs(x * y)
# 进行多重积分面积计算
result, error = integrate.nquad(fs, [(a1, b1), (a2, b2)])
print(f"The area is {result} with error {error}.")

#%%
# quad 函数用于一维积分，它需要指定被积函数、积分下限和积分上限。
# 它的返回值是积分的数值近似和估计的误差。如果需要进行多个一维积分，可以多次调用 quad。
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

# 定义被积函数
def f(x):
    return np.sin(x)

# 定义积分区间
a, b = 0, np.pi

# 计算积分
result, error = integrate.quad(f, a, b)

# 绘制函数图像
x = np.linspace(a, b, num=100)
y = f(x)

plt.plot(x, y, label='f(x)')
plt.fill_between(x, y, 0, alpha=0.1)
plt.legend()
plt.show()

print(f"积分结果：{result:.6f}")

#%%
# 计算出错
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def f(x):
    return np.sin(np.pi*x)/(np.pi*x)

a, b = -10, 10

# 绘制函数图像
x = np.linspace(a, b, 500)
y = f(x)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.axhline(y=0, color='black', lw=0.5)
ax.axvline(x=0, color='black', lw=0.5)
ax.set_title('Function Plot')
plt.show()

# 绘制积分区域
fig, ax = plt.subplots()
ax.fill_between(x, y, where=((x>=a) & (x<=b)), color='blue', alpha=0.2)
ax.axhline(y=0, color='black', lw=0.5)
ax.axvline(x=0, color='black', lw=0.5)
ax.set_xlim(a, b)
ax.set_ylim(np.min(y), np.max(y))
ax.set_title('Integration Area')
plt.show()

# 进行积分计算
result, error = integrate.quad(f, a, b)
print(f"The result is {result} with error {error}.")


#%%
# 交集面积

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 生成两个数据集
np.random.seed(1)
f1 = np.random.normal(loc=0, scale=1, size=1000)
f2 = np.random.normal(loc=1, scale=1, size=1000)

# 计算两个数据集的KDE密度函数
kde1 = stats.gaussian_kde(f1)
kde2 = stats.gaussian_kde(f2)

# 求交集区间
min_val = np.max([f1.min(), f2.min()])
max_val = np.min([f1.max(), f2.max()])
x = np.linspace(min_val, max_val, num=500)
intersect = np.min([kde1(x), kde2(x)], axis=0)

# 计算交集面积
intersect_area = np.trapz(intersect, x)

# 绘制两个函数的KDE密度函数图像及交集
fig, ax = plt.subplots()
ax.plot(x, kde1(x), label='f1')
ax.plot(x, kde2(x), label='f2')
ax.fill_between(x, intersect, color='red', alpha=0.5, label='intersect')
ax.legend()
plt.show()

print(f"The intersection area is {intersect_area:.3f}.")

#%%
# 并集
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, stats

# 生成两个正态分布随机数作为两个数据集
np.random.seed(123)
f1 = np.random.normal(loc=0, scale=1, size=1000)
f2 = np.random.normal(loc=2, scale=1, size=1000)

# 计算两个数据集的KDE密度函数
kde1 = stats.gaussian_kde(f1)
kde2 = stats.gaussian_kde(f2)

# 求并集区间
min_val = np.min([f1.min(), f2.min()])
max_val = np.max([f1.max(), f2.max()])
x = np.linspace(min_val, max_val, 500)
union = np.max([kde1(x), kde2(x)], axis=0)

# 将积分区间平均分为500小段，对每一小段进行求面积
area = integrate.simps(union, x=x)

# 绘制函数图像和并集区间
fig, ax = plt.subplots()
ax.plot(x, kde1(x), label='Distribution 1')
ax.plot(x, kde2(x), label='Distribution 2')
ax.fill_between(x, union, color='blue', alpha=0.2, label='Union Area')
ax.axhline(y=0, color='black', lw=0.5)
ax.axvline(x=0, color='black', lw=0.5)
ax.set_title('Function Plot')
ax.legend()
plt.show()

print(f"The union area is {area:.4f}.")


