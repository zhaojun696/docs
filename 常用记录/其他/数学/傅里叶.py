#%%
# 傅里叶变换是时域到频域转换的一种常用方法，它将时域上的信号转换成一个连续的复数函数，表示信号在各个频率上的成分

# 实现了对一个由10Hz和20Hz正弦波构成的信号进行傅里叶变换
# 并绘制了其频谱图。使用numpy.fft库的fft函数可以对信号进行变换
# 而np.abs则可以得到其幅值，即频谱图
import numpy as np

t = np.linspace(-1, 1, 200)
y = np.sin(2 * np.pi * 10 * t) + 2 * np.sin(2 * np.pi * 20 * t)
y_fft = np.fft.fft(y)

# 绘制频谱图
import matplotlib.pyplot as plt

plt.plot(np.abs(y_fft))
plt.show()
plt.plot(y)

#%%
import numpy as np
import matplotlib.pyplot as plt

# 时域信号
t = np.linspace(0, 1, 100)  # 时间轴
x = np.sin(2 * np.pi * 5 * t)  # 正弦波信号，频率为5Hz
X = np.fft.fft(x)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 时域图
plt.plot(t, x)
plt.title("Time Domain")
plt.xlabel("Time")
plt.ylabel("Amplitude")


# 可以看到一个尖峰在5Hz处，这表明我们的原始正弦波信号主要包含5Hz的频率成分。其他较低的频率分量由于正弦波的周期性而被抵消了。
plt.subplot(1, 2, 2)  # 频域图
plt.plot(X) 
plt.title("Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()