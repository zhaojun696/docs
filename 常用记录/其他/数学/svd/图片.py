#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

img = mpimg.imread('../../../统计知识/抽样方法.jpg')
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))

U, s, Vh = np.linalg.svd(gray)

def composite(U, s, Vh, n):
    print(U.shape,s.shape,Vh.shape)
    print(U[:, :n].shape,s[:n].shape,Vh[:n,:].shape)
    return np.dot(U[:, :n], np.dot(np.diag(s[:n]), Vh[:n,:]))

for i in [10, 20, 50]:
    new_img = composite(U, s, Vh, i)
    print(new_img.shape)
    plt.imshow(new_img, cmap='gray')
    title = "new_img_%s" % i
    plt.title(title)
    plt.show()
