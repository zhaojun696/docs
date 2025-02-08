

# 协方差（covariance）和皮尔逊相关系数（Pearson correlation coefficient）都是用于衡量两个变量之间线性关系的指标。

# 协方差衡量两个变量之间的总体方向，即是否存在正向关系或负向关系。它的值越大，代表两个变量的相关性越强。
# 但是，协方差的值不具有标准化的性质，所以无法直接比较不同变量之间的相关性大小。

# 皮尔逊相关系数则是协方差的标准化处理，它衡量的是两个变量之间线性相关程度的强弱，取值范围在-1到1之间。
# 当相关系数为正值时，表示两个变量呈正向相关；当相关系数为负值时，表示两个变量呈负向相关；当相关系数为0时，表示两个变量之间不存在线性关系。

# 斯皮尔曼相关系数（Spearman correlation coefficient）与皮尔逊相关系数的计算方式不同，它将原始数据转化为等级或者排序数据，然后计算排名之间的相关系数。
# 斯皮尔曼相关系数主要应用于样本数据的总体分布非正态或存在极端值（outliers）的情况下。

# 因此，协方差和相关系数都是用于衡量两个变量之间的关系强度，但是协方差不能直接用于比较不同变量之间的相关性，而相关系数则可以标准化处理后进行比较。
# 斯皮尔曼相关系数与皮尔逊相关系数相比，适用于更多的数据类型，但是计算量较大。



# 协方差与自身的协方差为该变量的方差。与其他的协方差为相关性

#%%
import pandas as pd


d=pd.DataFrame({
    'a':[3,1,3,4,2,0,1],
    'b':[9,11,33,4,3,1,1],
    'aa':[3,3,2,1,0,4,1],
    'bb':[9,33,3,11,1,4,1]
})
d.a.corr(d.b,method='spearman'),d.aa.corr(d.bb,method='spearman')

#%%


d=pd.DataFrame({
    'a':[2,2,1,1,6,6],
    'b':[-1,-1,3,4,1,1]
})
d1=pd.DataFrame({
    'a':[2,2,1,1,6,6],
    'b':[-1,-1,3,4,1,1]
})

q=pd.qcut(d.a,3,range(3))
q1=pd.qcut(d1.a,2,range(2))


c=d.groupby(q).b.mean()

c1=d.groupby(q1).b.mean()
c


#%%
# 先对df排序
_d=d.sort_values('b')
# groupby后会按顺序取前n个
_d.groupby('a').head(1)


#%%

# 获取相关性最高的n个特征
import pandas as pd

# 创建示例数据
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [1, 1, 6, 8, 10],
    'C': [3, 6, 9, 12, 15],
    'D': [4, 8, 12, 16, 20],
    'E': [5, 10, 15, 20, 25],
    'target': [33, 12, 18, 24, 30]
}

df = pd.DataFrame(data)



import numpy as np

# 假设df是你的DataFrame
# 计算相关矩阵
corr_matrix = df.corr()

# 将对角线元素设置为0，因为我们不考虑特征与自身的相关度
np.fill_diagonal(corr_matrix.values, 0)

# 特征列表
features = []
n = 3  # 你想要的特征数量
while len(features) < n:
    # 找到相关度最高的特征对
    max_corr = 0
    feature_pair = None
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if corr_matrix.iloc[i, j] > max_corr and corr_matrix.columns[j] not in features and corr_matrix.index[i] not in features:
                max_corr = corr_matrix.iloc[i, j]
                feature_pair = (corr_matrix.index[i], corr_matrix.columns[j])
    
    # 如果找到了特征对，添加到列表中
    if feature_pair:
        features.append(feature_pair[0])
        if len(features) < n:
            features.append(feature_pair[1])
    else:
        # 如果没有找到特征对，说明无法继续添加特征
        break

print(features)
