#%%


import numpy as np
# 推荐系统的工作流程：给定一个用户，系统会为此用户返回N个最好的推荐菜。
# （1）：寻找用户没有评级打分的菜肴，即在用户-物品矩阵中的0值。
# （2）：在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。这就是说我们认为用户可能对物品的打分（这就是相似度计算的初衷）
# （3）：对这些物品的评分从高到底进行排序，返回前N 个商品。

from numpy import linalg as la
# 余弦距离得到的值为-1到1，我们将其映射到0到1
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

"""
函数说明：加载数据,菜肴矩阵
行：代表人
列：代表菜肴名词
值：代表人对菜肴的评分，0代表未评分
"""
def loadExData():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


"""
Parameters:
    dataMat - 训练数据集
    user - 用户编号
    simMeas - 相似度计算方法
    item - 未评分的物品编号
    
Returns:
    评分（0~5之间的值）
"""
def svdEst(dataMat, user, simMeas, item):
    #假设user=1
    # 得到数据集中的物品数目
    n = np.shape(dataMat)[1]
    # 初始化两个评分值
    simTotal = 0.0
    ratSimTotal = 0.0
    # 奇异值分解
    # 在SVD分解之后，我们只利用包含90%能量值的奇异值，这些奇异值会以Numpy数组形式得以保存
    U, Sigma, VT = la.svd(dataMat)
    # 如果要进行矩阵运算，就必须要用这些奇异值构造出一个对角阵
    Sig4 = np.mat(np.eye(4) * Sigma[: 4])
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品（物品的4个主要特征）
    xformedItems = dataMat.T * U[:, :4] * Sig4.I#shape(11,4)
    # 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue
        # 相似度的计算也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('商品 %d 和商品 %d 相似度similarity is: %f' % (item, j, similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity 用户相似度   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 通过除以所有的评分和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal / simTotal

dataMat=np.mat(loadExData())
#假设当前用户时1
print('查看用户1未评级商品：',np.nonzero(dataMat[1, :].A == 0)[1])

# 得到数据集中的物品数目
n = np.shape(dataMat)[1]
# 初始化两个评分值
simTotal = 0.0
ratSimTotal = 0.0
# 奇异值分解
# 在SVD分解之后，我们只利用包含90%能量值的奇异值，这些奇异值会以Numpy数组形式得以保存
U, Sigma, VT = la.svd(dataMat)
# 如果要进行矩阵运算，就必须要用这些奇异值构造出一个对角阵
Sig4 = np.mat(np.eye(4) * Sigma[: 4])
# 利用U矩阵将物品转换到低维空间中，构建转换后的物品（物品的4个主要特征）
xformedItems = dataMat.T * U[:, :4] * Sig4.I
print('降维重构后的数据:',xformedItems,xformedItems.shape,dataMat.shape,n)

#%%
# 遍历行中的每个物品（对用户评过分的物品进行遍历，并将它与其他物品进行比较）
#假设未评分商品为第0个，开始计算
for j in range(n):
    userRating = dataMat[1, j]
    # 如果某个物品的评分值为0，则跳过这个物品
    if userRating == 0:
        continue
    # 相似度的计算也会作为一个参数传递给该函数
    similarity = cosSim(xformedItems[0, :].T, xformedItems[j, :].T)
    print('商品 %d 和商品 %d 相似度is: %f' % (0, j, similarity))
    # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
    # similarity 用户相似度   userRating 用户评分
    simTotal += similarity
    ratSimTotal += similarity * userRating
if simTotal == 0:
    print('商品0得分：',0)
    # 通过除以所有的评分和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
else:
    print( '商品0得分：',ratSimTotal / simTotal)

#%%
"""
函数说明：推荐引擎

Parameters:
    dataMat - 训练数据集
    user - 用户编号
    N- 产生N个推荐结果
    simMeas - 相似度计算方法
    estMethod - 推荐引擎方法

Returns:
    评分（0~5之间的值）
"""


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=svdEst):
    # 寻找未评级的物品
    # 对给定的用户建立一个未评分的物品列表
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return ('本店所有商品你均尝试过，打过分')
    # 物品的编号和评分值
    itemScores = []
    # 在未评分的物品上进行循环
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 寻找前N个未评级的物品，调用svdEst()来产生该物品的预测得分，该物品的编号和估计值会放在一个元素列表itemScores中
        itemScores.append((item, estimatedScore))
    # 返回元素列表，第一个就是最大值
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]

myMat = np.mat(loadExData())
recommend(myMat, 1, estMethod=svdEst)#对用户1进行推荐
print('\n----------------------------------------\n')
A = recommend(myMat, 1, estMethod=svdEst, simMeas=cosSim)
A=np.array(A)
for i in range(A.shape[0]):
    print('给您推荐的商品是{}，计算得分是{}'.format(A[i][0],A[i][1]))

