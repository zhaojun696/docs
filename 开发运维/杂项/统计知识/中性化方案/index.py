
# s是因子值，s1是市值、等连续性因子（不能为分类因子，分类数字化后也不行）


#%%
import numpy as np
import statsmodels.api as sm
import pandas as pd

# 创建示例数据集
np.random.seed(0)
X = np.random.rand(100, 1)  # 一个自变量
y = 2 * X + 1 + np.random.rand(100, 1)  # 线性关系加上噪声

# 将数据转换为 pandas DataFrame
data = pd.DataFrame({'XX': X.flatten(), 'y': y.flatten()})

# 添加常数项
X_with_constant = sm.add_constant(data.XX)

# 拟合线性回归模型
model = sm.OLS(y, X_with_constant)
res=model.fit()

# 打印模型摘要
res.params,res.params['XX']

#%%
import statsmodels.api as sm
import pandas as pd
s=pd.Series([1,2,3,4])
s1=s*3
s1=sm.add_constant(s1)
model = sm.OLS(s,s1)
res = model.fit()
resid = res.resid
resid,res.params

#%%
import statsmodels.api as sm
import pandas as pd
s=pd.Series([1,2,3,4])
s1=s*10
model = sm.OLS(s,s1)
res = model.fit()
resid = res.resid
resid,res.params



# 回归法本质就是去均值取残差。
# 详细一点，行业中性化后的值是原始因子相对本行业因子均值的距离（行业需要哑变量化）
# （市值中性化后的值是原始因子相对该市值下所有因子的均值的距离）。
#%%
import statsmodels.api as sm
import pandas as pd
s=pd.Series([1,2,3,4])
s1=pd.Series([11,12,13,14])
model = sm.OLS(s,s1)
res = model.fit()
resid = res.resid
resid


#%%
import statsmodels.api as sm
import pandas as pd
s=pd.Series([1,2,3,4])
s1=pd.Series([11,12,13,14])*100
model = sm.OLS(s,s1)
res = model.fit()
resid = res.resid
resid


#%%
# 回归法---->组内减均值；分组标准化 ---->组内减均值处以标准差

#%%
import statsmodels.api as sm
import pandas as pd
y = list(range(40)) + list(range(0, 120, 2))
g = [0] * 40 + [1] * 30+ [2] * 30 # 分为3组数
c = [1]*100
df = pd.DataFrame({'y': y, 'g': g,'c':c})
dummies=pd.get_dummies(df.g)
df=pd.concat([df,dummies],axis=1)
df.head() # 这个数据有100个，g 代表组别，c 为常数项，y 代表因子值，第一个组别为0 的因子分别是0-39，第二组是0-118（间隔为2）
#%%
# 组内减均值,不除以标准差
df['y_standard_sub_mean'] = df.groupby('g')['y'].apply(lambda x:(x -x.mean())) 
# 组内减均值，并除以标准差
df['y_standard_sub_mean_div_std'] = df.groupby('g')['y'].apply(lambda x:(x -x.mean())/x.std())
# 回归方法计算
model = sm.OLS(df['y'],df[[0,1,2,'c']])
res = model.fit()
resid = res.resid
df['resid'] = resid
df

#%%

# dummies
