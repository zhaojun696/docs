
#%%
import pandas  as pd
pd.options.plotting.backend='plotly'
pd.Series(range(20)).plot.line()
#%%

# 如果用Series进行复制，会比对index。index会赋值到相对应的位置
import pandas as pd
a=pd.DataFrame({'a':1},index=[1])
b=pd.DataFrame({'a':[3,5]},index=[3,1])
a['b']=b.a
a
#%%
# 运算也会对准index
a+b
#%%
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'hobbies': [['reading', 'cooking'], ['hiking', 'gardening'], ['painting']]
})

# 展开带数组的列
df = df.explode('hobbies')

df