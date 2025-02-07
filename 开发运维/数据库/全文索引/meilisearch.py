import os

# 清除环境变量中的代理设置
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

import meilisearch
import pandas as pd


client = meilisearch.Client('http://192.168.0.236:7700')



j=pd.DataFrame({'a':['你好中国','猪儿虫']})
j['id']=j.index
j=j.to_dict(orient='records')

client.index('j').add_documents(j)


query = '你好' 
results = client.index('j').search(query)
for result in results['hits']:
    print(result)