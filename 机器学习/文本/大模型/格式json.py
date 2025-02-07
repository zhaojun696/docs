#%%
from ollama import chat
from enum import Enum
from pydantic import BaseModel,Field,ConfigDict
from typing import Union,Optional


class cat(Enum):
    '''
    类别列表
    '''
    衣服='衣服'
    电脑='电脑'
    其他='其他'


class news(BaseModel):
    '''
    信息提取
    '''

    class Config:
        # title = "NewsSchema"  # 设置自定义 title
        
        @staticmethod
        def json_schema_extra(schema, model):
            # 将顶层的 title 改为 name
            if "title" in schema:
                schema["name"] = schema.pop("title")

            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    if "title" in prop_schema:
                        prop_schema["name"] = prop_schema.pop("title")


    名字: str=Field(description='名字')
    cats: cat=Field(description='商品类别',title='类别')
    评价: str=Field(description='对商品的评价描述',title='描述')
    评级最小1最大5: int=Field(description='对商品的评价等级',title='等级',gt=1,lt=5)
    

    

schema=news.model_json_schema()
schema
#%%
response = chat(
    messages=[
    {
        'role': 'system',
        'content': '''
            你是一个商品信息提取专家，需要跟据用户输入。返回json格式。
        ''',
    },
    {
        'role': 'user',
        'content': '''
            赵军是帅哥
        ''',
    }
    ],
    model='qwen2.5:1.5b',
    format=schema,
    options={
        'temperature':0
    },
    keep_alive=True
)

pets = news.model_validate_json(response.message.content)
pets
# %%
pets.cats.name



#%%
from ollama import chat
from enum import Enum
from pydantic import BaseModel,Field
from typing import Union,Optional


class cat(Enum):
    '''
    类别列表
    '''
    服装='服装'
    电脑='电脑'
    其他='其他'

class news(BaseModel):
    人名: str=Field(description='名字或者姓名',default=None)
    类别_无合适类别选其他: cat=Field(description='商品类别')
    满意度_最小1最大5: int=Field(gt=0,lt=5)

schema=news.model_json_schema()

fiels='(%s)'%','.join(list(news.model_fields.keys()))

response = chat(
    messages=[
        {
            'role': 'system',
            'content': f'''
                你是一个信息提取专家，需要跟据用户输入提取信息,回复json格式。
            ''',
        },
        {
            'role': 'user',
            'content': f'''
                赵军在衣服领域，声誉很高
            ''',
        }
    ],
    model='qwen2.5:1.5b',
    format=schema,
    options={
    'temperature':0
    },
    keep_alive=True
)

pets = news.model_validate_json(response.message.content)
pets
#%%

# news.model_validate({"类别": "衣服", "名字": "赵军"})
#%%
pets.人名
# %%
import ollama
import numpy as np 
a=ollama.embeddings(model='granite-embedding:278m',prompt='双眼皮手术')
b=ollama.embeddings(model='granite-embedding:278m',prompt='割双眼皮')
a=np.array(a.embedding)
b=np.array(b.embedding)


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(a.reshape(1, -1),b.reshape(1, -1))
