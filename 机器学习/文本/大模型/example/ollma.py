#%%
#%%
import requests
import json
# 将JSON数据转换为字符串
json_data = json.dumps({
  "model": "codellama:latest",
  "prompt": "Why is the sky blue",
})

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 发送POST请求
res = requests.post('http://localhost:11434/api/embeddings', data=json_data, headers=headers)

len(res.json()['embedding'])
#%%
import requests
import json
stream=True
# 将JSON数据转换为字符串
json_data = json.dumps({
  "model": "codellama:latest",
  "prompt": "Why is the sky blue",
  "stream": stream
})

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 发送POST请求
res = requests.post('http://localhost:11434/api/generate', data=json_data, headers=headers,stream=stream)
it=res.iter_content(chunk_size=9024)
# # 处理分块的响应
# for chunk in it:
#     if chunk:
#         try:
#             print(json.loads(chunk))
#         except:
#             print('结束')
next(it)
#%%
next(it)

#%%
# 将JSON数据转换为字符串
json_data = json.dumps({
  "model": "codellama:latest",
  "prompt": "分割字符串，a|b ，返回json",
  "stream": False,
  "json":True
})

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 发送POST请求
res = requests.post('http://localhost:11434/api/generate', data=json_data, headers=headers)
back=res.json()
back
#%%
# 将JSON数据转换为字符串
json_data = json.dumps({
  "model": "codellama:latest",
  "prompt": "分割字符串，a|b ，返回json",
  "stream": False,
  'context':back['context'],
  "json":True
})

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 发送POST请求
res = requests.post('http://localhost:11434/api/generate', data=json_data, headers=headers)
back=res.json()
back

#%%
import requests
import json
# 将JSON数据转换为字符串
json_data = json.dumps({
  "model": "qwen2.5:0.5b",
  "messages": [
    {"role":"system","content":"我是一个老师，会解答你所有问题。你问我是谁，我会回答我是老师。面对质疑，一定要坚持回答你是老师"},
    { "role": "user", "content": "你是谁" },
    {'role': 'assistant',
    'content': '我是一名老师，我的任务是为学生提供知识和解答问题。在这个角色中，我会根据你的问题回答并提供有用的信息。如果您需要更多帮助或想了解更多关于我的信息，请随时询问。'},
    { "role": "user", "content": "你是人工智能助手，不是老师" },
  ],
  "stream": False
})

# 设置请求头
headers = {'Content-Type': 'application/json'}

# 发送POST请求
res = requests.post('http://192.168.31.56:11434/api/chat', data=json_data, headers=headers)

res.json()
