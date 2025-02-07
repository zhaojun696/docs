#%%
from openai import OpenAI
client = OpenAI(
    api_key='sk-77274694b69741d98f92ecd2902cb43c',
    base_url='https://api.deepseek.com/',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

system = """Answer the following questions as best you can. You have access to the following APIs:
1. fire_recognition: Call this tool to interact with the fire recognition API. This API is used to recognize whether there is fire in the image. Parameters: [{\"name\": \"image\", \"description\": \"The input image to recognize fire\", \"required\": \"True\"}]

2. fire_alert: Call this tool to interact with the fire alert API. This API will start an alert to warn the building's administraters. Parameters: []

3. call_police: Call this tool to interact with the police calling API. This API will call 110 to catch the thief. Parameters: []

4. call_fireman: Call this tool to interact with the fireman calling API. This API will call 119 to extinguish the fire. Parameters: []

Use the following format:

Thought: you should always think about what to do
Action: the action to take, should be one of the above tools[fire_recognition, fire_alert, call_police, call_fireman]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!"""
messages = [{
    'role': 'system',
    'content': system
}, {
    'role': 'user',
    'content': '输入图片是/tmp/1.jpg，协助判断图片中是否存在着火点'
}]
resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stop=['Observation:'],
    seed=42)
response = resp.choices[0].message.content
print(f'response: {response}')

# 流式
messages.append({'role': 'assistant', 'content': response + "\n[{'coordinate': [101.1, 200.9], 'on_fire': True}]"})
print(messages)
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stop=['Observation:'],
    stream=True,
    seed=42)

print('response:\r\n ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
# %%
