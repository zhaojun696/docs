
#%%

import json
from openai import OpenAI
client = OpenAI(api_key="sk-4dc651d196134b0abb1bdbe0034f2983", base_url="https://api.deepseek.com")
client.models.list()

#%%
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    max_tokens=1024,
    temperature=0.7,
    stream=False
)

#%%

# deepseek-chat 不支持函数调用
modelName='deepseek-chat'


# 第一步：发送会话和函数调用给模型
messages = [{"role": "user", "content": "北京、上海、承德3地的天气现在是什么样的?"}]
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "返回指定地区的温度",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "承德市双桥区",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "华氏温度"]},
                },
                "required": ["location"],
            },
        },
    }
]
response = client.chat.completions.create(
    model=modelName,
    messages=messages,
    tools=tools,
    tool_choice="auto",  # auto 默认值，明确给出
)
response_message = response.choices[0].message
tool_calls = response_message.tool_calls
tool_calls
#%%
# 例子模拟函数调用的硬编码，返回相同数据格式的天气
# 在实际生产中，可以是后台API或第三方的API fahrenheit-华氏温度
def get_current_weather(location, unit="fahrenheit"):
    """返回指定地区的实时天气"""
    if "北京" in location.lower():
        return json.dumps({"location": "北京", "temperature": "16", "unit": unit})
    elif "上海" in location.lower():
        return json.dumps({"location": "上海", "temperature": "22", "unit": unit})
    elif "承德" in location.lower():
        return json.dumps({"location": "承德", "temperature": "12", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "未知"})

# 第二步： 检查模型是否需要调用一个函数
if tool_calls:
    # 第三步: 调用函数
    # 注意：返回的JSON消息不一定有效，一定要检查错误
    available_functions = {
        "get_current_weather": get_current_weather,
    }  # 这个例子只有一个参数调用，当然也可以有多个
    messages.append(response_message)  # 将助理的回复加入到消息中
    # 第四步: 将每一个函数调用及其相应的响应发送给模型
    for tool_call in tool_calls:
        function_name = tool_call.function.name  # 返回函数调用名
        function_to_call = available_functions[function_name]  # 返回JSON不一定有效，使其有效
        function_args = json.loads(tool_call.function.arguments)  # 将函数所用的参数解析为JSON
        function_response = function_to_call(  # 调用函数并传指定参数
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        """
            函数调用返回结果加入新的会话
            "tool_call_id": tool_call.id, -> 函数返回的ID
            "role": "tool", -> role 类型为tool
            "name": function_name, -> 动态解析的函数调用名
            "content": function_response, -> 内容为函数调用返回结果
        """
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )
        # 模型理解函数返回的响应后，并返回一个新的响应
    second_response = client.chat.completions.create(
        model=modelName,
        messages=messages,
    )




