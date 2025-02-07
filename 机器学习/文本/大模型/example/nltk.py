#%%
# rag
# 检索：将用户查询用于检索外部知识源中的相关上下文。为此，要使用一个嵌入模型将该用户查询嵌入到同一个向量空间中，使其作为该向量数据库中的附加上下文。这样一来，就可以执行相似性搜索，并返回该向量数据库中与用户查询最接近的 k 个数据对象。

# 增强：然后将用户查询和检索到的附加上下文填充到一个 prompt 模板中。

# 生成：最后，将经过检索增强的 prompt 馈送给 LLM。

# 角色
# 1 system：
# 它设定了 AI 的行为和角色，和背景。
# 常常用于开始对话，给出一个对话的大致方向，或者设置对话的语气和风格。
# 例如，可以把它设置为：“你是一个助理”或“你是一名历史教师”。这个消息可以帮助设定对话的语境，以便 AI 更好地理解其在对话中的角色。
# 也可以更加详细地进行设置。比如说，你需要一个导游，可以把它设置为：“我想让你做一个导游。我会把我的位置写给你，你会推荐一个靠近我的位置的地方。在某些情况下，我还会告诉您我将访问的地方类型。您还会向我推荐靠近我的第一个位置的类似类型的地方。”

# 2 user
# 就是我们输入的问题或请求。
# 比如说“北京王府井附近有什么值得去的地方？”

# 3 assistant
# 在使用 API 的过程中，你不需要直接生成 assistant 消息，因为它们是由 API 根据 system 和 user 消息自动生成的。




# 任务拆解
# 拆解，下面句子。如：姚明多少岁，他的岁数平方是多少（拆解格式为json数组['姚明多少岁'，'他的岁数平方是多少']），不要返回其他无关的解释内容
# 那家医美比较好，有那些项目，价格如何


#agent
# Answer the following questions as best you can.  You have access to the following tools: 

# Calculator: 如果是关于数学计算的问题，请使用它
# Search: 如果我想知道姚明岁数时这两个问题时，请使用它 
# Use the following format: 
# Question: the input question you must answer 
# Thought: you should always think about what to do
# Action: the action to take, should be one of [Calculator, Search] 
# Action Input: the input to the action 
# Observation: the result of the action
# ...  (this Thought/Action/Action Input/Observation can repeat N times) 
# Thought: I now know the final answer 
# Final Answer: the final answer to the original input question 

# Begin! 
# Question: 告诉我'姚明多少岁，他的岁数的平方是多少'是什么意思 # 问输入的问题
# Thought: 



# # 尽可能的去回答以下问题，你可以使用以下的工具：
# Answer the following questions as best you can.  You have access to the following tools: 

# Calculator: 如果是关于数学计算的问题，请使用它
# Search: 如果我想知道天气，'鸡你太美'这两个问题时，请使用它 
# Use the following format: # 请使用以下格式(回答)

# # 你必须回答输入的问题
# Question: the input question you must answer 
# # 你应该一直保持思考，思考要怎么解决问题
# Thought: you should always think about what to do
# # 你应该采取[计算器,搜索]之一
# Action: the action to take, should be one of [Calculator, Search] 
# Action Input: the input to the action # 动作的输入
# Observation: the result of the action # 动作的结果
# # 思考-行动-输入-输出 的循环可以重复N次
# ...  (this Thought/Action/Action Input/Observation can repeat N times) 
# # 最后，你应该知道最终结果
# Thought: I now know the final answer 
# # 针对于原始问题，输出最终结果
# Final Answer: the final answer to the original input question 

# Begin! # 开始
# Question: 告诉我'鸡你太美'是什么意思 # 问输入的问题
# Thought: 