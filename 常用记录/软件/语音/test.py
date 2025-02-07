
#%%
import pyttsx3
#语音播放 
# pyttsx3.speak("swarm")
# pyttsx3.speak("actions")
# pyttsx3.speak("Elasticsearch")
# pyttsx3.speak("Logstash")
# pyttsx3.speak("Kibana")
# pyttsx3.speak("Hyper")
pyttsx3.speak("cousor")

#%%
pyttsx3.speak("I am fine, thank you")

# 如果我们想要修改语速、音量、语音合成器等，可以用如下方法。
# 1、pyttsx3通过初始化来获取语音引擎，在调用init后会返回一个engine对象。

import pyttsx3
engine = pyttsx3.init() #初始化语音引擎
rate = engine.getProperty('rate')
print(f'语速：{rate}')
volume = engine.getProperty('volume')   
print (f'音量：{volume}') 
# 运行结果为：
# 语速：200
# 音量：1.0

# 3、设置语速、音量等参数
engine.setProperty('rate', 100)   #设置语速
engine.setProperty('volume',0.6)  #设置音量

# 4、查看语音合成器
voices = engine.getProperty('voices') 
for voice in voices:
    print(voice) 
# 运行结果如下：

# <Voice id=HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ZH-CN_HUIHUI_11.0
#           name=Microsoft Huihui Desktop - Chinese (Simplified)
#           languages=[]
#           gender=None
#           age=None>
# <Voice id=HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0
#           name=Microsoft Zira Desktop - English (United States)
#           languages=[]
#           gender=None
#           age=None>
    
# 合成器的主要参数如下：
# age 发音人的年龄，默认为None
# gender 以字符串为类型的发音人性别: male, female, or neutral.默认为None
# id 关于Voice的字符串确认信息
# languages 发音支持的语言列表，默认为一个空的列表
# name 发音人名称，默认为None
# 默认的语音合成器有两个，两个语音合成器均可以合成英文音频，但只有第一个合成器能合成中文音频。如果需要其他的语音合成器需要自行下载和设置。

# 5、设置语音合成器
# 若我们需要第一个语音合成器，代码如下：

voices = engine.getProperty('voices') 
engine.setProperty('voice',voices[0].id)   #设置第一个语音合成器

# 6、语音播报
engine.say("春光灿烂猪八戒")
engine.runAndWait()
engine.stop()

# 四、全套代码
import pyttsx3
engine = pyttsx3.init() #初始化语音引擎
 
engine.setProperty('rate', 100)   #设置语速
engine.setProperty('volume',0.6)  #设置音量
voices = engine.getProperty('voices') 
engine.setProperty('voice',voices[0].id)   #设置第一个语音合成器
engine.say("春光灿烂猪八戒")
engine.save_to_file('text', 'output.wav')

engine.runAndWait()

