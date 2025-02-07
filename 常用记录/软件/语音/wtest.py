# python -m pip install openai-whisper -i https://mirror.baidu.com/pypi/simple
# python -m pip install setuptools-rust -i https://mirror.baidu.com/pypi/simple
# 需设置ffmpeg
import whisper
 
model = whisper.load_model("base")
result = model.transcribe("output.wav")
print(result["text"])