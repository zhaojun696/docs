# pip install -U openai-whisper
# pip install whisper

# pip3 install torch torchvision torchaudio
# # 注：没科学上网会下载有可能很慢，可以替换成国内镜像加快下载速度
# pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
# whisper zh.wav --model small
# whisper audio.mp3 --model medium
# whisper japanese.wav --language Japanese
# whisper chinese.mp4 --language Chinese --task translate
# whisper audio.flac audio.mp3 audio.wav --model medium
# whisper output.wav --model medium  --language Chinese

#%%
import torch
import whisper  
audio_path='zh.wav'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
audio = whisper.load_audio(audio_path)  
audio = whisper.pad_or_trim(audio)

model = whisper.load_model("small-v2")

mel = whisper.log_mel_spectrogram(audio).to(model.device)

options = whisper.DecodingOptions(beam_size=5)

result = whisper.decode(model, mel, options)  
print(result.text)

