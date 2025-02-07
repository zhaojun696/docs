# pip install torch torchaudio
# pip intall funasr
# pip install modelscope

# https://www.modelscope.cn/models

from funasr import AutoModel

class SpeechReco:
    
    model_path = './models_from_modelscope'      # 指定模型的路径，否则会自动从新下载
    auto_model = AutoModel(
        model=model_path + 'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        vad_model=model_path + 'speech_fsmn_vad_zh-cn-16k-common-pytorch',
        punc_model=model_path + 'punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
        disable_log=True,
        disable_pbar=True,
        use_timestamp=False
    )

    @staticmethod
    def genSpeechText(audio_src):
        res = SpeechReco.auto_model.generate(input=audio_src)
        print('rec_result["text"]:',res[0]['text'])
        return res[0]['text']     # 语音识别为文字后的结果