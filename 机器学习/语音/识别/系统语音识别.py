

# win11 语音识别

import win32com.client

reco = win32com.client.Dispatch("SAPI.SpInProcRecoContext")
reco.Recognizer = win32com.client.Dispatch("SAPI.SpSharedRecognizer")
reco.Recognizer.AudioInput = reco.AudioInputs.Item(0)
reco.Recognizer.RecognizeStream = reco.CreateStream()
reco.Recognizer.State = 2  # 连续模式
reco.Recognizer.AudioInputStream = reco.Recognizer.RecognizeStream
reco.Voice = reco.GetVoices("Name=Microsoft Zira Desktop").Item(0)  # 使用Microsoft Zira Desktop作为语音合成器

while True:
    event = reco.WaitUntilEvent(1000)
    if event:
        if event.Type == win32com.client.constants.SpeechRecognitionType.SpeechHypothesis:
            text = event.Text
            print("识别结果:", text)