import rapid_paraformer as rp
import sounddevice as sd
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "FunAudioLLM/SenseVoiceSmall" #指定模型
#加载模型
model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    #cuda设备，我没有，我也不配
    #device="cuda:0", 
    hub="hf",
)

duration = 5  # 录音时长（秒）
fs = 44100  # 采样率（Hz）
channels = 2  # 声道数

# 循环运行Paraformer模型
while True:
    res = model.generate(
    input=r"C:\Users\xiaox\Documents\GitHub\Digital-Life-Project\temp.wav",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # 等待录音完成
    filename = 'temp.wav'  # 保存的文件名
    sf.write(filename, recording, fs)  # 保存录音
    text = rich_transcription_postprocess(res[0]["text"])
    print(text)
    with open(r'input\user_words.txt', 'w', encoding='utf-8') as file:
        file.write(text)