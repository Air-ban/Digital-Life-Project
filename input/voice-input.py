import rapid_paraformer as rp
import sounddevice as sd
import soundfile as sf

duration = 5  # 录音时长（秒）
fs = 44100  # 采样率（Hz）
channels = 2  # 声道数

# 初始化Paraformer模型
config_path = "resources_config.yaml"
paraformer = rp.RapidParaformer(config_path)

# 音频文件路径列表
wav_paths = [
    r"temp.wav",
    # 更多音频文件路径...
]

# 循环运行Paraformer模型
while True:
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()  # 等待录音完成
    filename = 'temp.wav'  # 保存的文件名
    sf.write(filename, recording, fs)  # 保存录音
    for wav_path in wav_paths:
        result = paraformer(wav_path)
        print(result)

# 未达到可用预期，需后期优化