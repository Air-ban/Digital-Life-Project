import rapid_paraformer as rp
import sounddevice as sd
import soundfile as sf
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class AudioTranscription:
    def __init__(self, model_dir, vad_model="fsmn-vad", duration=5, fs=44100, channels=2):
        self.model_dir = model_dir
        self.duration = duration
        self.fs = fs
        self.channels = channels
        self.model = AutoModel(
            model=self.model_dir,
            vad_model=vad_model,
            vad_kwargs={"max_single_segment_time": 30000},
            hub="hf",
        )
        self.transcription_text = ""  # 添加一个实例变量来保存转录文本
        self.filename = "temp.wav"

    def record_audio(self):
        recording = sd.rec(int(self.duration * self.fs), samplerate=self.fs, channels=self.channels)
        sd.wait()  # 等待录音完成
        sf.write(self.filename, recording, self.fs)  # 保存录音

    def transcribe_audio(self):
        res = self.model.generate(
            input=self.filename,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text

    def save_transcription(self, text):
        # 将转录文本保存到实例变量中，而不是写入文件
        self.transcription_text = text

    def run(self):
            self.record_audio()
            text = self.transcribe_audio()
            print(text)
            self.save_transcription(text)  # 调用修改后的方法来保存转录文本