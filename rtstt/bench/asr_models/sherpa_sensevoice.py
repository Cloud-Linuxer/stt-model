MAX_S = 8.0  # model input length limit
import sherpa_onnx
from asr_models._sherpa import THREADS, mdir, offline


def load(device):
    d = mdir("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
    return {lang: sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt", num_threads=THREADS, language=lang, use_itn=True)
        for lang in ("ko", "en", "auto")}


def transcribe(recs, chunks, lang):
    return offline(recs[lang], chunks)
