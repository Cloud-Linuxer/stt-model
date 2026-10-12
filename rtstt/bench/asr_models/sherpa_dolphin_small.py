MAX_S = 8.0  # model input length limit
import sherpa_onnx
from asr_models._sherpa import THREADS, mdir, offline


def load(device):
    d = mdir("sherpa-onnx-dolphin-small-ctc-multi-lang-int8")
    return sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
        model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt", num_threads=THREADS)


def transcribe(rec, chunks, lang):
    return offline(rec, chunks)
