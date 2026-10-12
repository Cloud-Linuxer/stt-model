import sherpa_onnx
from asr_models._sherpa import THREADS, mdir, offline


def load(device):
    d = mdir("sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8")
    return sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=f"{d}/model.int8.onnx", tokens=f"{d}/tokens.txt", num_threads=THREADS)


def transcribe(rec, chunks, lang):
    return offline(rec, chunks)
