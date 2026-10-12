import sherpa_onnx
from asr_models._sherpa import THREADS, mdir, online


def load(device):
    d = mdir("sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt", encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx",
        joiner=f"{d}/joiner.int8.onnx", num_threads=THREADS)


def transcribe(rec, chunks, lang):
    return online(rec, chunks, lang)
