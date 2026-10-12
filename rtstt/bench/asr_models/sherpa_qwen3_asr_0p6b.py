import sherpa_onnx
from asr_models._sherpa import THREADS, mdir, offline


def load(device):
    d = mdir("sherpa-onnx-qwen3-asr-0.6B-int8")
    return sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=f"{d}/conv_frontend.onnx", encoder=f"{d}/encoder.int8.onnx",
        decoder=f"{d}/decoder.int8.onnx", tokenizer=f"{d}/tokenizer", num_threads=THREADS,
        max_new_tokens=512)


def transcribe(rec, chunks, lang):
    return offline(rec, chunks)
