MAX_S = 8.0  # model input length limit
import sherpa_onnx
from asr_models._sherpa import THREADS, mdir, offline


def load(device):
    d = mdir("sherpa-onnx-moonshine-tiny-ko-quantized")
    return sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
        encoder=f"{d}/encoder_model.ort", decoder=f"{d}/decoder_model_merged.ort", tokens=f"{d}/tokens.txt",
        num_threads=THREADS)


def transcribe(rec, chunks, lang):
    return offline(rec, chunks)
