"""kangkyu/icefall-asr-ko-streaming-zipformer-174m (Korean-only Zipformer2 causal transducer, sherpa-onnx int8, chunk-16)."""
import sherpa_onnx
from asr_models._sherpa import THREADS, mdir, online

SFX = "epoch-99-avg-1-chunk-16-left-128.int8.onnx"


def load(device):
    d = mdir("sherpa-onnx-kangkyu-zipformer-ko-174m")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{d}/tokens.txt", encoder=f"{d}/encoder-{SFX}", decoder=f"{d}/decoder-{SFX}",
        joiner=f"{d}/joiner-{SFX}", num_threads=THREADS)


def transcribe(rec, chunks, lang):
    return online(rec, chunks, None)  # Korean-only model: no language option
