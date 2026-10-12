import os
from eot_models._smartturn import MODELS, SmartTurn


def load():
    return SmartTurn(os.path.join(MODELS, "smart-turn-v3.2-gpu.onnx"), ["CPUExecutionProvider"])   # fp32 weights, CPU run


def complete(h, audio):
    return h(audio)
