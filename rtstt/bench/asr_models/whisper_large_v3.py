from asr_models._fw import load_fw, transcribe_fw

PATH = "/home/stt-model/models/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"


def load(device):
    return load_fw(PATH, device)


transcribe = transcribe_fw
