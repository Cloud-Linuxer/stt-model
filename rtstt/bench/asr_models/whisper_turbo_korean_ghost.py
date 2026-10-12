from asr_models._fw import load_fw, transcribe_fw

PATH = "/home/stt-model/models/ct2-ghost613-turbo-korean"


def load(device):
    return load_fw(PATH, device)


transcribe = transcribe_fw
