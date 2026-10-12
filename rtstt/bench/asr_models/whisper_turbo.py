from asr_models._fw import load_fw, transcribe_fw

PATH = "/home/stt-model/models/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"


def load(device):
    return load_fw(PATH, device)


transcribe = transcribe_fw
