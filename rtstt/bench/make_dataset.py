"""Build synthetic multi-speaker benchmark set with ground truth.

Each scenario -> data/<name>.wav (16 kHz mono s16) + data/<name>.json:
  {"utterances": [{"speaker", "lang", "text", "ref_translation", "start", "end"}], ...}
TTS: edge-tts CLI (needs network). Usage: python make_dataset.py [--edge-tts-path DIR]
"""
import argparse
import json
import os
import random
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from faster_whisper import decode_audio

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))

# (speaker_id, voice) ; text ; reference translation (ko->en, en->ko)
DIALOG_KO = [
    ("A", "안녕하세요, 테이블오더 고객센터입니다. 무엇을 도와드릴까요?", "Hello, this is the Table Order customer center. How can I help you?"),
    ("B", "네, 매장에 있는 태블릿이 와이파이에 연결이 안 돼요.", "Yes, the tablet in our store won't connect to Wi-Fi."),
    ("A", "불편을 드려 죄송합니다. 혹시 공유기를 최근에 교체하셨나요?", "Sorry for the inconvenience. Did you recently replace your router?"),
    ("B", "아니요, 어제까지는 잘 됐는데 오늘 아침부터 안 돼요.", "No, it worked fine until yesterday, but it stopped working this morning."),
    ("A", "그러면 관리자 메뉴에 들어가셔서 네트워크 설정을 한 번 확인해 주시겠어요?", "Then could you go into the admin menu and check the network settings?"),
    ("B", "관리자 비밀번호가 기억이 안 나는데 어떻게 하죠?", "I can't remember the admin password, what should I do?"),
    ("A", "사업자 등록번호 뒷자리 네 자리를 입력하시면 됩니다.", "You can enter the last four digits of your business registration number."),
    ("B", "아, 들어가졌어요. 와이파이 목록에 저희 매장 이름이 안 보여요.", "Oh, I'm in. I don't see our store's name in the Wi-Fi list."),
    ("A", "공유기 전원을 껐다가 삼십 초 후에 다시 켜 주세요.", "Please turn off the router and turn it back on after thirty seconds."),
    ("B", "네, 다시 켰더니 이제 연결됐습니다. 감사합니다.", "Okay, I turned it back on and it's connected now. Thank you."),
]
DIALOG_MIX = [
    ("A", "오늘 회의에서는 신규 결제 단말기 도입 일정을 논의하겠습니다.", "In today's meeting, we will discuss the schedule for introducing the new payment terminals."),
    ("C", "Thanks. Our engineering team can deliver the firmware update by the end of next month.", "감사합니다. 저희 엔지니어링 팀은 다음 달 말까지 펌웨어 업데이트를 제공할 수 있습니다."),
    ("B", "그럼 시범 매장은 강남점과 판교점 두 곳으로 하면 될 것 같아요.", "Then I think we can use two stores, Gangnam and Pangyo, as pilot stores."),
    ("C", "That works for us. We will need remote access to both stores for testing.", "저희는 괜찮습니다. 테스트를 위해 두 매장 모두 원격 접속이 필요합니다."),
    ("A", "원격 접속 권한은 보안팀 승인을 받아서 다음 주 화요일까지 드리겠습니다.", "We will get approval from the security team and give you remote access by next Tuesday."),
    ("B", "결제 수수료 정산 방식도 미리 정리해 두는 게 좋겠습니다.", "It would be good to sort out the payment fee settlement method in advance."),
    ("C", "Agreed. I'll send a draft proposal by Friday.", "동의합니다. 금요일까지 초안 제안서를 보내드리겠습니다."),
    ("A", "좋습니다. 그럼 오늘 회의는 여기서 마치겠습니다.", "Good. Then let's end today's meeting here."),
]

SCENARIOS = {
    # name: (dialog, voice map, gap range seconds, snr_db or None)
    "ko_2spk": (DIALOG_KO, {"A": "ko-KR-SunHiNeural", "B": "ko-KR-InJoonNeural"}, (0.25, 1.2), None),
    "ko_2spk_samegender": (DIALOG_KO, {"A": "ko-KR-InJoonNeural", "B": "ko-KR-HyunsuMultilingualNeural"}, (0.25, 1.2), None),
    "ko_2spk_noisy": (DIALOG_KO, {"A": "ko-KR-SunHiNeural", "B": "ko-KR-InJoonNeural"}, (0.25, 1.2), 15),
    "mix_3spk": (DIALOG_MIX, {"A": "ko-KR-SunHiNeural", "B": "ko-KR-InJoonNeural", "C": "en-US-GuyNeural"}, (0.25, 1.2), None),
}


def tts(text, voice, edge_path, tmp):
    mp3 = os.path.join(tmp, "u.mp3")
    env = dict(os.environ, PYTHONPATH=edge_path) if edge_path else None
    subprocess.run(["python3", "-m", "edge_tts", "--voice", voice, "--text", text, "--write-media", mp3],
                   check=True, env=env, capture_output=True)
    a = decode_audio(mp3, sampling_rate=SR)
    # trim TTS leading/trailing silence so ground-truth boundaries are tight
    idx = np.where(np.abs(a) > 0.01)[0]
    return a[max(0, idx[0] - 800): idx[-1] + 800]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-tts-path", help="dir with edge_tts installed (pip --target)")
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)
    cache = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, (dialog, voices, (gmin, gmax), snr) in SCENARIOS.items():
            parts, utts, t = [np.zeros(int(SR * 0.5), np.float32)], [], 0.5
            for spk, text, ref in dialog:
                key = (voices[spk], text)
                if key not in cache:
                    cache[key] = tts(text, voices[spk], args.edge_tts_path, tmp)
                a = cache[key]
                lang = "en" if voices[spk].startswith("en") else "ko"
                utts.append({"speaker": spk, "lang": lang, "text": text, "ref_translation": ref,
                             "start": round(t, 3), "end": round(t + len(a) / SR, 3)})
                gap = rng.uniform(gmin, gmax)
                parts += [a, np.zeros(int(SR * gap), np.float32)]
                t += len(a) / SR + gap
            parts.append(np.zeros(SR, np.float32))
            audio = np.concatenate(parts)
            if snr is not None:
                p = np.mean(audio[audio != 0] ** 2)
                noise = np.random.default_rng(0).normal(0, np.sqrt(p / 10 ** (snr / 10)), len(audio))
                audio = (audio + noise).astype(np.float32)
            audio = np.clip(audio, -1, 1)
            sf.write(os.path.join(args.out, f"{name}.wav"), (audio * 32767).astype(np.int16), SR, subtype="PCM_16")
            json.dump({"name": name, "duration": len(audio) / SR, "num_speakers": len(voices),
                       "utterances": utts}, open(os.path.join(args.out, f"{name}.json"), "w"),
                      ensure_ascii=False, indent=1)
            print(f"{name}: {len(audio)/SR:.1f}s, {len(utts)} utts")


if __name__ == "__main__":
    main()
