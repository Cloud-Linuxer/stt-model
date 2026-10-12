"""Build a real-speech benchmark set from YouTube videos with human-made subtitles.

Download first (audio + manual subs), e.g. with yt-dlp:
  yt-dlp -f "bestaudio[ext=m4a]/bestaudio" --write-subs --sub-langs ko,en --sub-format vtt \
      -o "%(id)s.%(ext)s" https://www.youtube.com/watch?v=<id>
Then: python make_youtube_set.py <raw_dir> [out_dir] [--clips clips.json]
  ->  out_dir/<name>.wav + .json (same format as data/)
clips.json: {"clips": {name: [video_id, lang, translation_ref_lang|null, start_s, end_s]},
             "mix": [[speaker, video_id, lang, start_s, end_s], ...]}   (mix optional)
Subtitle files may carry variant suffixes (e.g. ID.ko-xxxx.vtt); the first match per language is used.

Clips without known speakers get speaker "?" (diarization is not scored for them).
`mix_real_5spk` interleaves cues from 5 different real speakers, so speaker truth is known.
"""
import json
import os
import random
import re
import sys

import numpy as np
import soundfile as sf
from faster_whisper import decode_audio

SR = 16000
HERE = os.path.dirname(os.path.abspath(__file__))

# name: (video id, source lang, translation-ref lang or None, clip start s, clip end s)
CLIPS = {
    "yt_ko_interview_a": ("cSwaRkus1q4", "ko", None, 30, 270),
    "yt_ko_interview_b": ("z6xCIEMf8wU", "ko", None, 14, 254),
    "yt_ko_chat_c": ("5VXY4vTTCwA", "ko", None, 41, 281),
    "yt_ko_mono_d": ("0CZRKFlQekU", "ko", "en", 0, 240),
    "yt_ko_mono_e": ("yh6FTtEVgv4", "ko", "en", 0, 240),
    "yt_en_ted_f": ("R1vskiVDwl4", "en", "ko", 12, 252),
    "yt_en_ted_g": ("eIho2S0ZahI", "en", "ko", 12, 252),
    "yt_en_ted_h": ("H14bBuluwB8", "en", "ko", 12, 252),
}
# real single-speaker stretches: (speaker, video id, lang, start, end)
MIX_SOURCES = [
    ("A", "0CZRKFlQekU", "ko", 0, 120),
    ("B", "5VXY4vTTCwA", "ko", 0, 38),
    ("C", "R1vskiVDwl4", "en", 12, 120),
    ("D", "eIho2S0ZahI", "en", 12, 120),
    ("E", "H14bBuluwB8", "en", 12, 120),
]


def sub_path(raw, vid, lang):
    import glob
    hits = sorted(glob.glob(os.path.join(raw, f"{vid}.{lang}.vtt")) + glob.glob(os.path.join(raw, f"{vid}.{lang}-*.vtt")))
    return hits[0] if hits else os.path.join(raw, f"{vid}.{lang}.vtt")


def parse_vtt(path):
    cues = []
    if not os.path.exists(path):
        return cues
    blocks = open(path, encoding="utf-8").read().split("\n\n")
    for b in blocks:
        lines = b.strip().splitlines()
        for i, ln in enumerate(lines):
            m = re.match(r"(\d+):(\d+):([\d.]+) --> (\d+):(\d+):([\d.]+)", ln)
            if m:
                h1, m1, s1, h2, m2, s2 = m.groups()
                text = " ".join(lines[i + 1:])
                text = re.sub(r"<[^>]+>", "", text)          # vtt tags
                text = re.sub(r"\([^)]*\)", "", text)        # (웃음), (공부) annotations
                text = re.sub(r"^(번역|검토|Translator|Reviewer)\s*:.*", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                if text and not re.match(r"^(번역|검토)\s*:", text):
                    cues.append((int(h1) * 3600 + int(m1) * 60 + float(s1),
                                 int(h2) * 3600 + int(m2) * 60 + float(s2), text))
                break
    return cues


def overlap_text(cues, start, end):
    return " ".join(t for s, e, t in cues if start <= (s + e) / 2 < end)


def find_audio(raw, vid):
    for ext in ("m4a", "webm", "mp4", "opus"):
        p = os.path.join(raw, f"{vid}.{ext}")
        if os.path.exists(p):
            return p
    return None


def main(raw, out, clips=CLIPS, mix=MIX_SOURCES, metas=None):
    os.makedirs(out, exist_ok=True)
    audio_cache = {}

    def audio(vid):
        if vid not in audio_cache:
            audio_cache[vid] = decode_audio(find_audio(raw, vid), sampling_rate=SR)
        return audio_cache[vid]

    for name, (vid, lang, tr_lang, cs, ce) in clips.items():
        if not find_audio(raw, vid):
            print("skip (no audio)", name)
            continue
        src = parse_vtt(sub_path(raw, vid, lang))
        tgt = parse_vtt(sub_path(raw, vid, tr_lang)) if tr_lang else []
        utts = [{"speaker": "?", "lang": lang, "text": t,
                 "ref_translation": overlap_text(tgt, s, e) if tgt else "",
                 "start": round(s - cs, 3), "end": round(e - cs, 3)}
                for s, e, t in src if s >= cs and e <= ce]
        a = audio(vid)[int(cs * SR):int(ce * SR)]
        sf.write(os.path.join(out, f"{name}.wav"), (np.clip(a, -1, 1) * 32767).astype(np.int16), SR,
                 subtype="PCM_16")
        meta = (metas or {}).get(vid, {})
        json.dump({"name": name, "source": f"https://www.youtube.com/watch?v={vid}", "duration": len(a) / SR,
                   "num_speakers": None, "translation_target": tr_lang, "utterances": utts,
                   "title": meta.get("title"), "channel": meta.get("channel"), "description": meta.get("description")},
                  open(os.path.join(out, f"{name}.json"), "w"), ensure_ascii=False, indent=1)
        print(f"{name}: {len(a)/SR:.0f}s, {len(utts)} cues")

    if not mix:
        return
    # real-voice multi-speaker mixture with known speaker labels
    rng = random.Random(0)
    pools = {}
    for src in mix:
        spk, vid, lang, s0, e0 = src[:5]
        skip = src[5] if len(src) > 5 else []   # [a, b] ranges of another voice (e.g. interview host), from the "- " cue markers
        if not find_audio(raw, vid):
            continue
        cues = [c for c in parse_vtt(sub_path(raw, vid, lang)) if c[0] >= s0 and c[1] <= e0
                and not any(c[0] < b and c[1] > a for a, b in skip)]
        # merge consecutive cues into ~3-8 s turns
        turns, cur = [], None
        for s, e, t in cues:
            if cur and s - cur[1] < 0.6 and e - cur[0] < 8:
                cur = (cur[0], e, cur[2] + " " + t)
            else:
                if cur:
                    turns.append(cur)
                cur = (s, e, t)
        if cur:
            turns.append(cur)
        pools[spk] = (vid, lang, [t for t in turns if t[1] - t[0] >= 1.5])
    parts, utts, t, prev = [np.zeros(int(SR * 0.5), np.float32)], [], 0.5, None
    for _ in range(24):
        spk = rng.choice([k for k in pools if k != prev and pools[k][2]])
        vid, lang, turns = pools[spk]
        s, e, text = turns.pop(0)
        seg = audio(vid)[int(s * SR):int(e * SR)]
        utts.append({"speaker": spk, "lang": lang, "text": text, "ref_translation": "",
                     "start": round(t, 3), "end": round(t + len(seg) / SR, 3)})
        gap = rng.uniform(0.2, 1.0)
        parts += [seg, np.zeros(int(SR * gap), np.float32)]
        t += len(seg) / SR + gap
        prev = spk
    a = np.concatenate(parts + [np.zeros(SR, np.float32)])
    sf.write(os.path.join(out, "yt_mix_real_5spk.wav"), (np.clip(a, -1, 1) * 32767).astype(np.int16), SR,
             subtype="PCM_16")
    json.dump({"name": "yt_mix_real_5spk", "duration": len(a) / SR,
               "num_speakers": len({u["speaker"] for u in utts}), "translation_target": None,
               "utterances": utts}, open(os.path.join(out, "yt_mix_real_5spk.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"yt_mix_real_5spk: {len(a)/SR:.0f}s, {len(utts)} turns, "
          f"{len({u['speaker'] for u in utts})} speakers")


if __name__ == "__main__":
    import argparse
    import glob
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out", nargs="?", default=os.path.join(HERE, "data_yt"))
    ap.add_argument("--clips", help="clips JSON (see module docstring); default: the built-in set 1")
    a = ap.parse_args()
    metas = {}
    for f in glob.glob(os.path.join(a.raw, "*.meta.json")):
        m = json.load(open(f))
        metas[m["id"]] = m
    if a.clips:
        spec = json.load(open(a.clips))
        main(a.raw, a.out, {k: tuple(v) for k, v in spec["clips"].items()}, [tuple(x) for x in spec.get("mix", [])], metas)
    else:
        main(a.raw, a.out, metas=metas)
