"""Side-by-side comparison of saved benchmark dumps (run_bench.py --dump), same scorer for all.

  python compare.py NAME=SYN_DUMP_DIR,YT_DUMP_DIR [NAME=...]
"""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_bench import score  # noqa: E402


def load(dump_dir, data_dir):
    out = {}
    for f in sorted(os.listdir(dump_dir)):
        name = f[:-5]
        ref = json.load(open(os.path.join(data_dir, f)))
        out[name] = score(json.load(open(os.path.join(dump_dir, f))), ref)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return st.mean(xs) if xs else float("nan")


def main():
    systems = {}
    for arg in sys.argv[1:]:
        name, dirs = arg.split("=", 1)
        syn, yt = dirs.split(",")
        systems[name] = {**load(syn, os.path.join(HERE, "data")), **load(yt, os.path.join(HERE, "data_yt"))}
    g = lambda m, scen, key: m.get(scen, {}).get(key)
    rows = [
        ("전사", "합성 한국어 CER", lambda m: mean([g(m, s, "cer_ko") for s in ("ko_2spk", "ko_2spk_noisy", "ko_2spk_samegender")]), "%"),
        ("전사", "합성 한·영 혼합 (CER / WER)", lambda m: (g(m, "mix_3spk", "cer_ko"), g(m, "mix_3spk", "wer_en")), "%/"),
        ("전사", "실제 한국어 혼자 말함 CER", lambda m: mean([g(m, s, "cer_ko") for s in ("yt_ko_mono_d", "yt_ko_mono_e")]), "%"),
        ("전사", "실제 한국어 잡담 CER", lambda m: g(m, "yt_ko_chat_c", "cer_ko"), "%"),
        ("전사", "실제 한국어 인터뷰 CER", lambda m: mean([g(m, s, "cer_ko") for s in ("yt_ko_interview_a", "yt_ko_interview_b")]), "%"),
        ("전사", "실제 영어 TED WER", lambda m: mean([g(m, s, "wer_en") for s in ("yt_en_ted_f", "yt_en_ted_g", "yt_en_ted_h")]), "%"),
        ("화자", "합성 화자 정확도", lambda m: mean([g(m, s, "speaker_acc") for s in ("ko_2spk", "ko_2spk_noisy", "ko_2spk_samegender", "mix_3spk")]), "%"),
        ("화자", "실제 5명 혼합: 정확도 / 찾은 화자 수", lambda m: (g(m, "yt_mix_real_5spk", "speaker_acc"), g(m, "yt_mix_real_5spk", "speakers_found")), "%n"),
        ("번역", "합성 한→영 chrF", lambda m: mean([g(m, s, "chrf_to_en") for s in ("ko_2spk", "ko_2spk_noisy", "ko_2spk_samegender")]), ""),
        ("번역", "실제 한→영 chrF (사람 번역 기준)", lambda m: mean([g(m, s, "chrf_to_en") for s in ("yt_ko_mono_d", "yt_ko_mono_e")]), ""),
        ("번역", "실제 영→한 chrF (사람 번역 기준)", lambda m: mean([g(m, s, "chrf_to_ko") for s in ("yt_en_ted_f", "yt_en_ted_g", "yt_en_ted_h")]), ""),
        ("지연", "임시 토큰 도착 p50 (실제)", lambda m: mean([g(m, s, "tok_nonfinal_p50_s") for s in m if s.startswith("yt_")]), "s"),
        ("지연", "확정 토큰 도착 p50 / p90 (실제)", lambda m: (mean([g(m, s, "tok_final_p50_s") for s in m if s.startswith("yt_")]),
                                                  mean([g(m, s, "tok_final_p90_s") for s in m if s.startswith("yt_")])), "ss"),
        ("지연", "자동 <end> p50 / p95 (실제)", lambda m: (mean([g(m, s, "end_lag_p50_s") for s in m if s.startswith("yt_")]),
                                                    mean([g(m, s, "end_lag_p95_s") for s in m if s.startswith("yt_")])), "ss"),
        ("지연", "번역 완료 p95 (실제, 원문 확정 기준)", lambda m: mean([g(m, s, "tr_done_lag_p95_s") for s in m if s.startswith("yt_")]), "s"),
    ]
    names = list(systems)
    print("| 항목 | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for cat, label, fn, fmt in rows:
        cells = []
        for n in names:
            v = fn(systems[n])
            if fmt == "%":
                cells.append(f"{100 * v:.1f}%")
            elif fmt == "%/":
                cells.append(f"{100 * v[0]:.1f}% / {100 * v[1]:.1f}%")
            elif fmt == "%n":
                cells.append(f"{100 * v[0]:.1f}% / {int(v[1])}명")
            elif fmt == "s":
                cells.append(f"{v:.2f}초")
            elif fmt == "ss":
                cells.append(f"{v[0]:.2f} / {v[1]:.2f}초")
            else:
                cells.append(f"{v:.1f}")
        print(f"| {cat}: {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
