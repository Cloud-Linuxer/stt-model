"""Effect of the start-request `context` on transcription and translation.

Streams each case twice (without / with context) and reports the error rate, term recall
(fraction of context-term occurrences in the reference that appear in the final transcript)
and glossary hits (translation_terms targets found in the translation).

  python context_eval.py [--url ws://localhost:8765/transcribe-websocket] [--cases ko_2spk,...]
"""
import argparse
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_bench import norm, score, stream  # noqa: E402

CASES = {
    "ko_2spk": {"dir": "data", "hints": ["ko"], "translation": {"type": "one_way", "target_language": "en"}, "context": {
        "general": [{"key": "domain", "value": "테이블오더 매장 태블릿 고객센터 상담 통화"}],
        "terms": ["테이블오더", "와이파이", "공유기", "관리자 메뉴", "사업자 등록번호", "네트워크 설정"],
        "translation_terms": [{"source": "테이블오더", "target": "Table Order"}, {"source": "와이파이", "target": "Wi-Fi"}]}},
    "mix_3spk": {"dir": "data", "hints": ["ko", "en"], "translation": {"type": "two_way", "language_a": "ko", "language_b": "en"}, "context": {
        "general": [{"key": "domain", "value": "결제 단말기 도입 회의, 한국어와 영어 참석자"}],
        "terms": ["결제 단말기", "강남점", "판교점", "펌웨어 업데이트", "원격 접속", "수수료 정산"],
        "translation_terms": [{"source": "판교점", "target": "Pangyo branch"}, {"source": "강남점", "target": "Gangnam branch"}]}},
    "yt_ko_mono_d": {"dir": "data_yt", "hints": ["ko"], "translation": {"type": "one_way", "target_language": "en"}, "context": {
        "general": [{"key": "speaker", "value": "한국어 한 조각 채널의 현 선생님"}, {"key": "topic", "value": "주말과 휴일 보내기"}],
        "terms": ["한국어 한 조각", "현 선생님", "유튜브 자막"]}},
    "yt_ko_interview_a": {"dir": "data_yt", "hints": ["ko"], "translation": {"type": "one_way", "target_language": "en"}, "context": {
        "general": [{"key": "topic", "value": "지인 인터뷰, 반말, 이상형과 소개팅 이야기"}],
        "terms": ["한국어 한 조각", "한재민", "갓생", "이상형", "소개팅", "눈매"]}},
    "yt_ko_interview_b": {"dir": "data_yt", "hints": ["ko"], "translation": {"type": "one_way", "target_language": "en"}, "context": {
        "general": [{"key": "topic", "value": "헝가리 유학생 인터뷰"}],
        "terms": ["한국어 한 조각", "하티제", "헝가리", "부다페스트", "국제관계학", "전 선생님", "터키"]}},
}


def count(hay, needle):
    hay, needle = norm(hay), norm(needle)
    return hay.count(needle) if needle else 0


async def run_case(name, c, url, api_key, model, dump=None, skip_context=False):
    ref = json.load(open(os.path.join(HERE, c["dir"], f"{name}.json")))
    wav = os.path.join(HERE, c["dir"], f"{name}.wav")
    ctx = dict(c["context"])
    # realistic context from the video page only (title / channel / description), never from the transcript
    if ctx.pop("use_title", False):
        ctx["general"] = [{"key": "title", "value": ref.get("title") or ""}, {"key": "channel", "value": ref.get("channel") or ""}]
    if ctx.pop("use_description", False):
        ctx["text"] = (ref.get("description") or "")[:600]
    c = {**c, "context": ctx}
    base = {"language_hints": c["hints"], "translation": c["translation"]}
    rows = {}
    variants = [("no-context", base)] + ([] if skip_context else [("context", {**base, "context": c["context"]})])
    for label, extra in variants:
        ev = await stream(wav, url, api_key, model, extra)
        if dump:
            os.makedirs(dump, exist_ok=True)
            json.dump(ev, open(os.path.join(dump, f"{name}.{label}.json"), "w"), ensure_ascii=False)
        m = score(ev, ref)
        final = "".join(t["text"] for _, t in ev if t["is_final"] and "start_ms" in t and not t["text"].startswith("<"))
        tr = "".join(t["text"] for _, t in ev if t.get("translation_status") == "translation" and t["is_final"])
        reftext = " ".join(u["text"] for u in ref["utterances"])
        want = sum(count(reftext, t) for t in c["context"]["terms"])
        got = sum(min(count(final, t), count(reftext, t)) for t in c["context"]["terms"])
        gl = c["context"].get("translation_terms") or []
        gl_hits = sum(count(tr, g["target"]) > 0 for g in gl)
        err = m.get("cer_ko", m.get("wer_en"))
        rows[label] = {"err": err, "term_recall": got / want if want else float("nan"), "glossary": f"{gl_hits}/{len(gl)}" if gl else "-",
                       "chrf": m.get("chrf_to_en"), "final": final}
    return name, rows


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765/transcribe-websocket")
    ap.add_argument("--api-key")
    ap.add_argument("--model", default="rtstt-v1")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--show", action="store_true", help="print the final transcripts")
    ap.add_argument("--only-terms", action="store_true", help="context with terms/translation_terms only (no general/text)")
    ap.add_argument("--dump", help="save raw token events per case/variant")
    ap.add_argument("--cases-file", help="JSON with extra cases (same structure as CASES)")
    ap.add_argument("--skip-context", action="store_true")
    a = ap.parse_args()
    if a.cases_file:
        CASES.update(json.load(open(a.cases_file)))
        if a.cases == ",".join(k for k in CASES if k not in json.load(open(a.cases_file))):
            a.cases = ",".join(json.load(open(a.cases_file)))
    if a.only_terms:
        for c in CASES.values():
            c["context"] = {k: v for k, v in c["context"].items() if k in ("terms", "translation_terms")}
    results = await asyncio.gather(*[run_case(n, CASES[n], a.url, a.api_key, a.model, a.dump, a.skip_context)
                                     for n in a.cases.split(",")])
    print(f"{'case':20s} {'err no-ctx':>10s} {'err ctx':>8s} {'terms no-ctx':>12s} {'terms ctx':>9s} {'glossary':>8s} {'chrF no/ctx':>12s}")
    for name, r in results:
        n, c = r["no-context"], r.get("context", r["no-context"])
        chrf = f"{n['chrf']:.1f}/{c['chrf']:.1f}" if n.get("chrf") is not None and c.get("chrf") is not None else "-"
        print(f"{name:20s} {100*n['err']:9.1f}% {100*c['err']:7.1f}% {100*n['term_recall']:11.0f}% {100*c['term_recall']:8.0f}% {c['glossary']:>8s} {chrf:>12s}")
        if a.show:
            print("   no-ctx:", n["final"][:300]); print("   ctx   :", c["final"][:300])


if __name__ == "__main__":
    asyncio.run(main())
