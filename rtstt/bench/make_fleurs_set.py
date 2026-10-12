"""Multilingual benchmark set from Google FLEURS (read speech, n-way parallel sentences).

For each supported language available in FLEURS, take N test utterances with distinct sentence
ids that also exist in English and Korean, so every clip has an English and a Korean reference
translation. Only the first part of each language's test.tar.gz is streamed (a few MB).

  python make_fleurs_set.py [--n 20] [--out bench/data_fleurs]
"""
import argparse
import csv
import io
import json
import os
import sys
import tarfile

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
# language code -> FLEURS folder (sq Albanian and eu Basque are not in FLEURS)
FLEURS = {
    "af": "af_za", "ar": "ar_eg", "az": "az_az", "be": "be_by", "bn": "bn_in", "bs": "bs_ba", "bg": "bg_bg",
    "ca": "ca_es", "zh": "cmn_hans_cn", "hr": "hr_hr", "cs": "cs_cz", "da": "da_dk", "nl": "nl_nl",
    "en": "en_us", "et": "et_ee", "fi": "fi_fi", "fr": "fr_fr", "gl": "gl_es", "de": "de_de", "el": "el_gr",
    "gu": "gu_in", "he": "he_il", "hi": "hi_in", "hu": "hu_hu", "id": "id_id", "it": "it_it", "ja": "ja_jp",
    "kn": "kn_in", "kk": "kk_kz", "ko": "ko_kr", "lv": "lv_lv", "lt": "lt_lt", "mk": "mk_mk", "ms": "ms_my",
    "ml": "ml_in", "mr": "mr_in", "no": "nb_no", "fa": "fa_ir", "pl": "pl_pl", "pt": "pt_br", "pa": "pa_in",
    "ro": "ro_ro", "ru": "ru_ru", "sr": "sr_rs", "sk": "sk_sk", "sl": "sl_si", "es": "es_419", "sw": "sw_ke",
    "sv": "sv_se", "tl": "fil_ph", "ta": "ta_in", "te": "te_in", "th": "th_th", "tr": "tr_tr", "uk": "uk_ua",
    "ur": "ur_pk", "vi": "vi_vn", "cy": "cy_gb",
}
csv.field_size_limit(sys.maxsize)


def tsv(folder, split):
    r = requests.get(f"{BASE}/{folder}/{split}.tsv", timeout=60)
    r.raise_for_status()
    rows = csv.reader(io.StringIO(r.text), delimiter="\t", quoting=csv.QUOTE_NONE)
    return [{"id": x[0], "file": x[1], "text": x[2], "norm": x[3]} for x in rows if len(x) >= 4]


def refs(folder):
    out = {}
    for split in ("train", "dev", "test"):
        for x in tsv(folder, split):
            out.setdefault(x["id"], x["text"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(HERE, "data_fleurs"))
    ap.add_argument("--langs", default=",".join(FLEURS))
    a = ap.parse_args()
    en, ko = refs(FLEURS["en"]), refs(FLEURS["ko"])
    print(f"reference sentences: en {len(en)}, ko {len(ko)}", flush=True)
    for lang in a.langs.split(","):
        folder = FLEURS[lang]
        d = os.path.join(a.out, lang)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        os.makedirs(d, exist_ok=True)
        rows = {x["file"]: x for x in tsv(folder, "test")}
        picked, seen = [], set()
        with requests.get(f"{BASE}/{folder}/audio/test.tar.gz", stream=True, timeout=60) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                for m in tar:
                    if not m.isfile():
                        continue
                    x = rows.get(os.path.basename(m.name))
                    if not x or x["id"] in seen or x["id"] not in en or x["id"] not in ko:
                        continue
                    name = f"{len(picked):02d}.wav"
                    with open(os.path.join(d, name), "wb") as f:
                        f.write(tar.extractfile(m).read())
                    seen.add(x["id"])
                    picked.append({"file": name, "id": x["id"], "text": x["text"], "norm": x["norm"],
                                   "en": en[x["id"]], "ko": ko[x["id"]]})
                    if len(picked) >= a.n:
                        break
        json.dump({"lang": lang, "fleurs": folder, "utts": picked}, open(os.path.join(d, "meta.json"), "w"),
                  ensure_ascii=False, indent=1)
        print(f"{lang} ({folder}): {len(picked)} clips", flush=True)


if __name__ == "__main__":
    main()
