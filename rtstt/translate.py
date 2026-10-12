"""Streaming translation via any OpenAI-compatible chat endpoint (vLLM, llama.cpp server, ...).

style="chat": general instruction LLM (system prompt + few-shot history).
style="hymt": Tencent Hy-MT2 translation models (official prompt templates, no system prompt).
"""
import json

import httpx

LANG_NAMES = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese", "es": "Spanish",
              "fr": "French", "de": "German", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian"}


class Translator:
    def __init__(self, url, model, timeout=30.0, style="chat"):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model
        self.style = style
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=2.0))   # fail fast when the server is down

    @staticmethod
    def _hymt_messages(text, tgt, history, glossary, general):
        """Official Hy-MT2 prompts. Context = previous sentences as earlier turns (the
        "background information" template makes the model translate the background too)."""
        lang = LANG_NAMES.get(tgt, tgt)
        terms = [t for t in glossary or [] if "source" in t and "target" in t]

        def prompt(src):
            if terms:
                return ("Reference the following translations:\n"
                        + "\n".join(f"{t['source']} translates to {t['target']}" for t in terms)
                        + f"\n\nTranslate the following text into {lang}. Note that you must ONLY output the "
                        f"translated result without any additional explanation:\n\n{src}")
            return (f"Translate the following text into {lang}. Note that you should only output the "
                    f"translated result without any additional explanation:\n\n{src}")
        msgs = []
        for h_src, h_tgt in history:
            msgs += [{"role": "user", "content": prompt(h_src)}, {"role": "assistant", "content": h_tgt}]
        return msgs + [{"role": "user", "content": prompt(text)}]

    async def stream(self, text, src, tgt, history=(), glossary=None, general=None):
        """Yield translated text chunks."""
        if self.style == "hymt":
            msgs = self._hymt_messages(text, tgt, history, glossary, general)
            body = {"model": self.model, "messages": msgs, "temperature": 0, "max_tokens": 1024, "stream": True,
                    "top_k": 20, "top_p": 0.6, "repeat_penalty": 1.05}
        else:
            msgs = self._chat_messages(text, src, tgt, history, glossary, general)
            body = {"model": self.model, "messages": msgs, "temperature": 0, "max_tokens": 1024, "stream": True,
                    "chat_template_kwargs": {"enable_thinking": False}}
        async with self.client.stream("POST", self.url, json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if delta:
                    yield delta

    @staticmethod
    def _chat_messages(text, src, tgt, history, glossary, general):
        sys = (f"You are a professional real-time interpreter. Translate the {LANG_NAMES.get(src, src)} "
               f"speech transcript into natural {LANG_NAMES.get(tgt, tgt)}. Output only the translation, "
               "no notes. Keep proper nouns, numbers and product names accurate. "
               "The transcript comes from speech recognition and may contain misrecognized words; "
               "use context to infer the intended meaning.")
        if general:
            sys += f"\nContext: {general}"
        if glossary:
            sys += "\nGlossary (source -> target): " + "; ".join(
                f"{t['source']} -> {t['target']}" for t in glossary if "source" in t and "target" in t)
        msgs = [{"role": "system", "content": sys}]
        for h_src, h_tgt in history:
            msgs += [{"role": "user", "content": h_src}, {"role": "assistant", "content": h_tgt}]
        msgs.append({"role": "user", "content": text})
        return msgs
