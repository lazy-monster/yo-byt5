"""Run gpt-oss:120b on the YAD test set through the Ollama chat API.

    export OLLAMA_API_KEY=...        # not needed for a local server
    python scripts/generate_llm.py --yad path/to/YAD
    python scripts/generate_llm.py --yad path/to/YAD --host http://127.0.0.1:11434

The prompt holds one worked example. Decoding uses temperature 0, seed 42 and
low reasoning effort. Finished items are appended to the output file as they
arrive, so an interrupted run resumes where it stopped. Items that fail are
left out; run the script again until it reports none missing.
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

from yobyt5.data import LLM, YAD_TEST, load_yad, output_path

SYSTEM = ("You restore Yoruba diacritics (tone marks and underdots). "
          "Return ONLY the fully diacritized sentence, nothing else.")
EXAMPLE_IN = "Eko ni kokoro aseyori."
EXAMPLE_OUT = "Ẹ̀kọ́ ni kọ́kọ́rọ́ àṣeyọrí."
OPTIONS = {"temperature": 0.0, "seed": 42, "num_predict": 1024}
THINK = "low"


def prompt(text):
    return f"{SYSTEM}\n\nInput: {EXAMPLE_IN}\nOutput: {EXAMPLE_OUT}\n\nInput: {text}\nOutput:"


def first_line(reply):
    """Return the sentence from a reply, without code fences or an 'Output:' label."""
    t = (reply or "").strip()
    if t.startswith("```"):
        t = "\n".join(l for l in t.splitlines() if not l.startswith("```")).strip()
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    while len(lines) > 1 and lines[0].endswith(":"):
        lines.pop(0)
    if not lines:
        return ""
    out = lines[0]
    if out.lower().startswith("output:"):
        out = out.split(":", 1)[1].strip()
    return out.strip()


def restore(host, model, key, text, timeout=60):
    payload = {"model": model, "stream": False, "think": THINK, "options": OPTIONS,
               "messages": [{"role": "user", "content": prompt(text)}]}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(host.rstrip("/") + "/api/chat",
                                 json.dumps(payload).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    if resp.get("done_reason") == "length":
        raise RuntimeError(f"reply cut at num_predict={OPTIONS['num_predict']}")
    return first_line(resp.get("message", {}).get("content", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yad", required=True, help="clone of github.com/ajesujoba/YAD")
    ap.add_argument("--host", default="https://ollama.com")
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    src, _ = load_yad(Path(args.yad) / YAD_TEST)
    path = output_path(LLM[0], "chat")
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}
    key = os.environ.get("OLLAMA_API_KEY")

    missing = 0
    with open(path, "a", encoding="utf-8") as fh:
        for i, s in enumerate(src):
            if i in done:
                continue
            hyp = ""
            for attempt in range(args.retries):
                try:
                    hyp = restore(args.host, LLM[1], key, s)
                    break
                except Exception as e:
                    print(f"item {i}, attempt {attempt + 1}: {type(e).__name__}: {e}")
                    time.sleep(2 * (attempt + 1))
            if not hyp:
                missing += 1
                continue
            fh.write(json.dumps({"id": i, "hyp": hyp}, ensure_ascii=False) + "\n")
            fh.flush()
    print(f"{len(src) - missing} of {len(src)} items in {path}; {missing} missing")


if __name__ == "__main__":
    main()
