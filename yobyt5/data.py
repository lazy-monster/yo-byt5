"""YAD test data, the evaluated systems, and saved system outputs."""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"
RESULTS = REPO / "results"

# Relative to a clone of https://github.com/ajesujoba/YAD
YAD_TEST = "data/json/test/no_diac/yad_test.json"

# name: (display name, Hugging Face repository, commit evaluated in the paper)
SYSTEMS = {
    "yo-byt5": ("Yo-ByT5", "lazymonster/yobyt5-restoration",
                "48504c80a3dff2d66a6dde83a271e38fa984f11f"),
    "mt5-base": ("mT5-base", "Davlan/mT5_base_yoruba_adr",
                 "9a6c4a7ba523ea0c6f6e5acc67592b454204e384"),
    "omowe-t5-all-und": ("omowe-T5 all-und", "Davlan/omowe-t5-small-diacritizer-all-und-full",
                         "89f8bffc723d92007297227ec34c81f94aa55c9f"),
    "omowe-t5-menyo": ("omowe-T5 menyo", "Davlan/omowe-t5-small-diacritizer-menyo",
                       "5f3b947bb5089de74fa57c1bf3f516f0e67c86d7"),
    "byt5-small-menyo": ("ByT5-small menyo", "Davlan/byt5-small-diacritizer-menyo",
                         "6ecaa2d0e0e9b75b1246186567f5c3a3e9d25c96"),
    "mt5-small-menyo": ("mT5-small menyo", "Davlan/mt5-small-diacritizer-menyo",
                        "1f501cb9990c6fddb1d5af9594300929a8ad075d"),
}
LLM = ("gpt-oss-120b", "gpt-oss:120b")


def load_yad(path):
    """Return (sources, references) from a YAD JSON Lines file."""
    src, ref = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)["translation"]
            src.append(t["unyo"])
            ref.append(t["dcyo"])
    return src, ref


def output_path(system, decoding):
    """`decoding` is 'greedy', 'beam5', or 'chat' for the language model."""
    return OUTPUTS / f"{system}.{decoding}.jsonl"


def write_outputs(path, hyps):
    """Write one {"id", "hyp"} object per line; `id` is the YAD test line index."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for i, h in enumerate(hyps):
            fh.write(json.dumps({"id": i, "hyp": h}, ensure_ascii=False) + "\n")


def read_outputs(path, n):
    """Read outputs written by `write_outputs` and check they cover all `n` lines."""
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r["hyp"]
    if sorted(rows) != list(range(n)):
        raise ValueError(f"{path} covers {len(rows)} of {n} test sentences")
    return [rows[i] for i in range(n)]
