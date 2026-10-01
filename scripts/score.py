"""Score every saved output against the corrected YAD test set.

    python scripts/score.py --yad path/to/YAD

Writes results/scores.json and prints the paper's main table (beam search,
5 beams) and the greedy CER of each system. The error typology in
scores.json comes from the greedy outputs, as in the paper.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from yobyt5 import metrics
from yobyt5.data import LLM, RESULTS, SYSTEMS, YAD_TEST, load_yad, output_path, read_outputs

COLUMNS = ["CER", "WER", "DER", "DER_tone", "DER_underdot", "WDER", "BLEU", "ChrF"]


def score_file(args):
    ref, path = args
    hyps = read_outputs(path, len(ref))
    scores = metrics.evaluate(ref, hyps)
    scores.update(metrics.bleu_chrf(ref, hyps))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yad", required=True, help="clone of github.com/ajesujoba/YAD")
    args = ap.parse_args()

    metrics.selftest()
    _, ref = load_yad(Path(args.yad) / YAD_TEST)

    runs = [(name, d) for name in SYSTEMS for d in ("beam5", "greedy")] + [(LLM[0], "chat")]
    with ProcessPoolExecutor() as pool:
        scored = pool.map(score_file, [(ref, output_path(n, d)) for n, d in runs])
        scores = {f"{n}.{d}": s for (n, d), s in zip(runs, scored)}

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "scores.json", "w", encoding="utf-8") as fh:
        json.dump(scores, fh, indent=2)

    rows = [(SYSTEMS[n][0], scores[f"{n}.beam5"]) for n in SYSTEMS]
    rows.append((LLM[1], scores[f"{LLM[0]}.chat"]))
    rows.sort(key=lambda r: r[1]["DER"])
    print("| System | " + " | ".join(COLUMNS) + " |")
    print("|---" * (len(COLUMNS) + 1) + "|")
    for label, s in rows:
        cells = [f"{s[c]:.4f}" if c in ("BLEU", "ChrF") else f"{s[c] * 100:.2f}" for c in COLUMNS]
        print(f"| {label} | " + " | ".join(cells) + " |")

    print("\n| System | CER greedy | CER beam5 |")
    print("|---|---|---|")
    for n in SYSTEMS:
        g, b = scores[f"{n}.greedy"]["CER"], scores[f"{n}.beam5"]["CER"]
        print(f"| {SYSTEMS[n][0]} | {g * 100:.2f} | {b * 100:.2f} |")

    classes = ["tone_only", "underdot_only", "mixed", *metrics.TEXT_ALTERING]
    print("\n| System | " + " | ".join(classes) + " | text-altering |")
    print("|---" * (len(classes) + 2) + "|")
    typology_runs = [(SYSTEMS[n][0], f"{n}.greedy") for n in SYSTEMS] + [(LLM[1], f"{LLM[0]}.chat")]
    for label, run in typology_runs:
        t = scores[run]["typology"]
        cells = [f"{t.get(c, 0.0):.2f}" for c in classes]
        print(f"| {label} | " + " | ".join(cells) + f" | {scores[run]['text_altering_rate'] * 100:.2f} |")


if __name__ == "__main__":
    main()
