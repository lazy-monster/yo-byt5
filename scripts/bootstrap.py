"""Paired bootstrap tests of the differences between Yo-ByT5 and mT5-base.

    python scripts/bootstrap.py --yad path/to/YAD

Resamples the 3,330 test sentences with replacement 10,000 times and
recomputes DER, CER, WER and WDER for both systems (beam search, corrected
test set) on each resample. Writes results/bootstrap.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from yobyt5 import metrics
from yobyt5.data import RESULTS, YAD_TEST, load_yad, output_path, read_outputs

RESAMPLES = 10_000
SEED = 0


def counts(ref, system):
    """Per-sentence (errors, total) for each metric."""
    hyps = read_outputs(output_path(system, "beam5"), len(ref))
    R = [metrics.prepare(x) for x in ref]
    H = [metrics.prepare(x) for x in hyps]
    aligned = metrics.align_all(R, H)
    table = {
        "DER": metrics.der_counts(R, H, aligned=aligned),
        "CER": metrics.cer_counts(R, H),
        "WER": metrics.wer_counts(R, H),
        "WDER": metrics.wder_counts(R, H, aligned=aligned),
    }
    return {k: np.array(v, dtype=np.int64) for k, v in table.items()}


def test(a, b, rng):
    """Bootstrap the rate difference a - b in percentage points."""
    assert (a[:, 1] == b[:, 1]).all(), "reference totals differ between systems"
    diff = np.empty(RESAMPLES)
    for start in range(0, RESAMPLES, 1000):
        idx = rng.integers(0, len(a), size=(min(1000, RESAMPLES - start), len(a)))
        diff[start:start + len(idx)] = (a[idx, 0].sum(1) - b[idx, 0].sum(1)) / a[idx, 1].sum(1) * 100
    total = a[:, 1].sum()
    low, high = np.percentile(diff, [2.5, 97.5])
    return {
        "yo_byt5": float(a[:, 0].sum() / total * 100),
        "mt5_base": float(b[:, 0].sum() / total * 100),
        "difference": float((a[:, 0].sum() - b[:, 0].sum()) / total * 100),
        "ci95": [float(low), float(high)],
        "share_yo_byt5_lower": float((diff < 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yad", required=True, help="clone of github.com/ajesujoba/YAD")
    args = ap.parse_args()

    _, ref = load_yad(Path(args.yad) / YAD_TEST)
    a, b = counts(ref, "yo-byt5"), counts(ref, "mt5-base")
    results = {"resamples": RESAMPLES, "seed": SEED}
    for metric in a:
        r = test(a[metric], b[metric], np.random.default_rng(SEED))
        results[metric] = r
        low, high = r["ci95"]
        print(f"{metric:5s} Yo-ByT5 {r['yo_byt5']:6.2f}  mT5-base {r['mt5_base']:6.2f}  "
              f"difference {r['difference']:+.2f}  95% CI [{low:+.2f}, {high:+.2f}]")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "bootstrap.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
