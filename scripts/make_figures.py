"""Draw Figures 2 and 3 of the paper from the saved outputs.

    python scripts/score.py --yad path/to/YAD      # writes results/scores.json
    python scripts/make_figures.py --yad path/to/YAD

Figure 2: error typology of each system (greedy decoding, corrected test set).
Figure 3: share of gpt-oss:120b outputs that keep the input's base text, by
input length.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from yobyt5.data import LLM, REPO, RESULTS, SYSTEMS, YAD_TEST, load_yad, output_path, read_outputs
from yobyt5.metrics import base_of

FIGURES = REPO / "figures"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 300,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
})

FIGURE2_ORDER = ["yo-byt5", "mt5-base", LLM[0], "omowe-t5-all-und",
                 "byt5-small-menyo", "omowe-t5-menyo", "mt5-small-menyo"]
CLASSES = [("tone_only", "Tone only", "#08519c"),
           ("underdot_only", "Underdot only", "#4292c6"),
           ("mixed", "Mixed marks", "#9ecae1"),
           ("lexical", "Lexical substitution", "#e6550d"),
           ("deletion", "Deletion", "#fd8d3c"),
           ("insertion", "Insertion", "#fdd0a2")]
LENGTH_BANDS = ["< 60", "60–119", "120–179", "180–239", "≥ 240"]


def save(fig, stem):
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / f"{stem}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure2(scores):
    runs = [f"{n}.chat" if n == LLM[0] else f"{n}.greedy" for n in FIGURE2_ORDER]
    labels = [LLM[1] if n == LLM[0] else SYSTEMS[n][0] for n in FIGURE2_ORDER]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    y = np.arange(len(runs))[::-1]
    left = np.zeros(len(runs))
    for key, name, colour in CLASSES:
        part = np.array([scores[r]["typology"].get(key, 0.0) for r in runs])
        ax.barh(y, part, height=0.62, left=left, color=colour, label=name,
                edgecolor="white", linewidth=0.5)
        left += part
    for yi, r in zip(y, runs):
        ax.text(101.5, yi, f"{scores[r]['text_altering_rate'] * 100:.2f}% text-altering",
                va="center", ha="left", fontsize=8.8, color="#7a3803")

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of wrong words (%)")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, columnspacing=1.2, handlelength=1.4)
    save(fig, "figure2-error-typology")


def figure3(src):
    hyps = read_outputs(output_path(LLM[0], "chat"), len(src))
    kept, count = np.zeros(5), np.zeros(5)
    for s, h in zip(src, hyps):
        band = min(len(s) // 60, 4)
        count[band] += 1
        kept[band] += base_of(h) == base_of(s)
    share = kept / count * 100

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(LENGTH_BANDS, share, marker="o", markersize=7, linewidth=2.2, color="#c0392b")
    for x, v in zip(LENGTH_BANDS, share):
        ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9.5, color="#7a1f16")
    ax.set_xlabel("Input sentence length (characters)")
    ax.set_ylabel("Outputs preserving the base text (%)")
    ax.set_ylim(0, max(share) * 1.15)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    save(fig, "figure3-copy-fidelity-by-length")
    print("base text kept by length band:",
          ", ".join(f"{b}: {v:.1f}% of {int(n)}" for b, v, n in zip(LENGTH_BANDS, share, count)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yad", required=True, help="clone of github.com/ajesujoba/YAD")
    args = ap.parse_args()

    src, _ = load_yad(Path(args.yad) / YAD_TEST)
    with open(RESULTS / "scores.json", encoding="utf-8") as fh:
        figure2(json.load(fh))
    figure3(src)


if __name__ == "__main__":
    main()
