"""List YAD dev and test sentences that also occur in a training file.

    python scripts/check_contamination.py --train train_diac_restore.csv --yad path/to/YAD

The training file is a CSV with a `target_text` column. Two sentences match
when they are equal after whitespace is collapsed, or equal after diacritics
are also removed and case is folded.
"""

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path

from yobyt5.data import YAD_TEST, load_yad
from yobyt5.metrics import base_of

YAD_DEV = "data/json/dev/dev.json"


def exact(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def unmarked(s):
    return base_of(exact(s)).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--yad", required=True, help="clone of github.com/ajesujoba/YAD")
    args = ap.parse_args()

    with open(args.train, encoding="utf-8") as fh:
        train = [row["target_text"] for row in csv.DictReader(fh)]
    train_exact = Counter(exact(s) for s in train)
    train_unmarked = Counter(unmarked(s) for s in train)
    print(f"{len(train)} training sentences")

    for split, rel in (("dev", YAD_DEV), ("test", YAD_TEST)):
        _, ref = load_yad(Path(args.yad) / rel)
        hits = Counter(exact(s) for s in ref if unmarked(s) in train_unmarked)
        print(f"\nYAD {split}: {sum(hits.values())} of {len(ref)} sentences occur in training")
        for s, n in hits.most_common():
            kind = "exact" if s in train_exact else "unmarked"
            print(f"  {n:3d} x {s!r} ({kind} match, {train_unmarked[unmarked(s)]} x in training)")


if __name__ == "__main__":
    main()
