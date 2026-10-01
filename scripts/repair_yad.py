"""Correct the underdot encoding of a YAD reference file.

    python scripts/repair_yad.py path/to/YAD/data/json/test/no_diac/yad_test.json yad_test.corrected.json

Some YAD references encode the Yorùbá underdot as U+0329 COMBINING VERTICAL
LINE BELOW instead of U+0323 COMBINING DOT BELOW, and some letters carry two
underdots. No Unicode normalisation form maps one codepoint to the other, so
a system that writes the standard underdot is scored wrong there. This script
maps U+0329 to U+0323, collapses doubled underdots, and writes the reference
in NFC. Inputs (`unyo`) are left unchanged.

The scoring code applies the same correction on the fly, so scoring does not
need this file. It is provided to inspect or reuse the corrected references.
"""

import argparse
import json
import re
import unicodedata

from yobyt5.metrics import DOT_BELOW, VERTICAL_LINE_BELOW, repair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("target")
    args = ap.parse_args()

    vertical = doubled = dots = changed = lines = 0
    out = []
    with open(args.source, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            ref = row["translation"]["dcyo"]
            nfd = unicodedata.normalize("NFD", ref)
            vertical += nfd.count(VERTICAL_LINE_BELOW)
            doubled += len(re.findall(DOT_BELOW + "{2,}", nfd.replace(VERTICAL_LINE_BELOW, DOT_BELOW)))
            fixed = unicodedata.normalize("NFC", repair(ref))
            dots += unicodedata.normalize("NFD", fixed).count(DOT_BELOW)
            if fixed != unicodedata.normalize("NFC", ref):
                changed += 1
                row["translation"]["dcyo"] = fixed
            out.append(json.dumps(row, ensure_ascii=False))
            lines += 1

    with open(args.target, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"{lines} lines, {changed} changed")
    print(f"U+0329 underdots replaced: {vertical} ({vertical / dots * 100:.2f}% of {dots} underdots)")
    print(f"doubled underdots collapsed: {doubled}")


if __name__ == "__main__":
    main()
