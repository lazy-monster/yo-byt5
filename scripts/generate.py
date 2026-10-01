"""Run the fine-tuned systems over the YAD test set and save their outputs.

    python scripts/generate.py --yad path/to/YAD
    python scripts/generate.py --yad path/to/YAD --systems yo-byt5 mt5-base

Each system is decoded greedily and with beam search (5 beams). Byte-level
models use a 1,024-token limit, subword models 512.
"""

import argparse
from pathlib import Path

from yobyt5.data import SYSTEMS, YAD_TEST, load_yad, output_path, write_outputs
from yobyt5.generation import generate, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yad", required=True, help="clone of github.com/ajesujoba/YAD")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=list(SYSTEMS))
    args = ap.parse_args()

    src, _ = load_yad(Path(args.yad) / YAD_TEST)
    for name in args.systems:
        _, repo_id, revision = SYSTEMS[name]
        byte_level = "byt5" in repo_id.lower()
        tok, model, device = load_model(repo_id, revision)
        for decoding, beams in (("greedy", 1), ("beam5", 5)):
            hyps = generate(tok, model, device, src, num_beams=beams,
                            max_length=1024 if byte_level else 512,
                            batch_size=8 if byte_level else 16)
            path = output_path(name, decoding)
            write_outputs(path, hyps)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
