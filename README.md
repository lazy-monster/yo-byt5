# Yo-ByT5

Code and system outputs for "Yo-ByT5: Efficient and High-Fidelity Diacritic Restoration for Yorùbá".

Model: https://huggingface.co/lazymonster/yobyt5-restoration

## Setup

```bash
pip install -e .
git clone https://github.com/ajesujoba/YAD.git
```

## Scoring

```bash
python scripts/score.py --yad YAD
python scripts/bootstrap.py --yad YAD
python scripts/make_figures.py --yad YAD
```

## Generating outputs

Beam search needs transformers below version 5.

```bash
pip install -e ".[generate]"
python scripts/generate.py --yad YAD
OLLAMA_API_KEY=... python scripts/generate_llm.py --yad YAD
```

## Training

```bash
pip install -e ".[train]"
python training/train.py --phase 1 --data-dir data --output phase1
python training/train.py --phase 2 --data-dir data --init phase1 --output phase2
```

The released model is `phase2/checkpoint-2896`. The training data is not included.

## Outputs

Each line of `outputs/<system>.<decoding>.jsonl` is `{"id": ..., "hyp": ...}`. `id` is the line number in YAD's `data/json/test/no_diac/yad_test.json`, starting at 0.

## License

Apache 2.0
