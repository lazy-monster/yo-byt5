"""Deterministic generation with a Hugging Face sequence-to-sequence model."""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(repo_id, revision=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, revision=revision)
    model.to(device).eval()
    return tok, model, device


def generate(tok, model, device, sources, num_beams=1, max_length=512, batch_size=16):
    """Restore diacritics for `sources`. Inputs and outputs are cut at `max_length` tokens."""
    kwargs = dict(num_beams=num_beams, max_length=max_length,
                  early_stopping=num_beams > 1, do_sample=False)
    outs = []
    for i in range(0, len(sources), batch_size):
        enc = tok(sources[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, **kwargs)
        outs.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return outs
