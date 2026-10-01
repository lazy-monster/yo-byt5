"""Fine-tune google/byt5-small into Yo-ByT5 on a TPU v6e-8.

    python training/train.py --phase 1 --data-dir data --output phase1
    python training/train.py --phase 2 --data-dir data --init phase1 --output phase2

`data-dir` holds train_diac_restore.csv and validation_diac_restore.csv, each
with `input_text` (unmarked) and `target_text` (marked) columns. Each of the
8 TPU cores runs a batch of 4 with 2 accumulation steps, a global batch of 64
and 724 steps per epoch on the 46,313 training sentences.

Phase 1 trains from google/byt5-small and saves the checkpoint with the lowest
validation loss to --output. Phase 2 continues from that checkpoint at a lower
learning rate. The released model is the last phase 2 checkpoint
(--output/checkpoint-2896).
"""

import argparse
import os

os.environ["PJRT_DEVICE"] = "TPU"

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from datasets import load_dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

MAX_LENGTH = 1024
PHASES = {
    1: dict(learning_rate=2e-4, warmup_steps=300, eval_steps=500, save_steps=500),
    2: dict(learning_rate=1e-4, warmup_steps=0, eval_steps=200, save_steps=200),
}


def tokenize(tokenizer, batch):
    inputs = tokenizer(batch["input_text"], max_length=MAX_LENGTH,
                       padding="max_length", truncation=True)
    labels = tokenizer(batch["target_text"], max_length=MAX_LENGTH,
                       padding="max_length", truncation=True)
    inputs["labels"] = [[t if t != tokenizer.pad_token_id else -100 for t in seq]
                        for seq in labels["input_ids"]]
    return inputs


def main(args):
    init = "google/byt5-small" if args.phase == 1 else args.init
    tokenizer = AutoTokenizer.from_pretrained(init)

    data = load_dataset("csv", data_files={
        "train": os.path.join(args.data_dir, "train_diac_restore.csv"),
        "validation": os.path.join(args.data_dir, "validation_diac_restore.csv"),
    })
    columns = data["train"].column_names

    # The first process tokenises and writes the cache; the others read it.
    if xm.is_master_ordinal():
        data = data.map(lambda b: tokenize(tokenizer, b), batched=True, num_proc=16,
                        remove_columns=columns, load_from_cache_file=False)
    xm.rendezvous("tokenized")
    if not xm.is_master_ordinal():
        data = data.map(lambda b: tokenize(tokenizer, b), batched=True,
                        remove_columns=columns)

    model = AutoModelForSeq2SeqLM.from_pretrained(init).to(xm.xla_device())

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=4,
        weight_decay=0.01,
        max_grad_norm=0.5,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        **PHASES[args.phase],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
    )
    trainer.train()

    if args.phase == 1 and xm.is_master_ordinal():
        trainer.save_model(args.output)
        tokenizer.save_pretrained(args.output)
    xm.rendezvous("done")


def _mp_fn(index, args):
    main(args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=[1, 2], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--init", help="phase 1 output directory (phase 2 only)")
    args = ap.parse_args()
    if args.phase == 2 and not args.init:
        ap.error("--init is required for phase 2")
    xmp.spawn(_mp_fn, args=(args,), start_method="spawn")
