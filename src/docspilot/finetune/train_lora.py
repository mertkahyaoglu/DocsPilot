"""LoRA fine-tune of a small encoder for intent classification (CPU-friendly).

LoRA trains small low-rank adapters (~1% of parameters) instead of the whole network:
cheap, fast, and the adapter is a few MB. For a 5-class router this is more machinery
than needed (see the TF-IDF baseline); it is here to show the mechanics and to measure
whether the extra cost buys anything."""

from __future__ import annotations

import json
import random
from pathlib import Path

from docspilot.finetune.dataset import Example
from docspilot.intents import INTENTS


def train(
    train: list[Example],
    out_dir: Path,
    *,
    base_model: str = "distilbert-base-uncased",
    epochs: int = 4,
    lr: float = 3e-4,
    batch_size: int = 16,
    rank: int = 8,
    seed: int = 3,
) -> dict[str, float]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    label2id = {lab: i for i, lab in enumerate(INTENTS)}
    id2label = {i: lab for lab, i in label2id.items()}

    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(INTENTS), id2label=id2label, label2id=label2id
    )
    peft_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],  # distilbert attention projections
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    data = list(train)
    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        random.shuffle(data)
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            enc = tok(
                [e.text for e in batch],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            labels = torch.tensor([label2id[e.label] for e in batch])
            out = model(**enc, labels=labels)
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            last_loss = float(out.loss)
        print(f"epoch {epoch + 1}/{epochs} loss={last_loss:.4f}")

    merged = model.merge_and_unload()  # fold adapters in: a plain checkpoint, no peft at runtime
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "meta.json").write_text(
        json.dumps({"kind": "transformers", "labels": list(INTENTS)})
    )
    return {"trainable_params": trainable, "total_params": total, "final_loss": last_loss}
