"""L1: fine-tuned Twitter RoBERTa that scores an account's English text.

A plain PyTorch loop (fp16 autocast, linear warm-up/decay, best-epoch selection on
a held-out validation slice) keeps the training transparent and reproducible.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "cardiffnlp/twitter-roberta-base"
    fallback_model: str = "distilroberta-base"
    max_length: int = 256
    batch_size: int = 16
    eval_batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    warmup_ratio: float = 0.06
    val_fraction: float = 0.10
    fp16: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pretrained(cfg: EncoderConfig):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for name in (cfg.model_name, cfg.fallback_model):
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=2)
            return tok, model, name
        except OSError:
            continue
    raise RuntimeError("could not load the encoder or its fallback")


class _Collate:
    def __init__(self, tok, max_length: int):
        self.tok, self.max_length = tok, max_length

    def __call__(self, batch):
        texts = [b[0] for b in batch]
        enc = self.tok(texts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        if batch[0][1] is not None:
            enc["labels"] = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return enc


@torch.no_grad()
def predict_proba(model, tok, texts: list[str], cfg: EncoderConfig, device: str = "cuda") -> np.ndarray:
    model.eval().to(device)
    dl = DataLoader([(t, None) for t in texts], batch_size=cfg.eval_batch_size, shuffle=False,
                    collate_fn=_Collate(tok, cfg.max_length))
    out = []
    for enc in dl:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.fp16 and device == "cuda"):
            logits = model(**enc).logits
        out.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_encoder(texts: list[str], labels: np.ndarray, cfg: EncoderConfig, log=print, device: str = "cuda"):
    """Fine-tune on (texts, labels); a stratified `val_fraction` slice picks the best epoch.

    Returns (model, tokenizer, info) with the best epoch's weights loaded.
    """
    from sklearn.model_selection import train_test_split
    from transformers import get_linear_schedule_with_warmup

    set_seed(cfg.seed)
    labels = np.asarray(labels)
    idx = np.arange(len(texts))
    tr_idx, va_idx = train_test_split(idx, test_size=cfg.val_fraction, stratify=labels, random_state=cfg.seed)
    tok, model, used_name = load_pretrained(cfg)
    model.to(device)

    g = torch.Generator().manual_seed(cfg.seed)
    train_dl = DataLoader([(texts[i], int(labels[i])) for i in tr_idx], batch_size=cfg.batch_size, shuffle=True,
                          generator=g, collate_fn=_Collate(tok, cfg.max_length))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    total = len(train_dl) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, math.ceil(cfg.warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)

    best = {"val_loss": float("inf")}
    best_state = None
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for step, enc in enumerate(train_dl, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.fp16):
                loss = model(**enc).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val_prob = predict_proba(model, tok, [texts[i] for i in va_idx], cfg, device)
        y_va = labels[va_idx]
        vl = log_loss(y_va, np.clip(val_prob, 1e-6, 1 - 1e-6), labels=[0, 1])
        vf1 = f1_score(y_va, (val_prob >= 0.5).astype(int), average="macro")
        history.append({"epoch": epoch, "train_loss": running / len(train_dl), "val_loss": vl, "val_f1_macro": vf1})
        log(f"  epoch {epoch}: train_loss={running / len(train_dl):.4f} val_loss={vl:.4f} val_F1m={vf1:.4f}")
        if vl < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": vl, "val_f1_macro": vf1}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    info = {"model_name": used_name, "best": best, "history": history,
            "train_seconds": time.perf_counter() - t0, "n_train": len(tr_idx), "n_val": len(va_idx)}
    return model, tok, info


def oof_plan(folds: dict[str, int]) -> dict[int, dict[str, list[str]]]:
    """For each fold k: which accounts train the fold model and which it scores.

    The scored accounts (fold k) never appear in that model's training set, so every
    training account's out-of-fold score comes from a model that never saw it.
    """
    plan = {}
    for k in sorted(set(folds.values())):
        plan[k] = {
            "train": [u for u, f in folds.items() if f != k],
            "score": [u for u, f in folds.items() if f == k],
        }
    return plan


def save_encoder(model, tok, path: Path, info: dict) -> None:
    import json

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "spamsentinel_info.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
