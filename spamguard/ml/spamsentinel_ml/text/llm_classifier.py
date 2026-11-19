"""L2: QLoRA fine-tuned small instruction LLM that labels an account spam / genuine.

The prompt shows profile fields, key behaviour numbers and sampled English tweets.
The model answers with one word; P(spam) is the softmax over the logits of the
first token of "spam" and "genuine" at the answer position, which gives a
probability usable for ROC-AUC and the review band.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

SYSTEM_PROMPT = (
    "You are a trust and safety analyst. Decide whether a social media account is a spam account "
    "(fake, automated or promotional spam) or a genuine account run by a real person. "
    "Answer with exactly one word: spam or genuine."
)

# Interpretable numbers shown to the LLM (a subset of the 64 engineered features).
PROMPT_NUMBERS = [
    ("followers_count", "Followers", "{:,.0f}"),
    ("followees_count", "Following", "{:,.0f}"),
    ("statuses_count", "Total tweets", "{:,.0f}"),
    ("account_age_days", "Account age (days)", "{:,.0f}"),
    ("tweets_per_day_lifetime", "Tweets per day", "{:.1f}"),
    ("frac_tweets_with_url", "Share of tweets with a link", "{:.0%}"),
    ("frac_retweets", "Share of retweets", "{:.0%}"),
    ("frac_replies", "Share of replies", "{:.0%}"),
    ("frac_third_party_client", "Share posted from third-party apps", "{:.0%}"),
    ("frac_duplicate_tweets", "Share of repeated tweets", "{:.0%}"),
    ("default_profile_image", "Default profile image", "yesno"),
]


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    max_length: int = 512
    max_tweets: int = 10
    max_tweet_chars: int = 200
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    warmup_ratio: float = 0.05
    train_limit: int | None = None  # subsample training accounts (None = all)
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["target_modules"] = list(self.target_modules)
        return d


def _fmt(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    if fmt == "yesno":
        return "yes" if value >= 0.5 else "no"
    return fmt.format(value)


def account_prompt(features: dict, description: str, tweets: list[str], cfg: LLMConfig) -> str:
    lines = ["Account profile:"]
    lines.append(f"- Bio: {description if description else '(none or not in English)'}")
    for key, label, fmt in PROMPT_NUMBERS:
        lines.append(f"- {label}: {_fmt(features.get(key), fmt)}")
    lines.append("Recent tweets (English only):")
    shown = [t[: cfg.max_tweet_chars] for t in tweets[: cfg.max_tweets]]
    if shown:
        lines.extend(f"{i}. {t}" for i, t in enumerate(shown, 1))
    else:
        lines.append("(no English tweets)")
    lines.append("Is this account spam or genuine?")
    return "\n".join(lines)


def chat_ids(tok, prompt: str, answer: str | None, max_length: int) -> tuple[list[int], int]:
    """Token ids for the chat; returns (ids, index where the answer starts)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"] if answer else []
    budget = max_length - len(a_ids)
    if len(p_ids) > budget:  # keep the end of the prompt (question + generation prompt) intact
        head = p_ids[: budget // 2]
        tail = p_ids[-(budget - len(head)):]
        p_ids = head + tail
    return p_ids + a_ids, len(p_ids)


def label_token_ids(tok) -> tuple[int, int]:
    spam = tok("spam", add_special_tokens=False)["input_ids"][0]
    genuine = tok("genuine", add_special_tokens=False)["input_ids"][0]
    if spam == genuine:
        raise ValueError("label words share their first token")
    return spam, genuine


def load_model(cfg: LLMConfig, adapter_path=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, quantization_config=bnb,
                                                 device_map={"": 0}, dtype=torch.bfloat16)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    return tok, model


@torch.no_grad()
def score(model, tok, prompts: list[str], cfg: LLMConfig, batch_size: int = 8) -> tuple[np.ndarray, float]:
    """P(spam) per prompt and mean inference milliseconds per account."""
    model.eval()
    spam_id, gen_id = label_token_ids(tok)
    probs = []
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        seqs = [chat_ids(tok, p, None, cfg.max_length)[0] for p in prompts[i: i + batch_size]]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, s in enumerate(seqs):  # left padding: the answer position is the last column
            ids[j, width - len(s):] = torch.tensor(s)
            mask[j, width - len(s):] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits[:, -1, :]
        pair = torch.stack([logits[:, spam_id], logits[:, gen_id]], dim=-1).float()
        probs.append(torch.softmax(pair, dim=-1)[:, 0].cpu().numpy())
    ms = (time.perf_counter() - t0) * 1000 / max(len(prompts), 1)
    return (np.concatenate(probs) if probs else np.zeros(0)), ms


def train_qlora(model, tok, prompts: list[str], labels: np.ndarray, cfg: LLMConfig, log=print,
                checkpoint_dir=None, checkpoint_every: int = 25):
    """QLoRA fine-tuning; with `checkpoint_dir` it saves adapters + optimizer/scheduler state
    every `checkpoint_every` updates and resumes from them, so a GPU reset (Windows TDR)
    only loses the work since the last checkpoint."""
    import hashlib
    import json
    from pathlib import Path

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"))
    examples = []
    for p, y in zip(prompts, labels):
        ids, start = chat_ids(tok, p, "spam" if y == 1 else "genuine", cfg.max_length)
        lab = [-100] * start + ids[start:]  # loss on the answer tokens only
        examples.append((ids, lab))
    rng = np.random.default_rng(cfg.seed)
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total_updates = math.ceil(steps_per_epoch * cfg.epochs / cfg.grad_accum)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(cfg.warmup_ratio * total_updates), total_updates)

    # A checkpoint is only reused for the identical run (same config and training data).
    run_key = hashlib.sha256(json.dumps([cfg.to_dict(), [len(e[0]) for e in examples], labels.tolist()])
                             .encode()).hexdigest()[:16]
    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    resume_step, prior_seconds = 0, 0.0
    if ckpt is not None and (ckpt / "trainer_state.pt").exists():
        state = torch.load(ckpt / "trainer_state.pt", weights_only=False)
        if state.get("run_key") == run_key:
            set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))
            opt.load_state_dict(state["opt"])
            sched.load_state_dict(state["sched"])
            resume_step, prior_seconds = state["step"], state["train_seconds"]
            log(f"  resuming from update {resume_step // cfg.grad_accum}/{total_updates}")

    model.train()
    t0, step, running = time.perf_counter(), 0, 0.0
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(examples))  # always drawn, so the order matches the original run
        for b in range(0, len(order), cfg.batch_size):
            step += 1
            if step <= resume_step:
                continue
            batch = [examples[i] for i in order[b: b + cfg.batch_size]]
            width = max(len(x[0]) for x in batch)
            ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(batch), width), -100, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, (x, y) in enumerate(batch):  # right padding for training
                ids[j, : len(x)] = torch.tensor(x)
                lab[j, : len(y)] = torch.tensor(y)
                mask[j, : len(x)] = 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=lab.cuda()).loss / cfg.grad_accum
            loss.backward()
            running += loss.item()
            if step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                update = step // cfg.grad_accum
                if update % 10 == 0:
                    log(f"  update {update}/{total_updates} loss={running / 10:.4f} "
                        f"{(prior_seconds + time.perf_counter() - t0) / 60:.1f} min")
                    running = 0.0
                if ckpt is not None and update % checkpoint_every == 0:
                    model.save_pretrained(ckpt)
                    torch.save({"run_key": run_key, "step": step, "opt": opt.state_dict(),
                                "sched": sched.state_dict(),
                                "train_seconds": prior_seconds + time.perf_counter() - t0},
                               ckpt / "trainer_state.pt")
    return model, {"train_seconds": prior_seconds + time.perf_counter() - t0, "updates": total_updates,
                   "n_train": len(examples), "resumed_from_update": resume_step // cfg.grad_accum}
