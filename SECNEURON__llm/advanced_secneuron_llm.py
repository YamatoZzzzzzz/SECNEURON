
# -*- coding: utf-8 -*-
"""
Advanced SECNEURON-style LLM Reproduction Suite

Experiments included:
1. Task-level capability control on small LLMs (OPT-125M / OPT-350M)
2. Data-level abuse protection: real PII extraction attempt + loss-based MIA
3. Multi-task dynamic permission control
4. Code-generation evaluation: HumanEval / MBPP compile@k and optional pass@k
5. Naive pruning vs task-specific neuron selection

Upgrades:
- Real HuggingFace datasets: TinyStories, CodeParrot, BioASQ, Enron Email, Arxiv
- Small LLMs: OPT-125M / OPT-350M
- Optional real CP-ABE via Charm-Crypto
- Real HumanEval / MBPP style generation metrics
- Real PII extraction and MIA pipelines

Important:
This is an advanced, practical reproduction scaffold. It is NOT the official
SECNEURON code and will not numerically match the 6.7B/30B paper results on a CPU.
"""

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from Crypto.Cipher import AES
    from Crypto.Util import Counter
    CRYPTO_OK = True
except Exception:
    CRYPTO_OK = False


# ----------------------------
# Config
# ----------------------------

DEFAULT_DATASETS = {
    "story": "roneneldan/TinyStories",
    "code": "codeparrot/github-code-clean",
    "health": "enelpol/rag-mini-bioasq-qas-clean",
    "email": "LLM-PBE/enron-email",
    # The paper name may appear as "haritzpuerto/the pile 00 arxiv".
    # HuggingFace identifiers sometimes change. Override this with --arxiv_dataset if needed.
    "arxiv": "haritzpuerto/the_pile_00_arxiv",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def now():
    return time.perf_counter()


# ----------------------------
# Dataset loading
# ----------------------------

def row_to_text(row: dict, task: str) -> str:
    """Try to extract/compose text from common HF dataset schemas."""
    # task-specific priority
    if task == "code":
        for k in ["code", "content", "text", "func_code"]:
            if k in row and isinstance(row[k], str) and row[k].strip():
                return row[k]
    if task == "story":
        for k in ["text", "story", "content"]:
            if k in row and isinstance(row[k], str) and row[k].strip():
                return row[k]
    if task == "health":
        # BioASQ-like QA
        parts = []
        for k in ["question", "answer", "context", "text", "content"]:
            v = row.get(k, None)
            if isinstance(v, str) and v.strip():
                parts.append(v.strip())
            elif isinstance(v, list):
                parts.extend([str(x) for x in v if str(x).strip()])
        if parts:
            return "\n".join(parts)
    if task == "email":
        for k in ["text", "email", "message", "content", "body"]:
            if k in row and isinstance(row[k], str) and row[k].strip():
                return row[k]
    if task == "arxiv":
        for k in ["text", "abstract", "article", "content"]:
            if k in row and isinstance(row[k], str) and row[k].strip():
                return row[k]

    # generic fallback: concatenate string fields
    parts = []
    for k, v in row.items():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, list):
            small = [str(x) for x in v[:3] if str(x).strip()]
            if small:
                parts.append(" ".join(small))
    return "\n".join(parts)[:8000]


def load_texts(dataset_name: str, task: str, split: str, limit: int,
               streaming: bool = True, allow_fallback: bool = False) -> List[str]:
    """Load `limit` texts from a HuggingFace dataset."""
    texts = []
    try:
        ds = load_dataset(dataset_name, split=split, streaming=streaming, trust_remote_code=True)
        it = iter(ds)
        for row in it:
            txt = row_to_text(row, task)
            if txt and len(txt.strip()) > 20:
                texts.append(txt.strip())
            if len(texts) >= limit:
                break
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                f"Failed to load dataset {dataset_name} split={split}. "
                f"Use --allow_fallback_demo to run toy samples, or override dataset id. Original error: {e}"
            )
        texts = fallback_texts(task, limit)
    return texts


def fallback_texts(task: str, limit: int) -> List[str]:
    samples = {
        "story": [
            "Once upon a time, a small bird learned to fly over the quiet forest.",
            "The child found a red box under the bed and wondered what was inside.",
            "A young painter dreamed of drawing the moon on a winter night.",
        ],
        "code": [
            "def add(a, b):\n    return a + b\n",
            "class Stack:\n    def __init__(self):\n        self.items=[]\n",
            "for i in range(10):\n    print(i)\n",
        ],
        "health": [
            "A patient with fever and cough should drink water and seek medical advice.",
            "Hypertension is commonly treated with lifestyle changes and medication.",
            "The symptoms include headache, nausea, and fatigue after infection.",
        ],
        "email": [
            "From: alice@example.com\nTo: bob@example.com\nPlease call me at 555-123-4567 about the meeting.",
            "Contact john.doe@company.com for the confidential report.",
            "My phone number is 212-555-0199 and my address is 5 Main Street.",
        ],
        "arxiv": [
            "We propose a neural architecture for representation learning in high dimensional data.",
            "This paper studies optimization methods for transformer language models.",
            "Experimental results demonstrate improvements over baseline algorithms.",
        ],
    }
    arr = samples.get(task, ["This is a generic document for language modeling."])
    return [arr[i % len(arr)] for i in range(limit)]


def prepare_task_texts(args) -> Dict[str, Dict[str, List[str]]]:
    dataset_ids = dict(DEFAULT_DATASETS)
    if args.tinystories_dataset: dataset_ids["story"] = args.tinystories_dataset
    if args.code_dataset: dataset_ids["code"] = args.code_dataset
    if args.bioasq_dataset: dataset_ids["health"] = args.bioasq_dataset
    if args.enron_dataset: dataset_ids["email"] = args.enron_dataset
    if args.arxiv_dataset: dataset_ids["arxiv"] = args.arxiv_dataset

    tasks = ["story", "health", "code", "email", "arxiv"]
    data = {}
    for t in tasks:
        print(f"[Data] Loading {t}: {dataset_ids[t]}")
        total = args.train_samples + args.eval_samples + args.mia_nonmember_samples + 10
        texts = load_texts(dataset_ids[t], t, args.dataset_split, total,
                           streaming=not args.no_streaming,
                           allow_fallback=args.allow_fallback_demo)
        train = texts[:args.train_samples]
        eval_ = texts[args.train_samples:args.train_samples + args.eval_samples]
        extra = texts[args.train_samples + args.eval_samples:]
        data[t] = {"train": train, "eval": eval_, "extra": extra, "dataset": dataset_ids[t]}
        print(f"       train={len(train)}, eval={len(eval_)}, extra={len(extra)}")
    return data


# ----------------------------
# Model and OPT neuron utilities
# ----------------------------

def load_lm(model_name: str, device: str, dtype: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = None
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def get_opt_layers(model):
    """
    Supports OPTForCausalLM: model.model.decoder.layers[i].fc1/fc2.
    Raises error otherwise.
    """
    if not hasattr(model, "model") or not hasattr(model.model, "decoder"):
        raise ValueError("This scaffold currently supports OPT-like models with model.model.decoder.layers.")
    layers = model.model.decoder.layers
    out = []
    for i, layer in enumerate(layers):
        if not hasattr(layer, "fc1") or not hasattr(layer, "fc2"):
            raise ValueError("This scaffold expects OPT MLP layers with fc1/fc2.")
        out.append((i, layer.fc1, layer.fc2))
    return out


def neuron_key(layer_id: int, neuron_id: int) -> str:
    return f"{layer_id}:{neuron_id}"


def parse_neuron_key(k: str) -> Tuple[int, int]:
    a, b = k.split(":")
    return int(a), int(b)


def tokenize_batch(tokenizer, texts: List[str], max_length: int, device: str):
    return tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)


def align_activation_and_mask(act: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align OPT MLP activation with attention_mask.

    Some OPT/Transformers versions pass MLP tensors as [seq_len, batch, hidden]
    instead of [batch, seq_len, hidden]. This helper normalizes them to
    [batch, seq_len, hidden] before applying the mask.
    """
    if act.dim() != 3:
        raise RuntimeError(f"Expected activation with 3 dims, got shape={tuple(act.shape)}")

    # Desired: [B, L, H]
    if act.shape[0] == mask.shape[0]:
        # Already [B, L, H]
        act_blh = act
    elif act.shape[1] == mask.shape[0]:
        # Likely [L, B, H]
        act_blh = act.transpose(0, 1)
    else:
        raise RuntimeError(
            f"Cannot align activation shape {tuple(act.shape)} with mask shape {tuple(mask.shape)}"
        )

    L = min(act_blh.shape[1], mask.shape[1])
    act_blh = act_blh[:, :L, :]
    mask_bl = mask[:, :L]
    return act_blh, mask_bl



@torch.no_grad()
def collect_activation_importance(model, tokenizer, data, tasks, device, max_length, batch_size) -> Dict[str, Dict[str, float]]:
    """Mean absolute fc1 output activation per neuron per task."""
    layers = get_opt_layers(model)
    importance = {}
    for task in tasks:
        print(f"[I] Collecting activation importance for {task}")
        sums = {neuron_key(i, j): 0.0 for i, fc1, fc2 in layers for j in range(fc1.out_features)}
        counts = 0

        captured = {}
        hooks = []
        def make_hook(layer_id):
            def hook(module, inp, out):
                captured[layer_id] = out.detach()
            return hook
        for i, fc1, _ in layers:
            hooks.append(fc1.register_forward_hook(make_hook(i)))

        texts = data[task]["train"]
        for b in tqdm(range(0, len(texts), batch_size), leave=False):
            batch = texts[b:b+batch_size]
            enc = tokenize_batch(tokenizer, batch, max_length, device)
            captured.clear()
            _ = model(**enc)
            mask = enc["attention_mask"].detach().float()  # [B,L]
            for layer_id, act in captured.items():
                act_blh, mask_bl = align_activation_and_mask(act, mask)
                m = mask_bl.unsqueeze(-1)
                s = (act_blh.abs().float() * m).sum(dim=(0, 1)).cpu().numpy()
                for j, val in enumerate(s):
                    sums[neuron_key(layer_id, j)] += float(val)
            counts += mask.sum().item()

        for h in hooks:
            h.remove()
        importance[task] = {k: v / max(counts, 1) for k, v in sums.items()}
    return importance


def collect_gradient_sensitivity(model, tokenizer, data, tasks, device, max_length, batch_size, max_batches=20):
    """Mean |activation * grad| on fc1 outputs, approximate and expensive."""
    layers = get_opt_layers(model)
    grad_scores = {}

    for task in tasks:
        print(f"[G] Collecting gradient sensitivity for {task}")
        sums = {neuron_key(i, j): 0.0 for i, fc1, fc2 in layers for j in range(fc1.out_features)}
        counts = 0

        captured = {}
        hooks = []
        def make_hook(layer_id):
            def hook(module, inp, out):
                out.retain_grad()
                captured[layer_id] = out
            return hook

        for i, fc1, _ in layers:
            hooks.append(fc1.register_forward_hook(make_hook(i)))

        texts = data[task]["train"]
        batches_done = 0
        for b in tqdm(range(0, len(texts), batch_size), leave=False):
            if batches_done >= max_batches:
                break
            batch = texts[b:b+batch_size]
            enc = tokenize_batch(tokenizer, batch, max_length, device)
            labels = enc["input_ids"].clone()
            model.zero_grad(set_to_none=True)
            captured.clear()
            out = model(**enc, labels=labels)
            loss = out.loss
            loss.backward()
            mask = enc["attention_mask"].detach().float()

            for layer_id, act in captured.items():
                if act.grad is None:
                    continue
                act_blh, mask_bl = align_activation_and_mask(act.detach(), mask)
                grad_blh, _ = align_activation_and_mask(act.grad.detach(), mask)
                m = mask_bl.unsqueeze(-1)
                s = (act_blh.abs().float() * grad_blh.abs().float() * m).sum(dim=(0, 1)).cpu().numpy()
                for j, val in enumerate(s):
                    sums[neuron_key(layer_id, j)] += float(val)

            counts += mask.sum().item()
            batches_done += 1

        for h in hooks:
            h.remove()
        grad_scores[task] = {k: v / max(counts, 1) for k, v in sums.items()}
    return grad_scores


def normalize_dict(d: Dict[str, float]):
    vals = np.array(list(d.values()), dtype=np.float64)
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-12:
        return {k: 0.0 for k in d}
    return {k: float((v - mn) / (mx - mn)) for k, v in d.items()}


def compute_scores(importance, tasks, lambda_penalty, gradient=None, grad_weight=0.0):
    I = {t: normalize_dict(importance[t]) for t in tasks}
    if gradient is not None:
        G = {t: normalize_dict(gradient[t]) for t in tasks}
    else:
        G = {t: {k: 0.0 for k in I[t]} for t in tasks}

    scores = {}
    keys = list(next(iter(I.values())).keys())
    for t in tasks:
        scores[t] = {}
        for k in keys:
            max_other = max(I[o][k] for o in tasks if o != t)
            scores[t][k] = I[t][k] + grad_weight * G[t][k] - lambda_penalty * max_other
    return scores


def select_neurons(scores, tasks, k_per_task):
    selected = {}
    for t in tasks:
        items = sorted(scores[t].items(), key=lambda x: x[1], reverse=True)
        selected[t] = set(k for k, v in items[:k_per_task])
    return selected


def build_membership(selected, tasks):
    all_keys = set()
    for t in tasks:
        all_keys |= set(selected[t])
    membership = {k: set() for k in all_keys}
    for t in tasks:
        for k in selected[t]:
            membership[k].add(t)
    return membership


# ----------------------------
# CP-ABE key management
# ----------------------------

class ABEManager:
    """
    Optional real CP-ABE through Charm-Crypto. Use --abe_backend charm.
    On Windows, Charm-Crypto is difficult to install; WSL/Linux is recommended.
    """
    def __init__(self, backend="mock"):
        self.backend = backend
        self.group = None
        self.cpabe = None
        self.pk = None
        self.msk = None
        if backend == "charm":
            try:
                from charm.toolbox.pairinggroup import PairingGroup, GT
                from charm.schemes.abenc.abenc_bsw07 import CPabe_BSW07
                self.GT = GT
                self.group = PairingGroup("SS512")
                self.cpabe = CPabe_BSW07(self.group)
                self.pk, self.msk = self.cpabe.setup()
            except Exception as e:
                raise RuntimeError(
                    "Charm-Crypto CP-ABE backend failed. Install charm-crypto in WSL/Linux, "
                    "or use --abe_backend mock. Original error: " + repr(e)
                )

    def _policy_from_tasks(self, taskset: Set[str], all_tasks: List[str]) -> str:
        if not taskset:
            taskset = set(all_tasks)
        attrs = [t.upper() for t in sorted(taskset)]
        if len(attrs) == 1:
            return attrs[0]
        return "(" + " or ".join(attrs) + ")"

    def encrypt_subset_keys(self, subsets: Dict[str, Set[str]], all_tasks: List[str]):
        """
        Returns dict subset_name -> encrypted key object.
        Mock: stores policy and raw random key.
        Charm: stores policy and CP-ABE ciphertext over random GT element.
        """
        enc = {}
        for subset_name, taskset in subsets.items():
            policy = self._policy_from_tasks(taskset, all_tasks)
            if self.backend == "charm":
                msg = self.group.random(self.GT)
                ct = self.cpabe.encrypt(self.pk, msg, policy)
                enc[subset_name] = {"policy": policy, "ct": ct, "msg_for_dev": msg}
            else:
                raw = os.urandom(16)
                enc[subset_name] = {"policy": policy, "raw_key": raw.hex()}
        return enc

    def derive_authorized_subsets(self, encrypted_keys, attrs: Set[str]):
        """
        Return subset names whose keys can be obtained.
        """
        attrs_upper = [a.upper() for a in attrs]
        auth = set()
        aes_keys = {}
        if self.backend == "charm":
            sk = self.cpabe.keygen(self.pk, self.msk, attrs_upper)
            for name, obj in encrypted_keys.items():
                try:
                    msg = self.cpabe.decrypt(self.pk, sk, obj["ct"])
                    if msg is not False and msg is not None:
                        key = hashlib.sha256(self.group.serialize(msg)).digest()[:16]
                        auth.add(name); aes_keys[name] = key.hex()
                except Exception:
                    pass
        else:
            for name, obj in encrypted_keys.items():
                policy = obj["policy"]
                # Minimal OR evaluator for policies like "(CODE or HEALTH)".
                allowed = [x.strip("() ").upper() for x in policy.replace("or", "OR").split("OR")]
                if any(a in attrs_upper for a in allowed):
                    auth.add(name); aes_keys[name] = obj["raw_key"]
        return auth, aes_keys


def membership_to_subsets(membership, all_tasks):
    """Map subset name to taskset."""
    out = {}
    for k, mem in membership.items():
        name = "+".join(sorted(mem)) if mem else "common"
        out.setdefault(name, set(mem))
    return out


# ----------------------------
# AES-CTR neuron encryption sanity
# ----------------------------

def aes_roundtrip_tensor(t: torch.Tensor, key: bytes, counter_id: int):
    if not CRYPTO_OK:
        return float("nan")
    arr = t.detach().cpu().contiguous().float().numpy()
    raw = arr.tobytes()
    ctr1 = Counter.new(128, initial_value=counter_id)
    enc = AES.new(key, AES.MODE_CTR, counter=ctr1).encrypt(raw)
    ctr2 = Counter.new(128, initial_value=counter_id)
    dec = AES.new(key, AES.MODE_CTR, counter=ctr2).decrypt(enc)
    rec = np.frombuffer(dec, dtype=np.float32).reshape(arr.shape)
    return float(np.max(np.abs(rec - arr)))


# ----------------------------
# Pruning / selective decryption
# ----------------------------

def subset_name_from_mem(mem: Set[str]) -> str:
    return "+".join(sorted(mem)) if mem else "common"


def unauthorized_neurons(membership, authorized_tasks: Set[str], passive_all_selected=True):
    pruned = []
    for k, mem in membership.items():
        if len(authorized_tasks) == 0 and passive_all_selected:
            pruned.append(k)
        elif mem and not mem.intersection(authorized_tasks):
            pruned.append(k)
    return pruned


@contextlib.contextmanager
def temporarily_prune_fc2(model, neuron_keys: List[str]):
    layers = get_opt_layers(model)
    layer_map = {i: (fc1, fc2) for i, fc1, fc2 in layers}
    backups = []
    with torch.no_grad():
        for nk in neuron_keys:
            li, ni = parse_neuron_key(nk)
            fc1, fc2 = layer_map[li]
            if ni >= fc2.weight.shape[1]:
                continue
            backups.append((fc2, ni, fc2.weight[:, ni].detach().clone()))
            fc2.weight[:, ni].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            for fc2, ni, w in backups:
                fc2.weight[:, ni].copy_(w)


@torch.no_grad()
def token_accuracy(model, tokenizer, texts, device, max_length, batch_size):
    correct = 0
    total = 0
    model.eval()
    for b in range(0, len(texts), batch_size):
        batch = texts[b:b+batch_size]
        enc = tokenize_batch(tokenizer, batch, max_length, device)
        logits = model(**enc).logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        preds = logits.argmax(dim=-1)
        correct += ((preds == labels) & mask).sum().item()
        total += mask.sum().item()
    return correct / max(total, 1)


def eval_tasks_with_auth(model, tokenizer, data, membership, authorized_tasks, eval_tasks, device, max_length, batch_size):
    pruned = unauthorized_neurons(membership, authorized_tasks)
    with temporarily_prune_fc2(model, pruned):
        accs = {t: token_accuracy(model, tokenizer, data[t]["eval"], device, max_length, batch_size) for t in eval_tasks}
    return accs, len(pruned)


def plot_bars(df, xcol, ycol, huecol, title, path, ylim=(0,1)):
    xs = list(df[xcol].unique())
    hues = list(df[huecol].unique())
    x = np.arange(len(xs))
    width = 0.8 / max(1, len(hues))
    plt.figure(figsize=(10, 5))
    for i, h in enumerate(hues):
        vals = []
        for xval in xs:
            sub = df[(df[xcol] == xval) & (df[huecol] == h)]
            vals.append(float(sub[ycol].values[0]) if len(sub) else 0.0)
        plt.bar(x + (i-len(hues)/2)*width + width/2, vals, width, label=h)
    plt.xticks(x, xs, rotation=25)
    plt.ylabel(ycol)
    plt.ylim(*ylim)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ----------------------------
# Real PII extraction / MIA
# ----------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")


def extract_pii(text):
    vals = set(EMAIL_RE.findall(text))
    vals |= set(m[0] if isinstance(m, tuple) else m for m in PHONE_RE.findall(text))
    return {v.strip() for v in vals if len(v.strip()) >= 6}


@torch.no_grad()
def generate_text(model, tokenizer, prompt, device, max_new_tokens=48):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    out = model.generate(
        **enc,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def run_pii_extraction(model, tokenizer, email_texts, membership, role_to_tasks, device, out_dir, max_cases=40):
    rows = []
    candidates = []
    for txt in email_texts:
        pii = extract_pii(txt)
        if pii:
            # prompt prefix before first PII if possible
            first = min((txt.find(p) for p in pii if txt.find(p) >= 0), default=0)
            prefix = txt[:max(30, first)]
            if len(prefix) > 250:
                prefix = prefix[-250:]
            candidates.append((prefix, pii))
        if len(candidates) >= max_cases:
            break

    if not candidates:
        pd.DataFrame([{"role":"none","success_rate":0,"cases":0}]).to_csv(os.path.join(out_dir, "exp2_real_pii_extraction.csv"), index=False)
        return

    for role, auth in role_to_tasks.items():
        pruned = unauthorized_neurons(membership, auth)
        success = 0
        total = 0
        with temporarily_prune_fc2(model, pruned):
            for prefix, pii in tqdm(candidates, desc=f"PII {role}", leave=False):
                gen = generate_text(model, tokenizer, prefix, device, max_new_tokens=48)
                total += 1
                if any(p in gen for p in pii):
                    success += 1
        rows.append({"role": role, "success_rate": success / max(total,1), "cases": total, "pruned_neurons": len(pruned)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp2_real_pii_extraction.csv"), index=False)
    plt.figure(figsize=(7,4))
    plt.bar(df["role"], df["success_rate"])
    plt.ylabel("PII extraction success rate")
    plt.title("Real PII extraction attempt on Enron-like texts")
    plt.xticks(rotation=25)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp2_real_pii_extraction.png"), dpi=200)
    plt.close()


@torch.no_grad()
def mean_nll(model, tokenizer, texts, device, max_length, batch_size):
    vals = []
    for b in range(0, len(texts), batch_size):
        batch = texts[b:b+batch_size]
        enc = tokenize_batch(tokenizer, batch, max_length, device)
        out = model(**enc, labels=enc["input_ids"])
        # batch mean loss repeated; for better per sample, process one by one for small sets
        vals.append(float(out.loss.item()))
    return np.array(vals)


@torch.no_grad()
def per_text_nll(model, tokenizer, texts, device, max_length):
    losses = []
    for txt in tqdm(texts, desc="NLL", leave=False):
        enc = tokenize_batch(tokenizer, [txt], max_length, device)
        out = model(**enc, labels=enc["input_ids"])
        losses.append(float(out.loss.item()))
    return np.array(losses)


def roc_auc(labels, scores):
    labels = np.array(labels); scores = np.array(scores)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos)==0 or len(neg)==0: return float("nan")
    s = 0.0
    for p in pos:
        s += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(s / (len(pos)*len(neg)))


def run_mia(model, tokenizer, arxiv_member, arxiv_nonmember, membership, role_to_tasks, device, out_dir, max_length):
    rows = []
    member = arxiv_member[:40]
    nonmember = arxiv_nonmember[:40]
    for role, auth in role_to_tasks.items():
        pruned = unauthorized_neurons(membership, auth)
        with temporarily_prune_fc2(model, pruned):
            lm = per_text_nll(model, tokenizer, member, device, max_length)
            ln = per_text_nll(model, tokenizer, nonmember, device, max_length)
        # lower loss => more likely member, so score=-loss
        scores = np.concatenate([-lm, -ln])
        labels = np.concatenate([np.ones(len(lm)), np.zeros(len(ln))])
        rows.append({"role": role, "mia_auroc": roc_auc(labels, scores), "member_cases": len(lm), "nonmember_cases": len(ln)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp2_real_mia_loss_auroc.csv"), index=False)
    plt.figure(figsize=(7,4))
    plt.bar(df["role"], df["mia_auroc"])
    plt.axhline(0.5, linestyle="--")
    plt.ylabel("MIA AUROC")
    plt.title("Real loss-based MIA on Arxiv texts")
    plt.xticks(rotation=25)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp2_real_mia_loss_auroc.png"), dpi=200)
    plt.close()


# ----------------------------
# Code generation metrics
# ----------------------------

def try_compile(code: str) -> bool:
    try:
        compile(code, "<generated>", "exec")
        return True
    except Exception:
        return False


def load_humaneval(limit=20):
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    rows = []
    for r in ds:
        rows.append({"prompt": r["prompt"], "test": r.get("test",""), "entry_point": r.get("entry_point","")})
        if len(rows) >= limit: break
    return rows


def load_mbpp(limit=20):
    try:
        ds = load_dataset("google-research-datasets/mbpp", split="test", trust_remote_code=True)
    except Exception:
        ds = load_dataset("mbpp", split="test", trust_remote_code=True)
    rows = []
    for r in ds:
        prompt = r.get("text") or r.get("prompt") or str(r)
        tests = r.get("test_list") or r.get("test") or []
        if isinstance(tests, list):
            tests = "\n".join(tests)
        rows.append({"prompt": prompt, "test": tests, "entry_point": ""})
        if len(rows) >= limit: break
    return rows


@torch.no_grad()
def sample_code(model, tokenizer, prompt, device, k=1, max_new_tokens=160, temperature=0.2):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outs = []
    for _ in range(k):
        out = model.generate(
            **enc,
            do_sample=True if temperature > 0 else False,
            temperature=max(temperature, 1e-5),
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        outs.append(text)
    return outs


def run_code_eval(model, tokenizer, membership, role_to_tasks, device, out_dir,
                  benchmark="humaneval", limit=10, k=1, allow_execution=False):
    if benchmark == "humaneval":
        problems = load_humaneval(limit)
    else:
        problems = load_mbpp(limit)

    rows = []
    predictions_by_role = {}
    references = []

    for p in problems:
        references.append(p["test"])

    for role, auth in role_to_tasks.items():
        pruned = unauthorized_neurons(membership, auth)
        compile_success = 0
        total = 0
        role_predictions = []
        with temporarily_prune_fc2(model, pruned):
            for p in tqdm(problems, desc=f"code {role}", leave=False):
                preds = sample_code(model, tokenizer, p["prompt"], device, k=k)
                role_predictions.append(preds)
                if any(try_compile(x) for x in preds):
                    compile_success += 1
                total += 1
        rows.append({
            "role": role,
            "benchmark": benchmark,
            "compile_at_k": compile_success / max(total,1),
            "problems": total,
            "k": k,
            "pruned_neurons": len(pruned),
        })
        predictions_by_role[role] = role_predictions

    df = pd.DataFrame(rows)

    if allow_execution:
        # Optional real pass@k through evaluate.code_eval.
        # WARNING: executes generated code. Run only inside a disposable sandbox.
        try:
            os.environ["HF_ALLOW_CODE_EVAL"] = "1"
            import evaluate
            code_eval = evaluate.load("code_eval")
            for role in list(predictions_by_role.keys()):
                pass_at_k, _ = code_eval.compute(
                    references=references,
                    predictions=predictions_by_role[role],
                    k=[k]
                )
                df.loc[df["role"] == role, f"pass@{k}"] = pass_at_k.get(f"pass@{k}", np.nan)
        except Exception as e:
            df["pass_error"] = repr(e)

    df.to_csv(os.path.join(out_dir, f"exp4_{benchmark}_code_generation.csv"), index=False)
    plt.figure(figsize=(7,4))
    plt.bar(df["role"], df["compile_at_k"])
    plt.ylabel(f"compile@{k}")
    plt.title(f"Code generation metric on {benchmark}")
    plt.xticks(rotation=25)
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"exp4_{benchmark}_compile_at_k.png"), dpi=200)
    plt.close()


# ----------------------------
# Experiments
# ----------------------------

def run_exp1_task_control(model, tokenizer, data, membership, out_dir, device, max_length, batch_size):
    roles = {
        "baseline": None,
        "admin": None,
        "story_only": {"story"},
        "health_only": {"health"},
        "code_only": {"code"},
        "passive": set(),
    }
    rows = []
    for role, auth in roles.items():
        if auth is None:
            accs = {t: token_accuracy(model, tokenizer, data[t]["eval"], device, max_length, batch_size) for t in ["story","health","code"]}
            pruned = 0
        else:
            accs, pruned = eval_tasks_with_auth(model, tokenizer, data, membership, auth, ["story","health","code"], device, max_length, batch_size)
        for t, a in accs.items():
            rows.append({"role": role, "task": t, "accuracy": a, "pruned_neurons": pruned})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp1_task_control_metrics.csv"), index=False)
    plot_bars(df, "role", "accuracy", "task", "Experiment 1: Task-level capability control", os.path.join(out_dir, "exp1_task_control.png"))
    return df


def run_exp3_multitask(model, tokenizer, data, membership, out_dir, device, max_length, batch_size):
    perms = {
        "all": {"story","health","code","email","arxiv"},
        "no_health": {"story","code","email","arxiv"},
        "story_health": {"story","health"},
        "no_code": {"story","health","email","arxiv"},
        "none": set(),
    }
    tasks = ["story","health","code","email","arxiv"]
    rows = []
    for pname, auth in perms.items():
        accs, pruned = eval_tasks_with_auth(model, tokenizer, data, membership, auth, tasks, device, max_length, batch_size)
        for t, a in accs.items():
            rows.append({"permission": pname, "task": t, "accuracy": a, "pruned_neurons": pruned})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp3_multitask_permissions.csv"), index=False)

    pivot = df.pivot(index="permission", columns="task", values="accuracy").loc[list(perms.keys()), tasks]
    plt.figure(figsize=(8, 4.8))
    plt.imshow(pivot.values, vmin=0, vmax=max(0.01, np.nanmax(pivot.values)), aspect="auto")
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(tasks)), tasks)
    plt.yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Experiment 3: Multi-task dynamic permissions")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp3_multitask_permissions_heatmap.png"), dpi=200)
    plt.close()
    return df


def run_exp5_naive_vs_specific(model, tokenizer, data, importance, scores, out_dir, device, max_length, batch_size):
    target = "code"
    auth_tasks = ["story","health","email","arxiv"]

    baseline = {t: token_accuracy(model, tokenizer, data[t]["eval"], device, max_length, batch_size) for t in auth_tasks + [target]}
    base_auth = np.mean([baseline[t] for t in auth_tasks])
    base_code = baseline[target]

    naive_rank = sorted(importance[target].items(), key=lambda x: x[1], reverse=True)
    spec_rank = sorted(scores[target].items(), key=lambda x: x[1], reverse=True)

    k_values = [max(5, int(x)) for x in np.linspace(5, min(len(spec_rank), 120), 6)]
    rows = []
    for method, rank in [("naive_I", naive_rank), ("task_specific_S", spec_rank)]:
        for k in k_values:
            pruned = [nk for nk, _ in rank[:k]]
            with temporarily_prune_fc2(model, pruned):
                acc_auth = [token_accuracy(model, tokenizer, data[t]["eval"], device, max_length, batch_size) for t in auth_tasks]
                acc_code = token_accuracy(model, tokenizer, data[target]["eval"], device, max_length, batch_size)
            rows.append({
                "method": method, "k": k,
                "authorized_drop": base_auth - float(np.mean(acc_auth)),
                "code_drop": base_code - acc_code,
                "authorized_mean_accuracy": float(np.mean(acc_auth)),
                "code_accuracy": acc_code
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "exp5_naive_vs_specific.csv"), index=False)
    plt.figure(figsize=(6,5))
    for method in ["naive_I", "task_specific_S"]:
        sub = df[df["method"] == method]
        plt.plot(sub["authorized_drop"], sub["code_drop"], marker="o", label=method)
        for _, r in sub.iterrows():
            plt.text(r["authorized_drop"], r["code_drop"], str(int(r["k"])), fontsize=8)
    plt.xlabel("Authorized task accuracy drop ↓")
    plt.ylabel("Unauthorized code accuracy drop ↑")
    plt.title("Experiment 5: Naive pruning vs task-specific selection")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp5_naive_vs_specific.png"), dpi=200)
    plt.close()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--dtype", default="fp32", choices=["auto","fp32","fp16","bf16"])
    ap.add_argument("--out_dir", default="advanced_results")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--train_samples", type=int, default=60)
    ap.add_argument("--eval_samples", type=int, default=30)
    ap.add_argument("--mia_nonmember_samples", type=int, default=60)
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--no_streaming", action="store_true")
    ap.add_argument("--allow_fallback_demo", action="store_true")

    ap.add_argument("--tinystories_dataset", default="")
    ap.add_argument("--code_dataset", default="")
    ap.add_argument("--bioasq_dataset", default="")
    ap.add_argument("--enron_dataset", default="")
    ap.add_argument("--arxiv_dataset", default="")

    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--k_per_task", type=int, default=80)
    ap.add_argument("--lambda_penalty", type=float, default=0.8)
    ap.add_argument("--use_gradient", action="store_true")
    ap.add_argument("--grad_weight", type=float, default=0.35)
    ap.add_argument("--grad_max_batches", type=int, default=8)

    ap.add_argument("--abe_backend", default="mock", choices=["mock","charm"])
    ap.add_argument("--run_humaneval", action="store_true")
    ap.add_argument("--run_mbpp", action="store_true")
    ap.add_argument("--code_limit", type=int, default=5)
    ap.add_argument("--code_k", type=int, default=1)
    ap.add_argument("--allow_code_execution", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)
    save_json(vars(args), os.path.join(args.out_dir, "run_config.json"))

    print("[1] Loading real datasets")
    data = prepare_task_texts(args)
    save_json({t: {"dataset": data[t]["dataset"], "train": len(data[t]["train"]), "eval": len(data[t]["eval"])} for t in data}, os.path.join(args.out_dir, "data_summary.json"))

    print("[2] Loading model")
    tokenizer, model = load_lm(args.model, args.device, args.dtype)

    tasks = ["story","health","code","email","arxiv"]

    print("[3] Collecting I(t,n)")
    importance = collect_activation_importance(model, tokenizer, data, tasks, args.device, args.max_length, args.batch_size)

    gradient = None
    if args.use_gradient:
        print("[4] Collecting G(t,n)")
        gradient = collect_gradient_sensitivity(model, tokenizer, data, tasks, args.device, args.max_length, args.batch_size, args.grad_max_batches)

    print("[5] Scoring and selecting neurons")
    scores = compute_scores(importance, tasks, args.lambda_penalty, gradient, args.grad_weight if args.use_gradient else 0.0)
    selected = select_neurons(scores, tasks, args.k_per_task)
    membership = build_membership(selected, tasks)

    save_json({t: sorted(list(v)) for t,v in selected.items()}, os.path.join(args.out_dir, "selected_neurons.json"))
    save_json({k: sorted(list(v)) for k,v in membership.items()}, os.path.join(args.out_dir, "membership.json"))

    print("    selected neurons per task:")
    for t in tasks:
        print(f"    {t}: {len(selected[t])}")
    print(f"    unique selected neurons: {len(membership)}")

    print("[6] Optional CP-ABE key management")
    subset_tasksets = membership_to_subsets(membership, tasks)
    abe = ABEManager(args.abe_backend)
    t0 = now()
    encrypted_keys = abe.encrypt_subset_keys(subset_tasksets, tasks)
    enc_key_time = now() - t0
    t1 = now()
    auth_subsets, aes_keys = abe.derive_authorized_subsets(encrypted_keys, {"CODE"})
    dec_key_time = now() - t1
    save_json({
        "backend": args.abe_backend,
        "num_subset_keys": len(encrypted_keys),
        "key_encrypt_time_sec": enc_key_time,
        "code_user_key_decrypt_time_sec": dec_key_time,
        "code_authorized_subset_count": len(auth_subsets),
        "note": "mock backend evaluates policy logic; charm backend performs real CP-ABE if installed."
    }, os.path.join(args.out_dir, "cpabe_key_management_report.json"))

    print("[7] AES-CTR sanity check on one neuron")
    if CRYPTO_OK and len(membership) > 0:
        nk = next(iter(membership.keys()))
        li, ni = parse_neuron_key(nk)
        layers = get_opt_layers(model)
        layer_map = {i:(fc1,fc2) for i,fc1,fc2 in layers}
        fc1, fc2 = layer_map[li]
        key = os.urandom(16)
        e1 = aes_roundtrip_tensor(fc1.weight[ni:ni+1, :], key, 123)
        e2 = aes_roundtrip_tensor(fc2.weight[:, ni:ni+1], key, 456)
        save_json({"fc1_weight_roundtrip_max_error": e1, "fc2_weight_roundtrip_max_error": e2}, os.path.join(args.out_dir, "aes_roundtrip_report.json"))
        print(f"    AES roundtrip errors: fc1={e1}, fc2={e2}")

    print("[8] Experiment 1: task-level control")
    run_exp1_task_control(model, tokenizer, data, membership, args.out_dir, args.device, args.max_length, args.batch_size)

    print("[9] Experiment 2: real PII and MIA")
    roles = {
        "baseline": None,
        "story_only": {"story"},
        "passive": set(),
    }
    # For PII and MIA functions, None baseline not accepted as set; handle baseline by all tasks.
    role_to_tasks = {"baseline": set(tasks), "story_only": {"story"}, "passive": set()}
    run_pii_extraction(model, tokenizer, data["email"]["eval"] + data["email"]["extra"], membership, role_to_tasks, args.device, args.out_dir, max_cases=20)
    run_mia(model, tokenizer, data["arxiv"]["train"], data["arxiv"]["extra"], membership, role_to_tasks, args.device, args.out_dir, args.max_length)

    print("[10] Experiment 3: multi-task permissions")
    run_exp3_multitask(model, tokenizer, data, membership, args.out_dir, args.device, args.max_length, args.batch_size)

    print("[11] Experiment 4: code-generation metrics")
    if args.run_humaneval:
        run_code_eval(model, tokenizer, membership, {"baseline": set(tasks), "code_auth": {"code"}, "story_only_no_code": {"story"}, "passive": set()},
                      args.device, args.out_dir, benchmark="humaneval", limit=args.code_limit, k=args.code_k, allow_execution=args.allow_code_execution)
    if args.run_mbpp:
        run_code_eval(model, tokenizer, membership, {"baseline": set(tasks), "code_auth": {"code"}, "story_only_no_code": {"story"}, "passive": set()},
                      args.device, args.out_dir, benchmark="mbpp", limit=args.code_limit, k=args.code_k, allow_execution=args.allow_code_execution)

    print("[12] Experiment 5: naive pruning vs task-specific selection")
    run_exp5_naive_vs_specific(model, tokenizer, data, importance, scores, args.out_dir, args.device, args.max_length, args.batch_size)

    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# Advanced SECNEURON-style LLM Reproduction Report\n\n")
        f.write("This run uses real HF datasets and a small OPT-like LLM.\n\n")
        f.write("Important: exact paper-level results require large LLMs and original engineering setup.\n\n")
        f.write("Generated figures:\n\n")
        for name in [
            "exp1_task_control.png",
            "exp2_real_pii_extraction.png",
            "exp2_real_mia_loss_auroc.png",
            "exp3_multitask_permissions_heatmap.png",
            "exp5_naive_vs_specific.png",
        ]:
            f.write(f"- {name}\n")
    print(f"Done. Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()
