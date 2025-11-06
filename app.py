"""
Flask summarization API — CLEANED for index1.html
Returns only the fields used by the frontend:
summary, latency_ms, energy_wh, co2_g, memory_mb, ppl, throughput_tok_s, model_name, lang_used
Keeps: baseline/optimized mode, low-rank+pruning+INT8, CodeCarbon & psutil metrics.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass
from math import isfinite
from typing import Dict, Optional, Tuple, List

from flask import Flask, request, jsonify, render_template

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

# Optional deps (graceful fallbacks)
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

try:
    from codecarbon import EmissionsTracker  # type: ignore
except Exception:
    EmissionsTracker = None  # type: ignore

# Prefer TorchAO for quantization if available, else torch.ao.quantization
_HAS_TORCHAO = False
try:
    from torchao.quantization import quantize_ as ao_quantize_
    from torchao.quantization import dtype as ao_dtype
    _HAS_TORCHAO = True
except Exception:
    _HAS_TORCHAO = False

try:
    from torch.ao.quantization import quantize_dynamic as pt_quantize_dynamic  # type: ignore
except Exception:
    pt_quantize_dynamic = None  # type: ignore


# ---------------- Config ----------------

DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "EleutherAI/pythia-70m-deduped")
MAX_NEW_TOKENS     = int(os.getenv("MAX_NEW_TOKENS", "48"))
CPU_THREADS        = int(os.getenv("CPU_THREADS", str(max(1, (os.cpu_count() or 2) // 2))))

# Low-rank + pruning
LR_MAX_LAYERS      = int(os.getenv("LR_MAX_LAYERS", "6"))
LR_RANK_CAP        = int(os.getenv("LR_RANK_CAP", "32"))
LR_RANK_FRAC       = float(os.getenv("LR_RANK_FRAC", "0.10"))
PRUNE_AMOUNT       = float(os.getenv("PRUNE_AMOUNT", "0.10"))

# Optional preload
PRELOAD_BASELINE   = os.getenv("PRELOAD_BASELINE", "1") == "1"
PRELOAD_OPTIMIZED  = os.getenv("PRELOAD_OPTIMIZED", "1") == "1"


# ---------------- Utils ----------------

_WORD_RE = r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'-]+"


def _finite_or_none(x, decimals: int = 2) -> Optional[float]:
    """Round to 'decimals' if finite, else None (JSON-safe)."""
    try:
        v = float(x)
        if isfinite(v):
            return round(v, decimals)
    except Exception:
        pass
    return None


def _sanitize_metrics(d: Dict) -> Dict:
    """Ensure JSON-safe metrics (replace non-finite numbers by None)."""
    for k, v in list(d.items()):
        if isinstance(v, (int, float)):
            try:
                if not isfinite(float(v)):
                    d[k] = None
            except Exception:
                d[k] = None
    return d


def _skip_name(name: str) -> bool:
    """Do not touch embeddings/LayerNorm/lm_head with LR/pruning/quantization."""
    n = name.lower()
    banned = ("lm_head", "embed", "embedding", "tok_embeddings",
              "word_embeddings", "wte", "wpe", "layernorm", "ln_f")
    return any(b in n for b in banned)


def _clamp_10_15_words(text: str) -> str:
    """Force 10–15 words without truncating mid-word (pad last token if <10)."""
    words = re.findall(_WORD_RE, text, flags=re.UNICODE)
    if not words:
        return "Summary unavailable"
    if len(words) > 15:
        words = words[:15]
    elif len(words) < 10:
        last = words[-1]
        while len(words) < 10:
            words.append(last)
    return " ".join(words)


def _prompt_for_lang(input_text: str, lang: str) -> str:
    lang = (lang or "fr").lower()
    if lang.startswith("en"):
        return ("Summarize the following text in clear English, 10 to 15 words, "
                "no prefix, concise and informative.\n"
                f"Text: {input_text}\nSummary:")
    else:
        return ("Résume le texte suivant en français clair, 10 à 15 mots, "
                "sans préfixe, concis et informatif.\n"
                f"Texte : {input_text}\nRésumé:")


def _looks_degenerate(s: str) -> bool:
    toks = s.lower().split()
    if len(toks) < 5:
        return True
    from collections import Counter
    mc = Counter(toks).most_common(1)[0][1]
    return mc >= max(3, len(toks) // 3)


# ---------------- Quantization ----------------

def apply_dynamic_int8(model: nn.Module) -> nn.Module:
    """Try dynamic INT8 quantization (TorchAO first, else torch.ao.quantization)."""
    if _HAS_TORCHAO:
        try:
            ao_quantize_(model, weights=ao_dtype.INT8, activations=ao_dtype.INT8, modules_to_exclude=())
            return model
        except Exception:
            warnings.warn("TorchAO quantization failed; trying torch.ao.quantization.", RuntimeWarning)

    if pt_quantize_dynamic is not None:
        try:
            return pt_quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except Exception:
            warnings.warn("torch.ao.quantization.quantize_dynamic failed; using original model.", RuntimeWarning)

    return model


# ---------------- Low-rank (SVD) + Pruning ----------------

def _parent_module(model: nn.Module, dotted: str) -> Optional[nn.Module]:
    parent = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None
        parent = getattr(parent, p)
    return parent


def _svd_low_rank_linear(linear: nn.Linear, rank: int) -> nn.Sequential:
    W = linear.weight.detach().to(torch.float32).cpu()  # [out, in]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    r = max(1, min(rank, U.shape[1], S.shape[0], Vh.shape[0]))

    Ur = U[:, :r]
    Sr = torch.diag(S[:r])
    Vhr = Vh[:r, :]

    lin1 = nn.Linear(linear.in_features, r, bias=False)                 # B
    lin2 = nn.Linear(r, linear.out_features, bias=linear.bias is not None)  # A

    with torch.no_grad():
        lin1.weight.copy_(Vhr)
        lin2.weight.copy_((Ur @ Sr).t())
        if linear.bias is not None and lin2.bias is not None:
            lin2.bias.copy_(linear.bias.detach().cpu())

    return nn.Sequential(lin1, lin2)


def _apply_low_rank(model: nn.Module,
                    max_layers: int = LR_MAX_LAYERS,
                    rank_cap: int = LR_RANK_CAP,
                    rank_frac: float = LR_RANK_FRAC) -> None:
    linear_infos: List[Tuple[str, nn.Linear, int]] = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not _skip_name(name):
            linear_infos.append((name, m, m.weight.numel()))
    if not linear_infos:
        return
    linear_infos.sort(key=lambda x: x[2], reverse=True)
    for name, m, _ in linear_infos[:max_layers]:
        try:
            r = int(min(rank_cap, max(1, rank_frac * min(m.in_features, m.out_features))))
            parent = _parent_module(model, name)
            if parent is None:
                continue
            setattr(parent, name.split(".")[-1], _svd_low_rank_linear(m, r))
        except Exception:
            continue


def _apply_pruning(model: nn.Module, amount: float = PRUNE_AMOUNT) -> None:
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and not _skip_name(name):
            try:
                prune.l1_unstructured(m, name="weight", amount=amount)
                prune.remove(m, "weight")
            except Exception:
                continue


# ---------------- Model store ----------------

@dataclass
class ModelBundle:
    tok: AutoTokenizer
    model: AutoModelForCausalLM
    device: torch.device
    name: str
    optimized: bool


class ModelStore:
    _bundle_baseline: Optional[ModelBundle] = None
    _bundle_opt: Optional[ModelBundle] = None

    @classmethod
    def is_loaded(cls, optimized: bool) -> bool:
        return (cls._bundle_opt is not None) if optimized else (cls._bundle_baseline is not None)

    @classmethod
    def load(cls, optimized: bool) -> ModelBundle:
        existing = cls._bundle_opt if optimized else cls._bundle_baseline
        if existing is not None:
            return existing

        torch.set_num_threads(CPU_THREADS)
        device = torch.device("cpu")

        tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        if tok.pad_token_id is None:
            if tok.eos_token_id is None:
                tok.add_special_tokens({"pad_token": "<|pad|>"})
            else:
                tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_NAME,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        if tok.pad_token_id is not None and tok.pad_token_id >= model.config.vocab_size:
            model.resize_token_embeddings(len(tok))

        if optimized:
            _apply_pruning(model, amount=PRUNE_AMOUNT)
            _apply_low_rank(model, max_layers=LR_MAX_LAYERS, rank_cap=LR_RANK_CAP, rank_frac=LR_RANK_FRAC)
            model = apply_dynamic_int8(model)

        model.to(device)
        model.eval()

        bundle = ModelBundle(tok=tok, model=model, device=device, name=DEFAULT_MODEL_NAME, optimized=optimized)
        if optimized:
            cls._bundle_opt = bundle
        else:
            cls._bundle_baseline = bundle
        return bundle

    @classmethod
    def get(cls, optimized: bool) -> ModelBundle:
        return cls.load(optimized=optimized)


# ---------------- Flask app ----------------

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['JSONIFY_MIMETYPE'] = 'application/json; charset=utf-8'
hf_logging.set_verbosity_error()


@app.route("/", methods=["GET"])
def index1():
    try:
        return render_template("index1.html")
    except Exception:
        return "<h3>Summarization API is running.</h3>", 200


@app.route("/health", methods=["GET"])
def health():
    info = {"status": "ok", "device": "cpu"}
    info["baseline_loaded"] = ModelStore.is_loaded(False)
    info["optimized_loaded"] = ModelStore.is_loaded(True)
    if ModelStore.is_loaded(False):
        info["baseline_model"] = ModelStore.get(False).name
    if ModelStore.is_loaded(True):
        info["optimized_model"] = ModelStore.get(True).name
    return jsonify(info), 200


# ---------------- Inference & metrics ----------------

def _generate_summary(text: str, bundle: ModelBundle, lang: str = "fr") -> Tuple[str, int]:
    tok, model, device = bundle.tok, bundle.model, bundle.device
    prompt = _prompt_for_lang(text, lang)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    inp_len = int(enc["input_ids"].shape[-1])

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.12 if bundle.optimized else 1.10,
        no_repeat_ngram_size=2,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    gen_ids = out[0][inp_len:] if out.shape[-1] > inp_len else out[0]
    gen_tokens = int(gen_ids.shape[-1])
    dec = tok.decode(gen_ids, skip_special_tokens=True)

    parts = re.split(r"(?:Summary|Résumé)\s*:\s*", dec, flags=re.IGNORECASE)
    candidate = parts[-1].strip() if len(parts) > 1 else dec.strip()
    candidate = candidate.split("\n")[0].strip()
    candidate = re.sub(r"\s+", " ", candidate)

    summary = _clamp_10_15_words(candidate)

    # Simple anti-degeneration retry (optimized only)
    if bundle.optimized and _looks_degenerate(summary):
        retry_kwargs = dict(gen_kwargs)
        retry_kwargs["temperature"] = 0.90
        with torch.no_grad():
            out2 = model.generate(**enc, **retry_kwargs)
        gen_ids2 = out2[0][inp_len:] if out2.shape[-1] > inp_len else out2[0]
        gen_tokens = int(gen_ids2.shape[-1])
        dec2 = tok.decode(gen_ids2, skip_special_tokens=True)
        parts2 = re.split(r"(?:Summary|Résumé)\s*:\s*", dec2, flags=re.IGNORECASE)
        cand2 = parts2[-1].strip() if len(parts2) > 1 else dec2.strip()
        cand2 = cand2.split("\n")[0].strip()
        cand2 = re.sub(r"\s+", " ", cand2)
        summary = _clamp_10_15_words(cand2)

    return summary, gen_tokens


def _proxy_perplexity(summary_text: str, bundle: ModelBundle) -> float:
    tok, model, device = bundle.tok, bundle.model, bundle.device
    with torch.no_grad():
        enc = tok(summary_text, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.detach().cpu())
        try:
            ppl = float(torch.exp(torch.tensor(loss)).item())
        except Exception:
            ppl = 0.0
    return round(ppl, 3)


def _stop_codecarbon_tracker(tracker) -> Tuple[float, float]:
    energy_wh = 0.0
    co2_g = 0.0
    try:
        maybe_kg = tracker.stop()
        if isinstance(maybe_kg, (int, float)) and isfinite(float(maybe_kg)):
            co2_g = float(maybe_kg) * 1000.0
        e = getattr(tracker, "final_emissions_data", None) or getattr(tracker, "_emissions_data", None)
        if e is not None:
            kwh = float(getattr(e, "energy_consumed", 0.0) or 0.0)
            energy_wh = kwh * 1000.0
            kg = float(getattr(e, "emissions", 0.0) or 0.0)
            if kg > 0.0 and co2_g == 0.0:
                co2_g = kg * 1000.0
    except Exception:
        pass
    return energy_wh, co2_g


@app.route("/summarize", methods=["POST"])
def summarize():
    """
    POST /summarize
    JSON: { "text": "...", "optimized": false, "lang": "fr|en" }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    text = (payload.get("text") or "").strip()
    use_opt = bool(payload.get("optimized", False))
    lang = str(payload.get("lang", "fr"))

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    # Load model bundle
    bundle = ModelStore.get(optimized=use_opt)

    # Memory before (MB)
    mem_before = None
    if psutil:
        try:
            mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            mem_before = None

    # CodeCarbon tracker
    tracker = None
    if EmissionsTracker is not None:
        try:
            tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False, log_level="error")
            tracker.start()
        except Exception:
            tracker = None

    # Inference + metrics
    t0 = time.perf_counter()
    summary, gen_tokens = _generate_summary(text, bundle=bundle, lang=lang)
    ppl = _proxy_perplexity(summary, bundle=bundle)
    dt = max(time.perf_counter() - t0, 1e-9)

    energy_wh = 0.0
    co2_g = 0.0
    if tracker is not None:
        e_wh, co2 = _stop_codecarbon_tracker(tracker)
        energy_wh, co2_g = e_wh, co2

    memory_mb = None
    if psutil and mem_before is not None:
        try:
            mem_after = psutil.Process().memory_info().rss / (1024 * 1024)
            memory_mb = max(0.0, mem_after - mem_before)
        except Exception:
            memory_mb = None

    throughput_tok_s = (gen_tokens / dt) if dt > 0 else 0.0

    resp = {
        "summary": summary,
        "energy_wh": _finite_or_none(energy_wh, 4),
        "co2_g": _finite_or_none(co2_g, 4),
        "latency_ms": _finite_or_none(dt * 1000.0, 0),
        "memory_mb": _finite_or_none(memory_mb, 2),
        "throughput_tok_s": _finite_or_none(throughput_tok_s, 2),
        "ppl": _finite_or_none(ppl, 2),
        "model_name": bundle.name + (" - Optimized" if bundle.optimized else " - Baseline"),
        "lang_used": ("en" if lang.lower().startswith("en") else "fr"),
    }
    return jsonify(_sanitize_metrics(resp)), 200


# ---------------- Preload (optional) ----------------

torch.set_num_threads(CPU_THREADS)

if PRELOAD_BASELINE:
    try:
        ModelStore.load(optimized=False)
    except Exception as e:
        print(f"[WARN] Baseline preload failed: {e}")

if PRELOAD_OPTIMIZED:
    try:
        ModelStore.load(optimized=True)
    except Exception as e:
        print(f"[WARN] Optimized preload failed: {e}")


if __name__ == "__main__":
    # Prefer gunicorn in prod; debug=False for realistic metrics.
    app.run(host="127.0.0.1", port=5000, debug=False)
