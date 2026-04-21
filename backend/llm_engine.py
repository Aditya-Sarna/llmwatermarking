"""
LLM engine — fast generation via model.generate() + LogitsProcessor (KV cache).
"""

import torch
import numpy as np
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from typing import List, Dict, Optional, Tuple, Generator

from watermark_core import PatternWatermark, PatternWatermarkProcessor
from image_utils import (
    compute_grid_size,
    binarize_image,
    pattern_to_bits,
    pattern_to_grid,
    reconstruct_grid,
)
from PIL import Image

_MODEL_CACHE: Dict = {}


def load_model(model_name: str = "gpt2"):
    if model_name not in _MODEL_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        _MODEL_CACHE[model_name] = (tokenizer, model)
    return _MODEL_CACHE[model_name]


def _to_completion_prompt(prompt: str) -> str:
    """
    GPT-2 / OPT are base completion models — they continue text, not answer questions.
    Convert question-style prompts into natural completion starters that small models
    can follow without going off the rails.
    """
    import re
    p = prompt.strip()
    pl = p.lower().rstrip("?. ")

    # "What is X" / "What are X" → "X is" / "X are"
    m = re.match(r'^what\s+(?:is|are)\s+(.+)$', pl, re.I)
    if m:
        subj = m.group(1).strip()
        verb = "are" if "are" in pl.split()[1] else "is"
        return f"{subj.capitalize()} {verb}"

    # "Who is X" → "X is"
    m = re.match(r'^who\s+(?:is|was|were)\s+(.+)$', pl, re.I)
    if m:
        return f"{m.group(1).strip().capitalize()} is"

    # "How does / How do X work" → "X works by"
    m = re.match(r'^how\s+(?:does|do)\s+(.+?)\s+work', pl, re.I)
    if m:
        return f"{m.group(1).strip().capitalize()} works by"

    # "Why is / Why are X" → "X is because"
    m = re.match(r'^why\s+(?:is|are)\s+(.+)$', pl, re.I)
    if m:
        return f"{m.group(1).strip().capitalize()} is"

    # "Tell me about / Explain / Describe X" → "X is"
    m = re.match(r'^(?:tell me (?:about)?|explain|describe|define)\s+(.+)$', pl, re.I)
    if m:
        return f"{m.group(1).strip().capitalize()} is"

    # Generic question → strip ? and return as-is for the model to continue
    if p.endswith("?") or p.lower().startswith(("is ", "are ", "was ", "were ", "does ", "do ", "did ", "can ", "could ", "will ", "would ")):
        return p.rstrip("?").strip()

    return p


def generate_watermarked(
    prompt: str,
    watermark_image: Image.Image,
    model_name: str = "gpt2",
    max_new_tokens: int = 200,
    secret_key: str = "secret",
    gamma: float = 0.5,
    delta: float = 2.0,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Dict:
    tokenizer, model = load_model(model_name)
    wm = PatternWatermark(secret_key=secret_key, gamma=gamma, delta=delta)
    vocab_size = model.config.vocab_size

    completion_prompt = _to_completion_prompt(prompt)
    input_ids = tokenizer(completion_prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    last_prompt_token = int(input_ids[0, -1].item())

    # Grid size based on max_new_tokens (no dry run needed)
    rows, cols = compute_grid_size(max_new_tokens)
    pattern = binarize_image(watermark_image, rows, cols)
    pattern_bits = pattern_to_bits(pattern)

    processor = PatternWatermarkProcessor(wm, pattern_bits, prompt_len)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[processor],
        )

    generated_ids = output[0, prompt_len:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # CRITICAL: truncate pattern_bits to the actual number of tokens generated.
    # If the model stopped early (EOS), only the first n bits were embedded.
    # Without this, LCS ratio = at_most_n / max_new_tokens < tau even for
    # perfectly watermarked text, causing false negatives.
    n_actual = len(generated_ids)
    pattern_bits_used = pattern_bits[:n_actual]

    vis_rows, vis_cols = compute_grid_size(n_actual)
    target_grid = pattern_to_grid(pattern_bits_used, vis_rows, vis_cols)

    return {
        "generated_text": text,
        "generated_ids": generated_ids,
        "last_prompt_token": last_prompt_token,
        "pattern_bits": pattern_bits_used,   # only the bits actually embedded
        "rows": rows,
        "cols": cols,
        "model_name": model_name,
        "vocab_size": vocab_size,
        "secret_key": secret_key,
        "gamma": gamma,
        "delta": delta,
        "target_grid": target_grid,
    }


def generate_plain(
    prompt: str,
    model_name: str = "gpt2",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Tuple[str, List[int]]:
    tokenizer, model = load_model(model_name)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(gen, skip_special_tokens=True), gen


def detect_watermark(
    watermark_image: Optional[Image.Image] = None,
    model_name: str = "gpt2",
    secret_key: str = "secret",
    gamma: float = 0.5,
    tau: float = 0.75,
    generated_ids: Optional[List[int]] = None,
    last_prompt_token: int = 0,
    rows: int = 0,
    cols: int = 0,
    pattern_bits: Optional[List[int]] = None,
) -> Dict:
    """Detect watermark using stored generated_ids from the embed step."""
    _, model = load_model(model_name)
    vocab_size = model.config.vocab_size
    wm = PatternWatermark(secret_key=secret_key, gamma=gamma)

    if generated_ids is None or not generated_ids:
        return {"error": "No generated token IDs provided."}

    n = len(generated_ids)

    # Recompute or use stored pattern_bits.
    # Always truncate to len(generated_ids) so the LCS denominator equals
    # the number of tokens where the watermark was actually embedded.
    if pattern_bits is None:
        if rows == 0 or cols == 0:
            rows, cols = compute_grid_size(n)
        full_pattern = pattern_to_bits(binarize_image(watermark_image, rows, cols))
    else:
        full_pattern = list(pattern_bits)
        if rows == 0 or cols == 0:
            sq = int(len(full_pattern) ** 0.5)
            rows = cols = sq

    # Use only bits that correspond to tokens that were generated
    pattern_bits_cmp = full_pattern[:n]

    recovered_bits = wm.recover_bits(generated_ids, last_prompt_token, vocab_size)

    # LCS ratio over the covered prefix only
    ratio = PatternWatermark.lcs_ratio(pattern_bits_cmp, recovered_bits)
    z = wm.z_score(generated_ids, last_prompt_token, vocab_size)

    # Bit-level accuracy (direct match, no alignment)
    min_len = min(len(pattern_bits_cmp), len(recovered_bits))
    bit_acc = sum(a == b for a, b in zip(pattern_bits_cmp[:min_len], recovered_bits[:min_len])) / max(min_len, 1)

    # Visualise using the covered prefix reshaped to a grid
    vis_rows, vis_cols = compute_grid_size(n)
    target_grid = pattern_to_grid(pattern_bits_cmp, vis_rows, vis_cols)
    recovered_grid = reconstruct_grid(pattern_bits_cmp, recovered_bits, vis_rows, vis_cols)

    # Use bit_accuracy for the detection decision, NOT lcs_ratio.
    # LCS of two random binary seqs ≈ 0.788*n, so lcs_ratio ≈ 0.788 for ANY text —
    # it cannot distinguish watermarked from non-watermarked.
    # bit_accuracy ≈ 0.5 for random text, ≈ 0.7-0.9 for watermarked text.
    return {
        "is_watermarked": bit_acc >= tau,
        "lcs_ratio": ratio,
        "bit_accuracy": bit_acc,
        "z_score": z,
        "target_grid": target_grid,
        "recovered_grid": recovered_grid,
        "pattern_bits": pattern_bits_cmp,
        "recovered_bits": recovered_bits,
        "bit_matches": sum(a == b for a, b in zip(pattern_bits_cmp[:min_len], recovered_bits[:min_len])),
        "pattern_length": len(pattern_bits_cmp),
        "rows": vis_rows,
        "cols": vis_cols,
        "n_tokens": n,
    }


def generate_watermarked_stream(
    prompt: str,
    watermark_image: Image.Image,
    model_name: str = "gpt2",
    max_new_tokens: int = 200,
    secret_key: str = "secret",
    gamma: float = 0.5,
    delta: float = 2.0,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Generator:
    """
    Synchronous generator for streaming token output.
    Yields (token_text: str, is_done: bool, result: dict|None).
    Each token chunk yields (text, False, None).
    Final value yields (None, True, result_dict).
    """
    tokenizer, model = load_model(model_name)
    wm = PatternWatermark(secret_key=secret_key, gamma=gamma, delta=delta)
    vocab_size = model.config.vocab_size

    completion_prompt = _to_completion_prompt(prompt)
    input_ids = tokenizer(completion_prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    last_prompt_token = int(input_ids[0, -1].item())

    rows, cols = compute_grid_size(max_new_tokens)
    pattern = binarize_image(watermark_image, rows, cols)
    pattern_bits = pattern_to_bits(pattern)

    processor = PatternWatermarkProcessor(wm, pattern_bits, prompt_len)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    output_holder: List = []

    def _gen():
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=[processor],
                streamer=streamer,
            )
        output_holder.append(out)

    thread = Thread(target=_gen, daemon=True)
    thread.start()

    for text_chunk in streamer:
        yield text_chunk, False, None

    thread.join()

    output = output_holder[0]
    generated_ids = output[0, prompt_len:].tolist()
    n_actual = len(generated_ids)
    pattern_bits_used = pattern_bits[:n_actual]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    vis_rows, vis_cols = compute_grid_size(n_actual)
    target_grid = pattern_to_grid(pattern_bits_used, vis_rows, vis_cols)

    yield None, True, {
        "generated_text": text,
        "generated_ids": generated_ids,
        "last_prompt_token": last_prompt_token,
        "pattern_bits": pattern_bits_used,
        "rows": rows,
        "cols": cols,
        "model_name": model_name,
        "vocab_size": vocab_size,
        "secret_key": secret_key,
        "gamma": gamma,
        "delta": delta,
        "target_grid": target_grid,
    }
