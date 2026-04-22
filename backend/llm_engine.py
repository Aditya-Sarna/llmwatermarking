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


# Instruction-tuned models — use their chat template; base models get the
# legacy completion-style conversion.
_INSTRUCTION_MODELS = {
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
}


def _format_prompt(prompt: str, model_name: str, tokenizer) -> str:
    """Return a prompt string ready for the given model."""
    if model_name in _INSTRUCTION_MODELS:
        # Use the tokenizer's built-in chat template when available
        if getattr(tokenizer, 'chat_template', None):
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        # Fallback for phi-2 which may lack a template
        return f"Instruct: {prompt.strip()}\nOutput:"
    return _to_completion_prompt(prompt)


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

    completion_prompt = _format_prompt(prompt, model_name, tokenizer)
    input_ids = tokenizer(completion_prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    last_prompt_token = int(input_ids[0, -1].item())

    # Grid size based on max_new_tokens (no dry run needed)
    rows, cols = compute_grid_size(max_new_tokens)
    pattern = binarize_image(watermark_image, rows, cols)
    pattern_bits = pattern_to_bits(pattern)

    prompt_tokens = input_ids[0].tolist()
    processor = PatternWatermarkProcessor(wm, pattern_bits, prompt_tokens)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[processor],
        )

    generated_ids = output[0, prompt_len:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    n_actual = len(generated_ids)
    pattern_bits_used = pattern_bits[:n_actual]

    vis_rows, vis_cols = compute_grid_size(n_actual)
    target_grid = pattern_to_grid(pattern_bits_used, vis_rows, vis_cols)

    return {
        "generated_text": text,
        "generated_ids": generated_ids,
        "prompt_tokens": prompt_tokens,
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
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen = out[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(gen, skip_special_tokens=True), gen


def detect_watermark(
    watermark_image: Optional[Image.Image] = None,
    model_name: str = "gpt2",
    secret_key: str = "secret",
    gamma: float = 0.5,
    tau: float = 4.0,
    generated_ids: Optional[List[int]] = None,
    prompt_tokens: Optional[List[int]] = None,
    rows: int = 0,
    cols: int = 0,
    pattern_bits: Optional[List[int]] = None,
) -> Dict:
    """
    Detect watermark using the KGW z-score as the primary statistic.
    tau is the z-score threshold (not a fraction). Under H0 (no watermark),
    z ~ N(0,1) so tau=4.0 gives a false positive rate of ~0.003%.
    Under H1 (watermarked with delta=2.0, 100+ tokens), z >> 4 reliably.
    """
    _, model = load_model(model_name)
    vocab_size = model.config.vocab_size
    wm = PatternWatermark(secret_key=secret_key, gamma=gamma)

    if generated_ids is None or not generated_ids:
        return {"error": "No generated token IDs provided."}

    if prompt_tokens is None:
        prompt_tokens = []

    n = len(generated_ids)

    if pattern_bits is None:
        if rows == 0 or cols == 0:
            rows, cols = compute_grid_size(n)
        full_pattern = pattern_to_bits(binarize_image(watermark_image, rows, cols))
    else:
        full_pattern = list(pattern_bits)
        if rows == 0 or cols == 0:
            sq = int(len(full_pattern) ** 0.5)
            rows = cols = sq

    pattern_bits_cmp = full_pattern[:n]

    recovered_bits = wm.recover_bits(generated_ids, prompt_tokens, vocab_size)

    # Primary detection statistic: pattern-match z-score.
    # Under H0: matches ~ Binomial(n, 0.5) → z ~ N(0,1).
    # Under H1: matches >> n/2 because we biased toward the pattern bit.
    # The standard KGW green-count z-score is INVALID for our scheme because
    # we bias green or red depending on pattern_bit; for a balanced pattern,
    # green-count is ~n/2 even under H1, giving z ≈ 0.
    z = wm.z_score_match(recovered_bits, pattern_bits_cmp)

    # Bit-match accuracy (same info as z, expressed as a percentage)
    min_len = min(len(pattern_bits_cmp), len(recovered_bits))
    bit_acc = sum(a == b for a, b in zip(pattern_bits_cmp[:min_len], recovered_bits[:min_len])) / max(min_len, 1)

    # LCS kept for reference only
    ratio = PatternWatermark.lcs_ratio(pattern_bits_cmp, recovered_bits)

    vis_rows, vis_cols = compute_grid_size(n)
    target_grid = pattern_to_grid(pattern_bits_cmp, vis_rows, vis_cols)
    recovered_grid = reconstruct_grid(pattern_bits_cmp, recovered_bits, vis_rows, vis_cols)

    return {
        "is_watermarked": z >= tau,
        "z_score": z,
        "tau": tau,
        "bit_accuracy": bit_acc,
        "lcs_ratio": ratio,
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

    completion_prompt = _format_prompt(prompt, model_name, tokenizer)
    input_ids = tokenizer(completion_prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    last_prompt_token = int(input_ids[0, -1].item())

    rows, cols = compute_grid_size(max_new_tokens)
    pattern = binarize_image(watermark_image, rows, cols)
    pattern_bits = pattern_to_bits(pattern)

    prompt_tokens = input_ids[0].tolist()
    processor = PatternWatermarkProcessor(wm, pattern_bits, prompt_tokens)
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
                repetition_penalty=1.15,
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
        "prompt_tokens": prompt_tokens,
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
