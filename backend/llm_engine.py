"""
LLM engine — fast generation via model.generate() + LogitsProcessor (KV cache).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
from PIL import Image

from watermark_core import PatternWatermark, PatternWatermarkProcessor
from image_utils import (
    compute_grid_size,
    binarize_image,
    pattern_to_bits,
    reconstruct_grid,
    pattern_to_grid,
)

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


def generate_watermarked(
    prompt: str,
    watermark_image: Image.Image,
    model_name: str = "gpt2",
    max_new_tokens: int = 120,
    secret_key: str = "llmwatermark",
    gamma: float = 0.5,
    delta: float = 0.3,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Dict:
    tokenizer, model = load_model(model_name)
    wm = PatternWatermark(secret_key=secret_key, gamma=gamma, delta=delta)
    vocab_size = model.config.vocab_size

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    last_prompt_token = int(input_ids[0, -1].item())

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
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[processor],
        )

    generated_ids = output[0, prompt_len:].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "generated_text": text,
        "generated_ids": generated_ids,
        "last_prompt_token": last_prompt_token,
        "pattern_bits": pattern_bits,
        "rows": rows,
        "cols": cols,
        "model_name": model_name,
        "vocab_size": vocab_size,
        "secret_key": secret_key,
        "gamma": gamma,
        "delta": delta,
        "target_grid": pattern_to_grid(pattern_bits, rows, cols),
    }


def detect_watermark(
    model_name: str,
    secret_key: str,
    gamma: float,
    tau: float,
    generated_ids: List[int],
    last_prompt_token: int,
    rows: int,
    cols: int,
    pattern_bits: List[int],
    watermark_image: Optional[Image.Image] = None,
) -> Dict:
    _, model = load_model(model_name)
    vocab_size = model.config.vocab_size
    wm = PatternWatermark(secret_key=secret_key, gamma=gamma)

    if not generated_ids:
        return {"error": "No generated token IDs provided."}

    if not pattern_bits:
        if rows == 0 or cols == 0:
            rows, cols = compute_grid_size(len(generated_ids))
        if watermark_image is not None:
            pattern = binarize_image(watermark_image, rows, cols)
            pattern_bits = pattern_to_bits(pattern)
        else:
            return {"error": "No pattern bits or reference image provided."}

    recovered_bits = wm.recover_bits(generated_ids, last_prompt_token, vocab_size)
    ratio = PatternWatermark.lcs_ratio(pattern_bits, recovered_bits)
    z = wm.z_score(generated_ids, last_prompt_token, vocab_size)

    target_grid = pattern_to_grid(pattern_bits, rows, cols)
    recovered_grid = reconstruct_grid(pattern_bits, recovered_bits, rows, cols)

    matches = sum(a == b for a, b in zip(pattern_bits, recovered_bits[: len(pattern_bits)]))
    return {
        "is_watermarked": bool(ratio >= tau),
        "lcs_ratio": float(ratio),
        "z_score": float(z),
        "n_tokens": len(generated_ids),
        "rows": rows,
        "cols": cols,
        "target_grid": target_grid,
        "recovered_grid": recovered_grid,
        "bit_matches": int(matches),
        "pattern_length": len(pattern_bits),
        "recovered_length": len(recovered_bits),
    }
