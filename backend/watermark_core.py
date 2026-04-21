"""
Core watermarking logic — Visual Pattern-Based Watermarking (Bashir 2025).
Ported from the original Streamlit reference implementation.
"""

import hashlib
import math
import struct
import numpy as np
import torch
from transformers import LogitsProcessor
from typing import List, Tuple


class PatternWatermark:
    def __init__(self, secret_key: str = "llmwatermark", gamma: float = 0.5, delta: float = 0.3):
        self.secret_key = secret_key
        self.gamma = gamma
        self.delta = delta

    def green_mask(self, vocab_size: int, prev_token_id: int, step: int) -> np.ndarray:
        header = struct.pack(">II", prev_token_id & 0xFFFFFFFF, step & 0xFFFFFFFF)
        seed_bytes = hashlib.sha3_256(header + self.secret_key.encode()).digest()
        seed = struct.unpack(">I", seed_bytes[:4])[0]
        rng = np.random.RandomState(seed)
        return rng.random_sample(vocab_size) < self.gamma

    def apply_bias(self, logits: torch.Tensor, pattern_bit: int, prev_token_id: int, step: int) -> torch.Tensor:
        mask = torch.from_numpy(self.green_mask(logits.shape[-1], prev_token_id, step)).to(logits.device)
        biased = logits.clone()
        if pattern_bit == 1:
            biased[mask] += self.delta
        else:
            biased[~mask] += self.delta
        return biased

    def token_bit(self, token_id: int, prev_token_id: int, step: int, vocab_size: int) -> int:
        return 1 if self.green_mask(vocab_size, prev_token_id, step)[token_id] else 0

    def recover_bits(self, token_ids: List[int], last_prompt_token: int, vocab_size: int) -> List[int]:
        bits = []
        for step, tok in enumerate(token_ids):
            prev = last_prompt_token if step == 0 else token_ids[step - 1]
            bits.append(self.token_bit(tok, prev, step, vocab_size))
        return bits

    @staticmethod
    def compute_lcs(seq_a: List[int], seq_b: List[int]) -> Tuple[int, List[Tuple[int, int, int]]]:
        m, n = len(seq_a), len(seq_b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        pairs = []
        i, j = m, n
        while i > 0 and j > 0:
            if seq_a[i - 1] == seq_b[j - 1]:
                pairs.append((seq_a[i - 1], i - 1, j - 1))
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        pairs.reverse()
        return dp[m][n], pairs

    @staticmethod
    def lcs_ratio(pattern: List[int], recovered: List[int]) -> float:
        if not pattern:
            return 0.0
        length, _ = PatternWatermark.compute_lcs(pattern, recovered)
        return length / len(pattern)

    def z_score(self, token_ids: List[int], last_prompt_token: int, vocab_size: int) -> float:
        n = len(token_ids)
        if n == 0:
            return 0.0
        k = sum(self.recover_bits(token_ids, last_prompt_token, vocab_size))
        return (k - self.gamma * n) / math.sqrt(n * self.gamma * (1 - self.gamma))


class PatternWatermarkProcessor(LogitsProcessor):
    def __init__(self, wm: PatternWatermark, pattern_bits: List[int], prompt_length: int):
        self.wm = wm
        self.pattern_bits = pattern_bits
        self.prompt_length = prompt_length
        self._step = 0
        self.generated_ids: List[int] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq = input_ids[0]
        prev_token = int(seq[-1].item())
        step = self._step
        if step < len(self.pattern_bits):
            bit = self.pattern_bits[step]
            biased = self.wm.apply_bias(scores[0], bit, prev_token, step)
            scores = biased.unsqueeze(0)
        self._step += 1
        return scores
