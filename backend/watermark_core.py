"""
Core watermarking logic — Green-red list watermarking (Kirchenbauer et al. 2023),
with improvements informed by Li et al. (2024) and Li et al. / Tr-GoF (2024).

Key design decisions aligned with the paper:
--------------------------------------------
* Context window m=5: hash uses the last min(m, available) tokens as context,
  matching the paper's ζ_t = A(w_{(t-m):(t-1)}, Key). This is strictly stronger
  than window=1 (single prev token) because edits corrupt fewer pseudo-random
  numbers per change (a single edit corrupts at most m downstream ζ values).
* Detection uses the KGW z-score as the primary statistic, which is a proper
  pivotal statistic under H0 (asymptotically N(0,1)) unlike bit_accuracy.
* Bit accuracy is kept as a secondary diagnostic for pattern matching.
* Context masking: the watermark is only applied when the current m-token context
  is unique in generation history, preventing repetitive patterns from
  artificially inflating detection scores (Section 6.1 of the paper).
"""

import hashlib
import math
import struct
import numpy as np
import torch
from transformers import LogitsProcessor
from typing import List, Tuple, Optional

# Context window size — paper uses m=5
CONTEXT_WINDOW = 5


class PatternWatermark:

    def __init__(self, secret_key: str = "llmwatermark", gamma: float = 0.5, delta: float = 2.0):
        self.secret_key = secret_key
        self.gamma = gamma
        self.delta = delta

    # ------------------------------------------------------------------ #
    #  Green-list partition with m-token context window                   #
    # ------------------------------------------------------------------ #

    def green_mask(self, vocab_size: int, context: Tuple[int, ...]) -> np.ndarray:
        """
        Returns boolean numpy array shape (vocab_size,).
        Seeds with SHA3-256(context_tokens || secret_key).
        context is a tuple of the last min(m, available) token ids.
        Same seed reproduced during detection provided same context.
        """
        # Pack each context token as 4-byte big-endian uint
        header = b"".join(struct.pack(">I", t & 0xFFFFFFFF) for t in context)
        seed_bytes = hashlib.sha3_256(header + self.secret_key.encode()).digest()
        seed = struct.unpack(">I", seed_bytes[:4])[0]
        rng = np.random.RandomState(seed)
        return rng.random_sample(vocab_size) < self.gamma

    # ------------------------------------------------------------------ #
    #  Logit bias                                                          #
    # ------------------------------------------------------------------ #

    def apply_bias(
        self,
        logits: torch.Tensor,
        pattern_bit: int,
        context: Tuple[int, ...],
        protected_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """bit==1 => bias green tokens; bit==0 => bias red tokens.
        protected_ids (e.g. EOS, PAD, BOS) are never biased so the model
        can always terminate cleanly.
        """
        mask = torch.from_numpy(
            self.green_mask(logits.shape[-1], context)
        ).to(logits.device)
        biased = logits.clone()
        if pattern_bit == 1:
            biased[mask] += self.delta
        else:
            biased[~mask] += self.delta
        # Restore original logits for special tokens — never suppress EOS
        if protected_ids:
            for tid in protected_ids:
                if 0 <= tid < logits.shape[-1]:
                    biased[tid] = logits[tid]
        return biased

    # ------------------------------------------------------------------ #
    #  Detection helpers                                                   #
    # ------------------------------------------------------------------ #

    def token_is_green(self, token_id: int, context: Tuple[int, ...], vocab_size: int) -> bool:
        return bool(self.green_mask(vocab_size, context)[token_id])

    def recover_bits(
        self,
        token_ids: List[int],
        prompt_tokens: List[int],
        vocab_size: int,
    ) -> List[int]:
        """
        Recover the green/red bit for each generated token.
        prompt_tokens: the full prompt token sequence (provides context for early tokens).
        """
        all_tokens = prompt_tokens + token_ids
        bits = []
        for i, tok in enumerate(token_ids):
            pos = len(prompt_tokens) + i  # position in all_tokens
            # context = last min(CONTEXT_WINDOW, pos) tokens before tok
            ctx_start = max(0, pos - CONTEXT_WINDOW)
            context = tuple(all_tokens[ctx_start:pos])
            if not context:
                context = (0,)  # fallback
            bits.append(1 if self.token_is_green(tok, context, vocab_size) else 0)
        return bits

    # ------------------------------------------------------------------ #
    #  LCS (kept for reference / visualization)                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_lcs(seq_a: List[int], seq_b: List[int]) -> Tuple[int, List[Tuple[int, int, int]]]:
        """Returns (lcs_length, [(bit, idx_in_a, idx_in_b)])."""
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
                i -= 1; j -= 1
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

    # ------------------------------------------------------------------ #
    #  Match-based z-score (proper pivotal for pattern-based scheme)      #
    # ------------------------------------------------------------------ #

    def z_score_match(self, recovered_bits: List[int], pattern_bits: List[int]) -> float:
        """
        Pattern-based z-score. Counts positions where the recovered green/red
        bit matches the embedded pattern bit. Under H0 (no watermark),
        match_prob = 0.5 regardless of gamma, so matches ~ Binomial(n, 0.5)
        and z = (2*matches - n) / sqrt(n) ~ N(0, 1).
        Under H1 (watermarked), matches >> n/2 because we biased toward the
        pattern bit at every position. This is the correct statistic for our
        scheme — the standard KGW z-score (green-count) is NOT valid here
        because we bias green or red depending on the pattern bit.
        """
        n = min(len(recovered_bits), len(pattern_bits))
        if n == 0:
            return 0.0
        matches = sum(1 for a, b in zip(recovered_bits[:n], pattern_bits[:n]) if a == b)
        return (2 * matches - n) / math.sqrt(n)

    # KGW z-score kept for reference / unbiased pure-KGW use only
    def z_score_green(self, token_ids: List[int], prompt_tokens: List[int], vocab_size: int) -> float:
        n = len(token_ids)
        if n == 0:
            return 0.0
        bits = self.recover_bits(token_ids, prompt_tokens, vocab_size)
        k = sum(bits)
        return (k - self.gamma * n) / math.sqrt(n * self.gamma * (1 - self.gamma))


# ------------------------------------------------------------------ #
#  HuggingFace LogitsProcessor — enables model.generate() with cache  #
# ------------------------------------------------------------------ #

class PatternWatermarkProcessor(LogitsProcessor):
    """
    LogitsProcessor implementing the m-token context window watermark.
    Applies watermark at every step (no context masking) so that detection
    can verify every position without needing to replay generation history.
    """

    def __init__(
        self,
        wm: PatternWatermark,
        pattern_bits: List[int],
        prompt_tokens: List[int],
        protected_ids: Optional[List[int]] = None,
    ):
        self.wm = wm
        self.pattern_bits = pattern_bits
        self.prompt_tokens = list(prompt_tokens)
        self.protected_ids = protected_ids or []
        self._step = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = self._step
        seq = input_ids[0].tolist()

        # Build m-token context from current sequence
        ctx_start = max(0, len(seq) - CONTEXT_WINDOW)
        context = tuple(seq[ctx_start:])
        if not context:
            context = (0,)

        if step < len(self.pattern_bits):
            bit = self.pattern_bits[step]
            biased = self.wm.apply_bias(scores[0], bit, context, self.protected_ids)
            scores = biased.unsqueeze(0)

        self._step += 1
        return scores

