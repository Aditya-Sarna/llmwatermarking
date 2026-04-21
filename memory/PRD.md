# LLM Watermark Studio — PRD

## Problem Statement (verbatim)
> https://github.com/Aditya-Sarna/llmwatermarking — Understand the context of the project. I want a super modern ui/ux for this ideally like a llm. collapse all tabs into one single interface. use white or cream as background but yes should be modern looking.

## What was built
- Ported the Streamlit reference implementation of "Visual Pattern-Based Watermarking for LLM-Generated Text" (Bashir 2025) to a **React + FastAPI** single-page app.
- Three Streamlit tabs (embed / detect / about) collapsed into **one chat-style interface** (cream background, Cabinet Grotesk + JetBrains Mono, swiss/editorial aesthetic).
- Local HuggingFace inference (watermarking requires direct logit access; hosted APIs cannot expose this).

## Tech
- **Backend** (`/app/backend/`): FastAPI + transformers + torch (CPU). Key files: `server.py`, `llm_engine.py`, `watermark_core.py`, `image_utils.py`.
- **Frontend** (`/app/frontend/src/`): React 19 chat UI. `App.js`, `components/WatermarkUI.jsx`, `components/HowItWorksPanel.jsx`, `index.css`.

## Features
- Chat-style prompt input with pattern picker chips (Square, Checkerboard, Circle, Diamond, Cross, Upload).
- Model selector: Qwen/Qwen2.5-0.5B-Instruct (default), HuggingFaceTB/SmolLM2-360M-Instruct, gpt2, gpt2-medium, facebook/opt-125m.
- Collapsible Advanced drawer: secret key, gamma, delta, tau, temperature, top-p.
- Assistant bubble with metrics pills (tokens / grid / delta / elapsed) and "embedded pattern" preview.
- Inline "Verify watermark" → verdict chip (Watermarked / Not Watermarked), LCS ratio, z-score, bit-match pill, side-by-side target vs recovered grid visualization.
- "How it works" slide-in panel (replaces the old about tab).

## Change log
- **2026-04-21 (MVP)** — Ported core algorithm. Built chat-style UI. Shipped with gpt2 default + δ=0.3.
- **2026-04-21 (bug fix #1)** — User reported (a) "model doesn't answer the question" → added two instruction-tuned models (Qwen2.5-0.5B-Instruct default, SmolLM2-360M-Instruct), wired `tokenizer.apply_chat_template` via `INSTRUCT_MODELS` set + `_build_input_ids()`. Added startup warmup thread so first request isn't blocked by cold download (prevents preview-proxy 502s).
- **2026-04-21 (bug fix #2)** — User reported (b) "recovered watermark doesn't match" → root cause: EOS-triggered early stop left pattern much longer than generated tokens → LCS computed on the full pattern (not the embedded portion) tanked the ratio. Fixed `detect_watermark()` to compare against `pattern_bits[:len(generated_ids)]` and re-grid visualizations to the effective token count. Raised default δ from 0.3 → 1.0 since instruction models are higher-confidence and need stronger bias.

## Verified
- GET /api/watermark/info returns 5 models + 5 patterns ✓
- POST /api/watermark/generate + /detect through the external preview URL: "List three colors of the rainbow." → Qwen answers correctly in ~17s, verdict **WATERMARKED**, LCS 0.75, 9/12 bits.
- Frontend E2E: hero, chips, model selector, advanced popover, How-it-works panel, generate → verify flow all working.

## Known limitations
- Sessions are process-local in-memory (fine for single-pod demo; would need Redis if scaled).
- CPU inference only — responses take ~15–40s depending on token count.
- Smaller generations (<20 tokens) have high LCS variance; increasing `max_tokens` improves reliability.

## Backlog / Next
- P1: Stream tokens to the UI (SSE) so long generations feel interactive.
- P1: Compare against an unwatermarked control generation to show the robustness gap visually.
- P2: Add Emergent-managed auth so individual users can save/share verified responses.
- P2: Export a shareable receipt card (PNG of the verdict card) — perfect for tweeting proofs of origin.

## Potential enhancement idea for the user
Would you like to add a one-click **"Tweet this proof"** button on the verdict card that renders the verdict + pattern-grid as a shareable PNG? It'd turn every successful verification into free distribution for the tool.
