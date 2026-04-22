# LLM Watermark Studio

Pattern-based LLM text watermarking demo implementing Li et al. (2024) Tr-GoF scheme, with statistical z-score detection and a React + FastAPI UI.

## Deploy

### Backend → Hugging Face Spaces (free)

1. Create a new Space at https://huggingface.co/spaces
   - **Owner**: your HF username
   - **Space name**: `llmwatermarking`
   - **SDK**: Docker
   - **Visibility**: Public

2. In the Space **Files** tab, upload everything from the `backend/` folder (including the `Dockerfile`).

3. Add a **Space secret** (Settings → Variables and secrets):
   - `CORS_ORIGINS` = `https://<your-netlify-site>.netlify.app`

4. Your API will be live at:
   `https://<hf-username>-llmwatermarking.hf.space`

### Frontend → Netlify

1. Connect this GitHub repo at https://netlify.com — `netlify.toml` handles the build.
2. Add environment variable in Netlify → Site config → Env vars:
   - `REACT_APP_BACKEND_URL` = `https://<hf-username>-llmwatermarking.hf.space`
3. Trigger a redeploy.

## Stack

- **Frontend**: React, TailwindCSS, craco — hosted on Netlify
- **Backend**: FastAPI, PyTorch (CPU), Transformers — hosted on HF Spaces (Docker)
- **Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (135M params, ~270MB)
- **Algorithm**: Pattern watermarking with m=5 context window, SHA3-256 hashing, z-score detection
