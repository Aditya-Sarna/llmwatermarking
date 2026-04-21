"""Backend API tests for LLM Watermark Studio.

Covers the bug-fix iteration:
  (1) New instruct models listed in /info (Qwen, SmolLM2).
  (2) Qwen chat-template actually answers user questions.
  (3) Default delta=0.6 + detect yields lcs_ratio >= tau (watermarked).
  (4) Backward compat with gpt2.
"""
import os
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://mark-detect-2.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"

# Shared state across the gpt2 generate -> detect flow
_gpt2 = {"session_id": None, "rows": None, "cols": None, "pattern_length": None}
# Shared state across the Qwen generate -> detect flow (real bug-fix verification)
_qwen = {"session_id": None, "generated_text": None}


@pytest.fixture(scope="session")
def api_client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# ---------------- /api/watermark/info ---------------- #
class TestInfo:
    def test_info_returns_new_instruct_models(self, api_client):
        r = api_client.get(f"{API}/watermark/info", timeout=30)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data and "patterns" in data
        # Bug-fix: the two new instruct models must be present
        assert "Qwen/Qwen2.5-0.5B-Instruct" in data["models"], data["models"]
        assert "HuggingFaceTB/SmolLM2-360M-Instruct" in data["models"], data["models"]
        # Back-compat: old base models still listed
        assert "gpt2" in data["models"]
        assert "gpt2-medium" in data["models"]
        assert "facebook/opt-125m" in data["models"]
        # Patterns unchanged
        for p in ["square", "checkerboard", "circle", "diamond", "cross"]:
            assert p in data["patterns"]
        # 5 total models
        assert len(data["models"]) == 5


# ---------------- /api/watermark/generate validation ---------------- #
class TestGenerateValidation:
    def test_invalid_model_returns_400(self, api_client):
        r = api_client.post(f"{API}/watermark/generate", json={
            "prompt": "hello",
            "model_name": "nonexistent-model",
            "max_new_tokens": 40,
            "pattern": "checkerboard",
        }, timeout=30)
        assert r.status_code == 400

    def test_upload_pattern_without_image_returns_400(self, api_client):
        r = api_client.post(f"{API}/watermark/generate", json={
            "prompt": "hello",
            "model_name": "gpt2",
            "max_new_tokens": 40,
            "pattern": "upload",
        }, timeout=30)
        assert r.status_code == 400


# ---------------- Qwen instruct model (answers question + strong watermark) ---------------- #
class TestQwenInstruct:
    def test_qwen_answers_capital_of_france(self, api_client):
        """Qwen chat template must be applied so the model answers 'Paris'."""
        r = api_client.post(f"{API}/watermark/generate", json={
            "prompt": "What is the capital of France?",
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_new_tokens": 80,
            "pattern": "checkerboard",
            "secret_key": "llmwatermark",
            "gamma": 0.5,
            "delta": 0.6,
            "temperature": 0.8,
            "top_p": 0.9,
            "tau": 0.75,
        }, timeout=600)  # first load of ~1GB can take a few minutes
        assert r.status_code == 200, f"Body: {r.text[:800]}"
        data = r.json()
        assert data["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
        gen = data["generated_text"] or ""
        assert len(gen) > 0
        # Key assertion: instruct template worked → answer mentions Paris
        assert "paris" in gen.lower(), f"Expected 'Paris' in answer, got: {gen!r}"
        _qwen["session_id"] = data["session_id"]
        _qwen["generated_text"] = gen

    def test_qwen_detect_watermark_strong(self, api_client):
        """With default delta=0.6 the recovered watermark should exceed tau=0.75."""
        sid = _qwen["session_id"]
        if not sid:
            pytest.skip("Qwen generate did not produce a session_id")
        r = api_client.post(f"{API}/watermark/detect", json={"session_id": sid}, timeout=300)
        assert r.status_code == 200, f"Body: {r.text[:500]}"
        data = r.json()
        assert isinstance(data["lcs_ratio"], float)
        assert isinstance(data["is_watermarked"], bool)
        # Bug-fix assertion: lcs_ratio now reliably >= 0.75
        assert data["lcs_ratio"] >= 0.75, (
            f"lcs_ratio={data['lcs_ratio']:.3f} < 0.75 — watermark too weak. "
            f"z_score={data.get('z_score')}, bit_matches={data.get('bit_matches')}/{data.get('pattern_length')}"
        )
        assert data["is_watermarked"] is True


# ---------------- Back-compat: gpt2 still works ---------------- #
class TestGpt2BackCompat:
    def test_generate_gpt2_checkerboard(self, api_client):
        r = api_client.post(f"{API}/watermark/generate", json={
            "prompt": "The future of AI is",
            "model_name": "gpt2",
            "max_new_tokens": 40,
            "pattern": "checkerboard",
            "secret_key": "llmwatermark",
            "gamma": 0.5,
            "delta": 0.6,
            "temperature": 0.8,
            "top_p": 0.9,
            "tau": 0.75,
        }, timeout=600)
        assert r.status_code == 200, f"Body: {r.text[:500]}"
        data = r.json()
        for k in ["session_id", "generated_text", "token_count", "rows", "cols",
                  "pattern_length", "model_name", "delta", "gamma", "tau",
                  "target_grid", "elapsed_s"]:
            assert k in data, f"missing key: {k}"
        assert data["model_name"] == "gpt2"
        assert data["tau"] == 0.75
        assert data["target_grid"][0][0] in ("0", "1")
        _gpt2["session_id"] = data["session_id"]
        _gpt2["rows"] = data["rows"]
        _gpt2["cols"] = data["cols"]
        _gpt2["pattern_length"] = data["pattern_length"]

    def test_detect_gpt2_session(self, api_client):
        sid = _gpt2["session_id"]
        if not sid:
            pytest.skip("No gpt2 session_id")
        r = api_client.post(f"{API}/watermark/detect", json={"session_id": sid}, timeout=300)
        assert r.status_code == 200, f"Body: {r.text[:500]}"
        data = r.json()
        for k in ["is_watermarked", "lcs_ratio", "z_score", "n_tokens",
                  "rows", "cols", "target_grid", "recovered_grid",
                  "bit_matches", "pattern_length", "recovered_length", "tau"]:
            assert k in data, f"missing key: {k}"
        assert data["rows"] == _gpt2["rows"]
        assert data["cols"] == _gpt2["cols"]
        assert data["pattern_length"] == _gpt2["pattern_length"]
        assert data["target_grid"][0][0] in ("0", "1")
        assert data["recovered_grid"][0][0] in ("0", "1", "-")

    def test_detect_with_invalid_session_returns_404(self, api_client):
        r = api_client.post(f"{API}/watermark/detect", json={
            "session_id": "non-existent-session-id-xyz"
        }, timeout=30)
        assert r.status_code == 404
