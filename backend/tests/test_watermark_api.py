"""Backend API tests for LLM Watermark Studio."""
import os
import pytest
import requests

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://mark-detect-2.preview.emergentagent.com").rstrip("/")
API = f"{BASE_URL}/api"

# Shared state for session reuse across detect tests
_shared = {"session_id": None, "target_grid": None, "rows": None, "cols": None, "pattern_length": None}


@pytest.fixture(scope="session")
def api_client():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


# ---------------- /api/watermark/info ---------------- #
class TestInfo:
    def test_info_returns_models_and_patterns(self, api_client):
        r = api_client.get(f"{API}/watermark/info", timeout=30)
        assert r.status_code == 200
        data = r.json()
        assert "models" in data and "patterns" in data
        assert "gpt2" in data["models"]
        assert "gpt2-medium" in data["models"]
        assert "facebook/opt-125m" in data["models"]
        for p in ["square", "checkerboard", "circle", "diamond", "cross"]:
            assert p in data["patterns"]


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


# ---------------- /api/watermark/generate happy path (slow) ---------------- #
class TestGenerate:
    def test_generate_gpt2_checkerboard(self, api_client):
        # Timeout large to allow first-time gpt2 weight download (~500MB)
        r = api_client.post(f"{API}/watermark/generate", json={
            "prompt": "The future of AI is",
            "model_name": "gpt2",
            "max_new_tokens": 40,
            "pattern": "checkerboard",
            "secret_key": "llmwatermark",
            "gamma": 0.5,
            "delta": 0.3,
            "temperature": 0.8,
            "top_p": 0.9,
            "tau": 0.75,
        }, timeout=600)
        assert r.status_code == 200, f"Body: {r.text[:500]}"
        data = r.json()

        # Required keys
        for k in ["session_id", "generated_text", "token_count", "rows", "cols",
                  "pattern_length", "model_name", "delta", "gamma", "tau",
                  "target_grid", "elapsed_s"]:
            assert k in data, f"missing key: {k}"

        # Value assertions
        assert isinstance(data["session_id"], str) and len(data["session_id"]) > 0
        assert isinstance(data["generated_text"], str) and len(data["generated_text"]) > 0
        assert data["token_count"] > 0
        assert data["rows"] > 0 and data["cols"] > 0
        assert data["model_name"] == "gpt2"
        assert data["tau"] == 0.75
        # target_grid is a 2D list of '0'/'1' strings
        assert isinstance(data["target_grid"], list)
        assert isinstance(data["target_grid"][0], list)
        sample = data["target_grid"][0][0]
        assert sample in ("0", "1"), f"cell value unexpected: {sample!r}"

        _shared["session_id"] = data["session_id"]
        _shared["target_grid"] = data["target_grid"]
        _shared["rows"] = data["rows"]
        _shared["cols"] = data["cols"]
        _shared["pattern_length"] = data["pattern_length"]


# ---------------- /api/watermark/detect ---------------- #
class TestDetect:
    def test_detect_with_valid_session(self, api_client):
        sid = _shared["session_id"]
        if not sid:
            pytest.skip("No session_id from generate step")
        r = api_client.post(f"{API}/watermark/detect", json={"session_id": sid}, timeout=300)
        assert r.status_code == 200, f"Body: {r.text[:500]}"
        data = r.json()
        for k in ["is_watermarked", "lcs_ratio", "z_score", "n_tokens",
                  "rows", "cols", "target_grid", "recovered_grid",
                  "bit_matches", "pattern_length", "recovered_length", "tau"]:
            assert k in data, f"missing key: {k}"
        assert isinstance(data["is_watermarked"], bool)
        assert isinstance(data["lcs_ratio"], float)
        assert isinstance(data["z_score"], float)
        assert data["rows"] == _shared["rows"]
        assert data["cols"] == _shared["cols"]
        assert data["pattern_length"] == _shared["pattern_length"]
        # Grids are 2D list of '0'/'1' strings
        assert data["target_grid"][0][0] in ("0", "1")
        # recovered_grid may contain '-' for unrecovered cells
        assert data["recovered_grid"][0][0] in ("0", "1", "-")

    def test_detect_with_invalid_session_returns_404(self, api_client):
        r = api_client.post(f"{API}/watermark/detect", json={
            "session_id": "non-existent-session-id-xyz"
        }, timeout=30)
        assert r.status_code == 404
