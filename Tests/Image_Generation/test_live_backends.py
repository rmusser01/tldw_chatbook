"""Opt-in live integration tests for the image-generation backends (spec §8).

Each test drives a real backend end to end through the same public entry point
the app uses (``worker.build_request`` + ``worker.run_generation``): a cloud
API, a locally-running server, or a locally-installed binary. None of that is
available in CI, so every test here skips by default and only runs when a
developer explicitly supplies the relevant credentials/server/binary via
environment variables (mirrors the opt-in convention in
``Tests/Chat/test_live_thinking_provider_apis.py``).

Env vars, one group per backend:
    openrouter:            OPENROUTER_API_KEY
    novita:                NOVITA_API_KEY
    together:               TOGETHER_API_KEY
    modelstudio:            DASHSCOPE_API_KEY
    swarmui:                TLDW_LIVE_SWARMUI_BASE_URL (a reachable SwarmUI server)
    stable_diffusion_cpp:   TLDW_LIVE_SD_CPP_BINARY + TLDW_LIVE_SD_CPP_MODEL_PATH
    fal:                    TLDW_LIVE_FAL_API_KEY (+ optional TLDW_LIVE_FAL_MODEL)
    gemini:                 TLDW_LIVE_GEMINI_API_KEY (+ optional TLDW_LIVE_GEMINI_MODEL)

fal/gemini deliberately use their own ``TLDW_LIVE_*`` opt-in var, distinct
from the production secret env vars their adapters otherwise fall back to
(``FAL_KEY``, ``GEMINI_API_KEY``/``GOOGLE_API_KEY``) -- so a developer who
happens to have one of those exported for unrelated reasons never
accidentally triggers a paid live call just by running the suite. The key is
fed to the backend via a crafted config section (like the sd.cpp/swarmui
groups above), never by relying on the adapter's own env fallback.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.slow]

_PROMPT = "a small red circle on a plain white background"


@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Image_Generation import adapter_registry as r
    from tldw_chatbook.Image_Generation import config as c

    c.reset_image_generation_config_cache()
    r.reset_registry()
    yield
    c.reset_image_generation_config_cache()
    r.reset_registry()


def _required_env(*names: str) -> dict[str, str]:
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        pytest.skip(
            "Set " + ", ".join(missing) + " to run this live image-generation test."
        )
    return values


def _enable_backend(monkeypatch, backend: str, *, toml: dict | None = None) -> None:
    """Point the config loader at a single enabled backend for this test.

    Secrets (API keys) are resolved directly from the real environment by
    ``config._resolve_secret`` -- only non-secret fields (like a base URL or a
    local binary/model path) need to flow through the fake TOML section here.
    """
    from tldw_chatbook.Image_Generation import config as c

    section: dict = {"default_backend": backend, "enabled_backends": [backend]}
    if toml:
        section[backend] = toml
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: section, raising=False)


def _generate(backend: str, *, prompt: str = _PROMPT, **kwargs):
    from tldw_chatbook.Image_Generation.worker import build_request, run_generation

    req = build_request(
        backend=backend, prompt=prompt, seed=1, image_format="png", **kwargs
    )
    return run_generation(req)


def _assert_real_image(res) -> None:
    assert res.bytes_len > 0
    assert res.content_type.startswith("image/")
    assert len(res.content) == res.bytes_len


def _tiny_png_bytes() -> bytes:
    """A real (not merely well-formed-looking) 1x1 PNG, for the gemini
    reference-image edit test -- Gemini actually decodes the inline data, so
    an arbitrary placeholder byte string would get rejected server-side
    (mirrors ``Tests/Image_Generation/test_demo_screen.py``'s own helper)."""
    import io
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (1, 1), color=(255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def test_live_openrouter_generates_image(monkeypatch):
    _required_env("OPENROUTER_API_KEY")
    _enable_backend(monkeypatch, "openrouter")
    _assert_real_image(_generate("openrouter"))


def test_live_novita_generates_image(monkeypatch):
    _required_env("NOVITA_API_KEY")
    _enable_backend(monkeypatch, "novita")
    _assert_real_image(_generate("novita"))


def test_live_together_generates_image(monkeypatch):
    _required_env("TOGETHER_API_KEY")
    _enable_backend(monkeypatch, "together")
    _assert_real_image(_generate("together"))


def test_live_modelstudio_generates_image(monkeypatch):
    _required_env("DASHSCOPE_API_KEY")
    _enable_backend(monkeypatch, "modelstudio")
    _assert_real_image(_generate("modelstudio"))


def test_live_swarmui_generates_image(monkeypatch):
    env = _required_env("TLDW_LIVE_SWARMUI_BASE_URL")
    _enable_backend(
        monkeypatch, "swarmui", toml={"base_url": env["TLDW_LIVE_SWARMUI_BASE_URL"]}
    )
    _assert_real_image(_generate("swarmui"))


def test_live_stable_diffusion_cpp_generates_image(monkeypatch):
    env = _required_env("TLDW_LIVE_SD_CPP_BINARY", "TLDW_LIVE_SD_CPP_MODEL_PATH")
    binary = env["TLDW_LIVE_SD_CPP_BINARY"]
    model_path = env["TLDW_LIVE_SD_CPP_MODEL_PATH"]
    if not (shutil.which(binary) or Path(binary).is_file()):
        pytest.skip(f"TLDW_LIVE_SD_CPP_BINARY {binary!r} is not an executable file")
    if not Path(model_path).is_file():
        pytest.skip(f"TLDW_LIVE_SD_CPP_MODEL_PATH {model_path!r} does not exist")
    _enable_backend(
        monkeypatch,
        "stable_diffusion_cpp",
        toml={"binary_path": binary, "model_path": model_path},
    )
    # Keep the run fast -- this is a smoke test, not a quality check.
    _assert_real_image(_generate("stable_diffusion_cpp", steps=4, width=64, height=64))


def test_live_fal_generates_image(monkeypatch):
    env = _required_env("TLDW_LIVE_FAL_API_KEY")
    model = os.environ.get("TLDW_LIVE_FAL_MODEL", "").strip()
    _enable_backend(
        monkeypatch, "fal", toml={"api_key": env["TLDW_LIVE_FAL_API_KEY"]}
    )
    kwargs = {"model": model} if model else {}
    _assert_real_image(_generate("fal", **kwargs))


def test_live_gemini_generates_image(monkeypatch):
    env = _required_env("TLDW_LIVE_GEMINI_API_KEY")
    model = os.environ.get("TLDW_LIVE_GEMINI_MODEL", "").strip()
    _enable_backend(
        monkeypatch, "gemini", toml={"api_key": env["TLDW_LIVE_GEMINI_API_KEY"]}
    )
    kwargs = {"model": model} if model else {}
    _assert_real_image(_generate("gemini", **kwargs))


def test_live_gemini_reference_image_edit(monkeypatch):
    """gemini additionally supports a reference-image edit request (a tiny
    in-memory PNG + an edit instruction) -- gated on the same
    TLDW_LIVE_GEMINI_API_KEY as the plain generation test above."""
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage

    env = _required_env("TLDW_LIVE_GEMINI_API_KEY")
    model = os.environ.get("TLDW_LIVE_GEMINI_MODEL", "").strip()
    _enable_backend(
        monkeypatch, "gemini", toml={"api_key": env["TLDW_LIVE_GEMINI_API_KEY"]}
    )

    ref_bytes = _tiny_png_bytes()
    reference_image = ResolvedReferenceImage(
        file_id=1,
        filename="ref.png",
        mime_type="image/png",
        width=1,
        height=1,
        bytes_len=len(ref_bytes),
        content=ref_bytes,
        temp_path=None,
    )
    kwargs = {"reference_image": reference_image}
    if model:
        kwargs["model"] = model
    result = _generate(
        "gemini",
        prompt="Recolor this reference image's background to solid blue.",
        **kwargs,
    )
    _assert_real_image(result)
