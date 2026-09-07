from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (10, 10, 200)).save(buffer, "PNG")
    return buffer.getvalue()


def test_sd_cpp_generate_reports_resolved_model_filename(monkeypatch, tmp_path):
    """task-558: sd.cpp always knows the exact model file it resolved
    (request override, else configured diffusion/model path) before it ever
    invokes the binary -- capture it on the result so the Console card can
    show a model instead of always blank."""
    from tldw_chatbook.Image_Generation import config as c
    from tldw_chatbook.Image_Generation.adapters import stable_diffusion_cpp_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest

    binary_path = tmp_path / "sd"
    binary_path.write_text("#!/bin/sh\n")
    model_path = tmp_path / "sdxl_base_1.0.gguf"
    model_path.write_bytes(b"fake-model-bytes")

    monkeypatch.setattr(
        c,
        "_read_image_generation_toml",
        lambda: {
            "stable_diffusion_cpp": {
                "binary_path": str(binary_path),
                "model_path": str(model_path),
            }
        },
        raising=False,
    )
    monkeypatch.setattr(c, "_keyring_get", lambda backend: None, raising=False)
    c.reset_image_generation_config_cache()

    def fake_run(cmd, cwd, capture_output, text, timeout):
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_bytes(_png_bytes())
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(m.subprocess, "run", fake_run)

    req = ImageGenRequest(
        backend="stable_diffusion_cpp", prompt="cat", negative_prompt=None,
        width=64, height=64, steps=5, cfg_scale=7.0, seed=-1,
        sampler=None, model=None, format="png", extra_params={},
    )
    try:
        res = m.StableDiffusionCppAdapter().generate(req)
        assert res.resolved_model == "sdxl_base_1.0.gguf"
        # sd.cpp's stdout carries no parseable resolved-seed signal in this
        # adapter's current (unread) stdout capture -- must not fabricate one.
        assert res.resolved_seed is None
    finally:
        c.reset_image_generation_config_cache()


def test_sd_cpp_missing_binary_raises(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    from tldw_chatbook.Image_Generation.adapters import stable_diffusion_cpp_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    # Deterministic: no backend config at all -> no sd binary path -> must raise.
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: None, raising=False)
    c.reset_image_generation_config_cache()

    req = ImageGenRequest(
        backend="stable_diffusion_cpp", prompt="cat", negative_prompt=None,
        width=512, height=512, steps=10, cfg_scale=7.0, seed=-1,
        sampler=None, model=None, format="png", extra_params={},
    )
    with pytest.raises(ImageBackendUnavailableError):
        m.StableDiffusionCppAdapter().generate(req)

    c.reset_image_generation_config_cache()
