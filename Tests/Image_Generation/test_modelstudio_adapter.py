"""Test ModelStudio image adapter."""

import base64
import io

import pytest
from PIL import Image

from tldw_chatbook.Image_Generation import config as _c
from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest


def _b64():
    """Generate a test 8x8 PNG in base64."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (120, 0, 160)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_modelstudio_sync_no_reference_image(monkeypatch):
    """Test ModelStudio sync mode with no reference image.

    Verifies:
    - Image content is extracted from sync response
    - reference_image_data_url is never called when reference_image=None
    """
    from tldw_chatbook.Image_Generation.adapters import modelstudio_image_adapter as m

    _c.reset_image_generation_config_cache()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setattr(m.time, "sleep", lambda *_: None)
    # reference_image=None must never call reference_image_data_url
    monkeypatch.setattr(
        m,
        "reference_image_data_url",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("reference_image_data_url must not be called")
        ),
    )
    monkeypatch.setattr(
        m,
        "fetch_json",
        lambda method, url, **kw: {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": [{"image": "data:image/png;base64," + _b64()}]
                        }
                    }
                ]
            }
        },
    )
    req = ImageGenRequest(
        backend="modelstudio",
        prompt="lotus",
        negative_prompt=None,
        width=None,
        height=None,
        steps=None,
        cfg_scale=None,
        seed=None,
        sampler=None,
        model="qwen-image",
        format="png",
        extra_params={"mode": "sync"},
        reference_image=None,
    )
    res = m.ModelStudioImageAdapter().generate(req)
    assert res.bytes_len > 0


def test_modelstudio_generate_trusts_configured_private_base_host_for_returned_image(monkeypatch):
    """The configured base_url may itself be a private/local dashscope-compatible
    proxy (e.g. http://192.168.1.20:8080). An image URL the API returns on that
    SAME host must be trusted through to the byte fetch (task-498: trust
    extends only as far as the configured endpoint, never further)."""
    from tldw_chatbook.Image_Generation.adapters import modelstudio_image_adapter as m

    _c.reset_image_generation_config_cache()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("MODELSTUDIO_IMAGE_BASE_URL", "http://192.168.1.20:8080/api/v1")
    monkeypatch.setattr(
        m,
        "fetch_json",
        lambda method, url, **kw: {
            "output": {
                "choices": [
                    {"message": {"content": [{"image": "http://192.168.1.20:8080/img/output.png"}]}}
                ]
            }
        },
    )
    png_bytes = base64.b64decode(_b64())
    seen = {}

    def fake_fetch_image_bytes(url, **kw):
        seen["url"] = url
        seen["trusted_origins"] = kw.get("trusted_origins")
        return png_bytes, "image/png"

    monkeypatch.setattr(m, "fetch_image_bytes", fake_fetch_image_bytes)
    req = ImageGenRequest(
        backend="modelstudio", prompt="lotus", negative_prompt=None, width=None, height=None,
        steps=None, cfg_scale=None, seed=None, sampler=None, model="qwen-image",
        format="png", extra_params={"mode": "sync"}, reference_image=None,
    )
    res = m.ModelStudioImageAdapter().generate(req)
    assert res.bytes_len > 0
    assert seen["trusted_origins"] == frozenset({"192.168.1.20"})


def test_modelstudio_blocks_returned_image_url_off_the_configured_host(monkeypatch):
    """An image URL returned by the API for a DIFFERENT host than the
    configured base_url (and not the aliyuncs.com allowlist entry) must be
    rejected before any byte fetch is attempted -- the dead local `allowlist`
    from the Phase-1 guard is now enforced via the adopted egress policy."""
    from tldw_chatbook.Image_Generation.adapters import modelstudio_image_adapter as m
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    _c.reset_image_generation_config_cache()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("MODELSTUDIO_IMAGE_BASE_URL", "http://192.168.1.20:8080/api/v1")
    monkeypatch.setattr(
        m,
        "fetch_json",
        lambda method, url, **kw: {
            "output": {"choices": [{"message": {"content": [{"image": "http://192.168.1.99/other.png"}]}}]}
        },
    )

    def _must_not_be_called(url, **kw):
        raise AssertionError("fetch_image_bytes must not run for a blocked host")

    monkeypatch.setattr(m, "fetch_image_bytes", _must_not_be_called)
    req = ImageGenRequest(
        backend="modelstudio", prompt="lotus", negative_prompt=None, width=None, height=None,
        steps=None, cfg_scale=None, seed=None, sampler=None, model="qwen-image",
        format="png", extra_params={"mode": "sync"}, reference_image=None,
    )
    with pytest.raises(ImageGenerationError):
        m.ModelStudioImageAdapter().generate(req)
