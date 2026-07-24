import pytest


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
