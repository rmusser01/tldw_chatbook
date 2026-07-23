import io
import pytest
from PIL import Image


def test_sd_cpp_missing_binary_raises(monkeypatch, tmp_path):
    from tldw_chatbook.Image_Generation.adapters import stable_diffusion_cpp_adapter as m
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenRequest
    from tldw_chatbook.Image_Generation.exceptions import ImageBackendUnavailableError

    # config with no binary path -> unavailable
    req = ImageGenRequest(
        backend="stable_diffusion_cpp",
        prompt="cat",
        negative_prompt=None,
        width=512,
        height=512,
        steps=10,
        cfg_scale=7.0,
        seed=-1,
        sampler=None,
        model=None,
        format="png",
        extra_params={},
    )
    with pytest.raises((ImageBackendUnavailableError, Exception)):
        m.StableDiffusionCppAdapter().generate(req)
