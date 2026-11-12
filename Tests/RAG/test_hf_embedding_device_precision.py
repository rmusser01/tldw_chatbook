"""The model's load precision follows its selected execution device."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.Embeddings import Embeddings_Lib as lib


@pytest.mark.parametrize(
    ("requested", "cuda_available", "selected", "dtype"),
    [
        ("cpu", True, "cpu", "float32"),
        ("cpu", False, "cpu", "float32"),
        ("cuda", True, "cuda", "float16"),
        ("auto", True, "cuda", "float16"),
        ("auto", False, "cpu", "float32"),
    ],
)
def test_hf_load_precision_follows_selected_device(
    monkeypatch, requested, cuda_available, selected, dtype
):
    class Device:
        def __init__(self, name):
            self.type = name.split(":", 1)[0]

        def __str__(self):
            return self.type

    class Model:
        device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            pass

    loaded = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        device=Device,
        float16="float16",
        float32="float32",
    )
    monkeypatch.setattr(lib, "_ensure_torch", lambda: fake_torch)
    monkeypatch.setattr(lib, "_ensure_transformers", lambda: object())
    monkeypatch.setattr(lib, "torch", fake_torch)
    monkeypatch.setattr(lib, "transformers", object())
    monkeypatch.setattr(
        lib,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(
        lib,
        "AutoModel",
        SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: loaded.append(kwargs) or Model()
        ),
    )

    embedder = object.__new__(lib._HuggingFaceEmbedder)
    lib._HuggingFaceEmbedder.__init__.__wrapped__(
        embedder, lib.HFModelCfg(model_name_or_path="synthetic/model", device=requested)
    )

    assert embedder._device.type == selected
    assert embedder._dtype == dtype
    assert loaded[0]["torch_dtype"] == dtype
