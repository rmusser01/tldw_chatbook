import pytest


def test_build_request_defaults_format_png():
    from tldw_chatbook.Image_Generation.worker import build_request
    req = build_request(backend="swarmui", prompt="cat")
    assert req.format == "png"
    assert req.extra_params == {}          # never None
    assert req.negative_prompt is None


def test_run_generation_unknown_backend_raises(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError
    req = worker.build_request(backend="nope", prompt="cat")
    with pytest.raises(ImageGenerationError):
        worker.run_generation(req)   # registry resolve_backend -> None -> error


def test_run_generation_dispatches_to_adapter(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
    class FakeAdapter:
        name = "swarmui"
        supported_formats = {"png"}
        def generate(self, req):
            return ImageGenResult(content=b"x", content_type="image/png", bytes_len=1)
    class FakeReg:
        def resolve_backend(self, name):
            return "swarmui" if name == "swarmui" else None
        def get_adapter(self, name):
            return FakeAdapter()
    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    res = worker.run_generation(worker.build_request(backend="swarmui", prompt="cat"))
    assert res.bytes_len == 1
