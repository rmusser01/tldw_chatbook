import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Image_Generation import adapter_registry as r
    r.reset_registry()
    yield
    r.reset_registry()


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


def test_run_generation_adapter_load_failure_raises(monkeypatch):
    # resolve_backend() says the backend is enabled, but get_adapter() failed
    # to construct it (e.g. a bad adapter spec / import error swallowed by the
    # registry) and returns None -- run_generation must not call .generate()
    # on None, it must raise a clear ImageGenerationError instead.
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeReg:
        def resolve_backend(self, name):
            return "swarmui" if name == "swarmui" else None
        def get_adapter(self, name):
            return None  # adapter failed to load

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    req = worker.build_request(backend="swarmui", prompt="cat")
    with pytest.raises(ImageGenerationError, match="failed to load"):
        worker.run_generation(req)
