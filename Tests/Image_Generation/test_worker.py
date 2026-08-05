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


def _make_reference_image(**overrides):
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage
    defaults = dict(
        file_id=1,
        filename="ref.png",
        mime_type="image/png",
        width=64,
        height=64,
        bytes_len=4,
        content=b"1234",
        temp_path=None,
    )
    defaults.update(overrides)
    return ResolvedReferenceImage(**defaults)


def test_build_request_reference_image_defaults_none():
    from tldw_chatbook.Image_Generation.worker import build_request
    req = build_request(backend="swarmui", prompt="cat")
    assert req.reference_image is None


def test_build_request_threads_reference_image():
    from tldw_chatbook.Image_Generation.worker import build_request
    ref = _make_reference_image()
    req = build_request(backend="fal", prompt="cat", reference_image=ref)
    assert req.reference_image is ref


def test_run_generation_reference_image_unsupported_backend_raises_before_adapter(monkeypatch):
    # A legacy backend with a reference image attached must be refused at the
    # validation choke point in run_generation() -- the adapter must never be
    # reached (get_adapter() would raise if it were called).
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeReg:
        def resolve_backend(self, name):
            return "swarmui" if name == "swarmui" else None

        def get_adapter(self, name):
            raise AssertionError("adapter must not be reached when validation fails")

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    req = worker.build_request(backend="swarmui", prompt="cat", reference_image=_make_reference_image())
    with pytest.raises(ImageGenerationError, match="does not support reference images"):
        worker.run_generation(req)


def test_run_generation_reference_image_supported_backend_dispatches(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult

    class FakeAdapter:
        name = "fal"
        supported_formats = {"png"}

        def generate(self, req):
            assert req.reference_image is not None
            return ImageGenResult(content=b"x", content_type="image/png", bytes_len=1)

    class FakeReg:
        def resolve_backend(self, name):
            return "fal" if name == "fal" else None

        def get_adapter(self, name):
            return FakeAdapter()

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    req = worker.build_request(backend="fal", prompt="cat", reference_image=_make_reference_image())
    res = worker.run_generation(req)
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
