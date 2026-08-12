import threading
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image


_FAKE_CONFIG = SimpleNamespace()


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
    assert req.cancel_event is None


def test_build_request_preserves_cancel_event_identity():
    from tldw_chatbook.Image_Generation.worker import build_request

    cancel_event = threading.Event()
    req = build_request(backend="swarmui", prompt="cat", cancel_event=cancel_event)

    assert req.cancel_event is cancel_event


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
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return "swarmui" if name == "swarmui" else None
        def get_adapter(self, name):
            return FakeAdapter()
    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    res = worker.run_generation(worker.build_request(backend="swarmui", prompt="cat"))
    assert res.bytes_len == 1


def _make_reference_image(**overrides):
    from tldw_chatbook.Image_Generation.capabilities import ResolvedReferenceImage
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    content = buffer.getvalue()
    defaults = dict(
        file_id=1,
        filename="ref.png",
        mime_type="image/png",
        width=2,
        height=2,
        bytes_len=len(content),
        content=content,
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
        config = _FAKE_CONFIG

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
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return "fal" if name == "fal" else None

        def get_adapter(self, name):
            return FakeAdapter()

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    req = worker.build_request(backend="fal", prompt="cat", reference_image=_make_reference_image())
    res = worker.run_generation(req)
    assert res.bytes_len == 1


def test_run_generation_comfyui_requires_reference_before_adapter_construction(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeReg:
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return "comfyui" if name == "h3-alias" else None

        def get_adapter(self, name):
            raise AssertionError("adapter must not be constructed before validation")

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())

    with pytest.raises(ImageGenerationError, match="requires a reference image"):
        worker.run_generation(worker.build_request(backend="h3-alias", prompt="edit"))


def test_run_generation_disabled_alias_stays_unavailable(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeReg:
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return None

        def get_adapter(self, name):
            raise AssertionError("disabled backend must not construct an adapter")

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())

    with pytest.raises(ImageGenerationError, match="not enabled/available"):
        worker.run_generation(worker.build_request(backend="h3-alias", prompt="edit"))


def test_run_generation_comfyui_invalid_reference_precedes_adapter_construction(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeReg:
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return "comfyui"

        def get_adapter(self, name):
            raise AssertionError("adapter must not be constructed before validation")

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    reference = _make_reference_image(content=b"not-an-image", bytes_len=12)

    with pytest.raises(ImageGenerationError, match="could not be decoded"):
        worker.run_generation(
            worker.build_request(backend="comfyui", prompt="edit", reference_image=reference)
        )


@pytest.mark.parametrize("content", ["not-bytes", object()])
def test_run_generation_non_bytes_reference_precedes_adapter_construction(monkeypatch, content):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeReg:
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return "comfyui"

        def get_adapter(self, name):
            raise AssertionError("adapter must not be constructed before validation")

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    reference = _make_reference_image(content=content, bytes_len=1)

    with pytest.raises(ImageGenerationError, match="content must be bytes"):
        worker.run_generation(
            worker.build_request(backend="comfyui", prompt="edit", reference_image=reference)
        )


def test_run_generation_optional_reference_backend_still_allows_text_to_image(monkeypatch):
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult

    class FakeAdapter:
        def generate(self, request):
            assert request.reference_image is None
            return ImageGenResult(content=b"x", content_type="image/png", bytes_len=1)

    class FakeReg:
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return "fal"

        def get_adapter(self, name):
            return FakeAdapter()

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())

    assert worker.run_generation(worker.build_request(backend="fal", prompt="cat")).bytes_len == 1


def test_run_generation_adapter_load_failure_raises(monkeypatch):
    # resolve_backend() says the backend is enabled, but get_adapter() failed
    # to construct it (e.g. a bad adapter spec / import error swallowed by the
    # registry) and returns None -- run_generation must not call .generate()
    # on None, it must raise a clear ImageGenerationError instead.
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.exceptions import ImageGenerationError

    class FakeReg:
        config = _FAKE_CONFIG

        def resolve_backend(self, name):
            return "swarmui" if name == "swarmui" else None
        def get_adapter(self, name):
            return None  # adapter failed to load

    monkeypatch.setattr(worker, "get_registry", lambda: FakeReg())
    req = worker.build_request(backend="swarmui", prompt="cat")
    with pytest.raises(ImageGenerationError, match="failed to load"):
        worker.run_generation(req)


def test_run_generation_uses_one_registry_config_snapshot_across_reset(monkeypatch):
    from tldw_chatbook.Image_Generation import adapter_registry as registry
    from tldw_chatbook.Image_Generation import config as image_config
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult

    phase = ["A"]
    capability_entered = threading.Event()
    release_capability = threading.Event()
    capability_configs = []
    validation_configs = []
    adapter_configs = []
    results = []
    errors = []
    real_capability = worker.resolve_backend_reference_image_capability
    real_validate = worker.validate_image_generation_request

    raw_by_phase = {
        "A": {
            "default_backend": "stable_diffusion_cpp",
            "enabled_backends": ["stable_diffusion_cpp"],
            "max_width": 64,
            "stable_diffusion_cpp": {"allowed_extra_params": ["cli_args"]},
        },
        "B": {
            "default_backend": "stable_diffusion_cpp",
            "enabled_backends": ["stable_diffusion_cpp"],
            "max_width": 16,
            "stable_diffusion_cpp": {"allowed_extra_params": []},
        },
    }

    class SnapshotAdapter:
        name = "stable_diffusion_cpp"
        supported_formats = {"png"}

        def __init__(self):
            self.config = image_config.get_image_generation_config()

        def generate(self, _request):
            adapter_configs.append(self.config)
            payload = str(self.config.max_width).encode()
            return ImageGenResult(
                content=payload,
                content_type="image/png",
                bytes_len=len(payload),
            )

    def blocked_capability(backend, *, config=None):
        capability_configs.append(config)
        if len(capability_configs) == 1:
            capability_entered.set()
            assert release_capability.wait(5)
        return real_capability(backend, config=config)

    def captured_validation(structured, *, config=None):
        validation_configs.append(config)
        return real_validate(structured, config=config)

    monkeypatch.setattr(
        image_config,
        "_read_image_generation_toml",
        lambda: raw_by_phase[phase[0]],
    )
    monkeypatch.setitem(
        registry.ImageAdapterRegistry.DEFAULT_ADAPTERS,
        "stable_diffusion_cpp",
        SnapshotAdapter,
    )
    monkeypatch.setattr(
        worker, "resolve_backend_reference_image_capability", blocked_capability
    )
    monkeypatch.setattr(
        worker, "validate_image_generation_request", captured_validation
    )
    image_config.reset_image_generation_runtime()
    registry_a = registry.get_registry()
    config_a = registry_a.config

    request_a = worker.build_request(
        backend="stable_diffusion_cpp",
        prompt="neutral image",
        width=32,
        extra_params={"cli_args": []},
    )

    def run_a():
        try:
            results.append(worker.run_generation(request_a))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    request_thread = threading.Thread(target=run_a, daemon=True)
    request_thread.start()
    assert capability_entered.wait(5)

    phase[0] = "B"
    image_config.reset_image_generation_runtime()
    registry_b = registry.get_registry()
    config_b = registry_b.config
    release_capability.set()
    request_thread.join(5)

    assert not request_thread.is_alive()
    assert errors == []
    assert results[0].content == b"64"
    assert capability_configs[0] is config_a
    assert validation_configs[0] is config_a
    assert adapter_configs[0] is config_a

    result_b = worker.run_generation(
        worker.build_request(
            backend="stable_diffusion_cpp", prompt="next neutral image", width=8
        )
    )
    assert result_b.content == b"16"
    assert capability_configs[1] is config_b
    assert validation_configs[1] is config_b
    assert adapter_configs[1] is config_b
    image_config.reset_image_generation_runtime()
