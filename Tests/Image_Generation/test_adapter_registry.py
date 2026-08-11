import threading

import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Image_Generation import adapter_registry as r
    r.reset_registry()
    yield
    r.reset_registry()


def test_resolve_backend_requires_enabled():
    from tldw_chatbook.Image_Generation.adapter_registry import ImageAdapterRegistry
    reg = ImageAdapterRegistry(config_override={"enabled_backends": ["swarmui"], "default_backend": "swarmui"})
    assert reg.resolve_backend("swarmui") == "swarmui"
    assert reg.resolve_backend("novita") is None      # not enabled
    assert reg.resolve_backend(None) == "swarmui"     # default


def test_nothing_enabled_by_default():
    from tldw_chatbook.Image_Generation.adapter_registry import ImageAdapterRegistry
    reg = ImageAdapterRegistry(config_override={"enabled_backends": [], "default_backend": "swarmui"})
    assert reg.resolve_backend("swarmui") is None


def test_default_adapters_point_at_local_package():
    from tldw_chatbook.Image_Generation.adapter_registry import DEFAULT_ADAPTERS
    assert set(DEFAULT_ADAPTERS) == {
        "stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio", "gemini", "fal", "comfyui"
    }
    assert all(v.startswith("tldw_chatbook.Image_Generation.adapters.") for v in DEFAULT_ADAPTERS.values())


def test_comfyui_adapter_spec_is_lazy(monkeypatch):
    from tldw_chatbook.Image_Generation import adapter_registry as registry

    imports: list[str] = []

    def forbidden_import(module_name):
        imports.append(module_name)
        raise AssertionError("adapter import must remain lazy")

    monkeypatch.setattr(registry.importlib, "import_module", forbidden_import)
    reg = registry.ImageAdapterRegistry(
        config_override={"enabled_backends": ["comfyui"], "default_backend": None}
    )

    assert reg.resolve_backend("comfyui") == "comfyui"
    assert imports == []
    assert registry.DEFAULT_ADAPTERS["comfyui"].endswith(
        ".comfyui_image_adapter.ComfyUIImageAdapter"
    )


def test_listing_then_worker_share_refreshed_registry_and_config_snapshot(monkeypatch):
    from tldw_chatbook.Image_Generation import adapter_registry as registry
    from tldw_chatbook.Image_Generation import config as image_config
    from tldw_chatbook.Image_Generation import listing, worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
    from tldw_chatbook.Image_Generation.capabilities import ReferenceImageCapability

    constructed_with = []

    class SnapshotAdapter:
        name = "comfyui"
        supported_formats = {"png"}

        def __init__(self):
            self.config = image_config.get_image_generation_config()
            constructed_with.append(self.config)

        def generate(self, request):
            return ImageGenResult(
                content=b"png", content_type="image/png", bytes_len=3
            )

    monkeypatch.setattr(
        image_config,
        "_read_image_generation_toml",
        lambda: {
            "default_backend": "comfyui",
            "enabled_backends": ["comfyui"],
            "comfyui": {"base_url": "http://127.0.0.1:8288"},
        },
    )
    monkeypatch.setitem(
        registry.ImageAdapterRegistry.DEFAULT_ADAPTERS, "comfyui", SnapshotAdapter
    )
    monkeypatch.setattr(
        worker,
        "resolve_backend_reference_image_capability",
        lambda _backend, *, config=None: ReferenceImageCapability(
            supported=True, required=False
        ),
    )
    monkeypatch.setattr(
        worker,
        "validate_image_generation_request",
        lambda _request, *, config=None: [],
    )

    image_config.reset_image_generation_runtime()
    entries = listing.list_image_models_for_catalog()
    listing_registry = registry.get_registry()
    result = worker.run_generation(
        worker.build_request(backend="comfyui", prompt="neutral edit")
    )

    assert [entry["name"] for entry in entries] == ["comfyui"]
    assert registry.get_registry() is listing_registry
    assert listing_registry.config is image_config.get_image_generation_config()
    assert result.content == b"png"
    assert constructed_with == [image_config.get_image_generation_config()]
    assert constructed_with[0].comfyui_image_base_url == "http://127.0.0.1:8288"


def test_inflight_registry_constructs_adapter_with_its_captured_config(monkeypatch):
    from tldw_chatbook.Image_Generation import adapter_registry as registry
    from tldw_chatbook.Image_Generation import config as image_config
    from tldw_chatbook.Image_Generation import worker
    from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
    from tldw_chatbook.Image_Generation.capabilities import ReferenceImageCapability

    origin = ["http://127.0.0.1:8188"]
    validation_entered = threading.Event()
    release_validation = threading.Event()
    validation_calls = 0
    results = []
    errors = []

    class SnapshotAdapter:
        name = "comfyui"
        supported_formats = {"png"}

        def __init__(self):
            self.config = image_config.get_image_generation_config()

        def generate(self, request):
            payload = self.config.comfyui_image_base_url.encode()
            return ImageGenResult(
                content=payload,
                content_type="image/png",
                bytes_len=len(payload),
            )

    def blocked_first_validation(_request, *, config=None):
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            validation_entered.set()
            assert release_validation.wait(5)
        return []

    monkeypatch.setattr(
        image_config,
        "_read_image_generation_toml",
        lambda: {
            "default_backend": "comfyui",
            "enabled_backends": ["comfyui"],
            "comfyui": {"base_url": origin[0]},
        },
    )
    monkeypatch.setitem(
        registry.ImageAdapterRegistry.DEFAULT_ADAPTERS, "comfyui", SnapshotAdapter
    )
    monkeypatch.setattr(
        worker,
        "resolve_backend_reference_image_capability",
        lambda _backend, *, config=None: ReferenceImageCapability(
            supported=True, required=False
        ),
    )
    monkeypatch.setattr(
        worker, "validate_image_generation_request", blocked_first_validation
    )

    image_config.reset_image_generation_runtime()
    old_registry = registry.get_registry()

    def run_old_request():
        try:
            results.append(
                worker.run_generation(
                    worker.build_request(backend="comfyui", prompt="neutral edit")
                )
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    old_request = threading.Thread(target=run_old_request, daemon=True)
    old_request.start()
    assert validation_entered.wait(5)

    origin[0] = "http://127.0.0.1:8288"
    image_config.reset_image_generation_runtime()
    new_registry = registry.get_registry()
    release_validation.set()
    old_request.join(5)

    assert not old_request.is_alive()
    assert errors == []
    old_adapter = old_registry.get_adapter("comfyui")
    assert old_adapter is not None
    assert old_adapter.config is old_registry.config
    assert old_adapter.config.comfyui_image_base_url == "http://127.0.0.1:8188"
    assert results[0].content == b"http://127.0.0.1:8188"

    new_result = worker.run_generation(
        worker.build_request(backend="comfyui", prompt="next neutral edit")
    )
    new_adapter = new_registry.get_adapter("comfyui")
    assert new_adapter is not None
    assert new_adapter.config is new_registry.config
    assert new_adapter.config.comfyui_image_base_url == "http://127.0.0.1:8288"
    assert new_result.content == b"http://127.0.0.1:8288"


def test_runtime_reset_waits_for_inflight_config_load_before_clearing(monkeypatch):
    from tldw_chatbook.Image_Generation import config as image_config

    image_config.reset_image_generation_runtime()
    load_started = threading.Event()
    release_load = threading.Event()
    reset_started = threading.Event()
    reset_done = threading.Event()
    loaded = []
    calls = 0

    def blocking_load():
        nonlocal calls
        calls += 1
        if calls == 1:
            load_started.set()
            assert release_load.wait(5)
            origin = "http://127.0.0.1:8188"
        else:
            origin = "http://127.0.0.1:8288"
        return {"comfyui_image_base_url": origin}, {}

    monkeypatch.setattr(image_config, "_load_image_generation_section", blocking_load)

    load_thread = threading.Thread(
        target=lambda: loaded.append(
            image_config.get_image_generation_config(reload=True)
        ),
        daemon=True,
    )

    def reset_runtime():
        reset_started.set()
        image_config.reset_image_generation_runtime()
        reset_done.set()

    reset_thread = threading.Thread(target=reset_runtime, daemon=True)
    load_thread.start()
    assert load_started.wait(5)
    reset_thread.start()
    assert reset_started.wait(5)
    reset_returned_before_release = reset_done.wait(0.2)
    release_load.set()
    load_thread.join(5)
    reset_thread.join(5)

    assert not load_thread.is_alive()
    assert not reset_thread.is_alive()
    assert not reset_returned_before_release
    assert loaded[0].comfyui_image_base_url == "http://127.0.0.1:8188"
    fresh = image_config.get_image_generation_config()
    assert fresh.comfyui_image_base_url == "http://127.0.0.1:8288"
    assert fresh is not loaded[0]


def test_runtime_reset_waits_for_inflight_registry_construction_before_clearing(
    monkeypatch,
):
    from tldw_chatbook.Image_Generation import adapter_registry as registry
    from tldw_chatbook.Image_Generation import config as image_config

    image_config.reset_image_generation_runtime()
    construction_started = threading.Event()
    release_construction = threading.Event()
    reset_started = threading.Event()
    reset_done = threading.Event()
    constructed = []

    class BlockingRegistry:
        def __init__(self):
            constructed.append(self)
            if len(constructed) == 1:
                construction_started.set()
                assert release_construction.wait(5)

    monkeypatch.setattr(registry, "ImageAdapterRegistry", BlockingRegistry)
    returned = []
    registry_thread = threading.Thread(
        target=lambda: returned.append(registry.get_registry()), daemon=True
    )

    def reset_runtime():
        reset_started.set()
        image_config.reset_image_generation_runtime()
        reset_done.set()

    reset_thread = threading.Thread(target=reset_runtime, daemon=True)
    registry_thread.start()
    assert construction_started.wait(5)
    reset_thread.start()
    assert reset_started.wait(5)
    reset_returned_before_release = reset_done.wait(0.2)
    release_construction.set()
    registry_thread.join(5)
    reset_thread.join(5)

    assert not registry_thread.is_alive()
    assert not reset_thread.is_alive()
    assert not reset_returned_before_release
    fresh = registry.get_registry()
    assert fresh is not returned[0]
    assert constructed == [returned[0], fresh]


def test_concurrent_first_registry_callers_share_one_instance(monkeypatch):
    from tldw_chatbook.Image_Generation import adapter_registry as registry

    registry.reset_registry()
    first_lock_entered = threading.Event()
    second_lock_entered = threading.Event()
    construction_started = threading.Event()
    release_construction = threading.Event()
    constructed = []

    class InstrumentedLock:
        def __init__(self):
            self._lock = threading.Lock()
            self._entries = 0

        def __enter__(self):
            self._entries += 1
            if self._entries == 1:
                first_lock_entered.set()
            elif self._entries == 2:
                second_lock_entered.set()
            self._lock.acquire()
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            self._lock.release()

    class BlockingRegistry:
        def __init__(self):
            constructed.append(self)
            construction_started.set()
            assert release_construction.wait(5)

    monkeypatch.setattr(
        registry, "_IMAGE_GENERATION_RUNTIME_LOCK", InstrumentedLock()
    )
    monkeypatch.setattr(registry, "ImageAdapterRegistry", BlockingRegistry)
    returned = []
    first = threading.Thread(
        target=lambda: returned.append(registry.get_registry()), daemon=True
    )
    second = threading.Thread(
        target=lambda: returned.append(registry.get_registry()), daemon=True
    )
    first.start()
    if not first_lock_entered.wait(5):
        release_construction.set()
        first.join(5)
        pytest.fail("first caller never entered registry serialization")
    if not construction_started.wait(5):
        release_construction.set()
        first.join(5)
        pytest.fail("first caller never began registry construction")
    second.start()
    if not second_lock_entered.wait(5):
        release_construction.set()
        first.join(5)
        second.join(5)
        pytest.fail("second caller never reached registry serialization")
    release_construction.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(constructed) == 1
    assert returned == [constructed[0], constructed[0]]
