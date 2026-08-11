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
        lambda _backend: ReferenceImageCapability(supported=True, required=False),
    )
    monkeypatch.setattr(worker, "validate_image_generation_request", lambda _request: [])

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
    construction_started = threading.Event()
    second_construction_started = threading.Event()
    release_construction = threading.Event()
    constructed = []

    class BlockingRegistry:
        def __init__(self):
            constructed.append(self)
            if len(constructed) == 1:
                construction_started.set()
                assert release_construction.wait(5)
            else:
                second_construction_started.set()

    monkeypatch.setattr(registry, "ImageAdapterRegistry", BlockingRegistry)
    returned = []
    first = threading.Thread(
        target=lambda: returned.append(registry.get_registry()), daemon=True
    )
    second = threading.Thread(
        target=lambda: returned.append(registry.get_registry()), daemon=True
    )
    first.start()
    assert construction_started.wait(5)
    second.start()
    second_started_before_release = second_construction_started.wait(0.2)
    release_construction.set()
    first.join(5)
    second.join(5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not second_started_before_release
    assert len(constructed) == 1
    assert returned == [constructed[0], constructed[0]]
