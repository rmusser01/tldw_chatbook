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
    assert result.content == b"png"
    assert constructed_with == [image_config.get_image_generation_config()]
    assert constructed_with[0].comfyui_image_base_url == "http://127.0.0.1:8288"
