import pytest


@pytest.fixture(autouse=True)
def _reset():
    from tldw_chatbook.Video_Generation import adapter_registry as r
    r.reset_registry()
    yield
    r.reset_registry()


def test_resolve_backend_requires_enabled():
    from tldw_chatbook.Video_Generation.adapter_registry import VideoAdapterRegistry
    reg = VideoAdapterRegistry(config_override={"enabled_backends": ["minimax"], "default_backend": "minimax"})
    assert reg.resolve_backend("minimax") == "minimax"
    assert reg.resolve_backend("comfyui") is None      # not enabled
    assert reg.resolve_backend(None) == "minimax"      # default


def test_nothing_enabled_by_default():
    from tldw_chatbook.Video_Generation.adapter_registry import VideoAdapterRegistry
    reg = VideoAdapterRegistry(config_override={"enabled_backends": [], "default_backend": "minimax"})
    assert reg.resolve_backend("minimax") is None


def test_default_adapters_point_at_local_package():
    from tldw_chatbook.Video_Generation.adapter_registry import DEFAULT_ADAPTERS
    assert set(DEFAULT_ADAPTERS) == {"minimax", "comfyui", "stable_diffusion_cpp"}
    assert all(v.startswith("tldw_chatbook.Video_Generation.adapters.") for v in DEFAULT_ADAPTERS.values())


def test_lazy_specs_do_not_import_until_get_adapter():
    """Lazy resolution: resolve_backend never imports (a spec whose module
    does not exist still resolves), and get_adapter fails cleanly with None
    rather than raising when the import fails."""
    from tldw_chatbook.Video_Generation.adapter_registry import VideoAdapterRegistry
    reg = VideoAdapterRegistry(config_override={"enabled_backends": ["ghost"], "default_backend": "ghost"})
    # A deliberately nonexistent module path -- the real backend specs (.3/.6/.7)
    # will become importable as those tasks land, so this test must never rely
    # on a real backend name failing to import.
    reg.register_adapter("ghost", "tldw_chatbook.Video_Generation.adapters.does_not_exist.GhostAdapter")
    assert reg.resolve_backend("ghost") == "ghost"
    assert reg.get_adapter("ghost") is None  # logged, not raised


def test_register_adapter_accepts_class_directly():
    from tldw_chatbook.Video_Generation.adapter_registry import VideoAdapterRegistry

    class FakeAdapter:
        name = "fake"
        supported_formats = {"mp4"}

        def generate(self, request):  # pragma: no cover - not called here
            raise NotImplementedError

    reg = VideoAdapterRegistry(config_override={"enabled_backends": ["fake"], "default_backend": "fake"})
    reg.register_adapter("fake", FakeAdapter)
    assert reg.resolve_backend("fake") == "fake"
    adapter = reg.get_adapter("fake")
    assert isinstance(adapter, FakeAdapter)
    # Cached: second call returns the same instance.
    assert reg.get_adapter("fake") is adapter
