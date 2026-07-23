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
        "stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio"
    }
    assert all(v.startswith("tldw_chatbook.Image_Generation.adapters.") for v in DEFAULT_ADAPTERS.values())
