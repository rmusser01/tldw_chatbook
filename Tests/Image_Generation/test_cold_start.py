import sys
import importlib


def test_importing_package_pulls_no_adapters_or_pillow():
    """Verify that importing Image_Generation package doesn't pull adapters or Pillow."""
    # drop anything already imported so the assertion is meaningful
    for name in list(sys.modules):
        if name.startswith("tldw_chatbook.Image_Generation") or name == "PIL":
            del sys.modules[name]
    importlib.import_module("tldw_chatbook.Image_Generation")
    loaded = set(sys.modules)
    assert not any(
        n.startswith("tldw_chatbook.Image_Generation.adapters.")
        and n.endswith("_adapter")
        for n in loaded
    ), "adapters must be lazy"
    assert "tldw_chatbook.Image_Generation.adapters.image_format_utils" not in loaded
    assert "PIL" not in loaded, "Pillow must not import at package import time"


def test_lazy_accessors_available():
    """Verify that lazy accessors are available after import."""
    import tldw_chatbook.Image_Generation as ig

    assert callable(ig.get_image_generation_config)
    assert callable(ig.get_registry)
