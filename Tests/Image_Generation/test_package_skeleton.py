def test_exceptions_hierarchy():
    from tldw_chatbook.Image_Generation.exceptions import (
        ImageGenerationError, ImageBackendUnavailableError,
    )
    assert issubclass(ImageGenerationError, RuntimeError)
    assert issubclass(ImageBackendUnavailableError, ImageGenerationError)

def test_package_imports_clean():
    import importlib
    mod = importlib.import_module("tldw_chatbook.Image_Generation")
    assert mod is not None


def test_getattr_raises_descriptive_attribute_error():
    import pytest
    import tldw_chatbook.Image_Generation as ig

    with pytest.raises(AttributeError, match=r"Image_Generation.*bogus_name"):
        ig.bogus_name
