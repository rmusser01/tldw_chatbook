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
