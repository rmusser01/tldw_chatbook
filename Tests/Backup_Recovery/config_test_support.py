"""Fresh actual config modules for tests selecting a new independent profile."""

import importlib.util
import sys


def install_config_source(monkeypatch):
    """Import a real selected source, preserving existing participant registries."""
    import tldw_chatbook
    from tldw_chatbook import config

    spec = importlib.util.spec_from_file_location(config.__name__, config.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(tldw_chatbook, "config", module)
    spec.loader.exec_module(module)
    return module
