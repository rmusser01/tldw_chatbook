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


def select_config_source(monkeypatch, path, *namespaces):
    """Select one independent test source and bind only explicit consumers.

    Call this before patching config behavior or creating its consumers. Later
    environment changes still hit the real source-selection guard. No live
    participant registry or unrelated imported module is reset.
    """
    if path is None:
        monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    else:
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    source = install_config_source(monkeypatch)
    for namespace in namespaces:
        for name, value in tuple(namespace.items()):
            if getattr(value, "__name__", None) == source.__name__:
                monkeypatch.setitem(namespace, name, source)
            elif (
                getattr(value, "__module__", None) == source.__name__
                and getattr(value, "__name__", None) is not None
            ):
                monkeypatch.setitem(namespace, name, getattr(source, value.__name__))
    return source
