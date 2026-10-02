"""Fresh actual config modules for tests selecting a new independent profile."""

import importlib.util
import sys
from collections.abc import MutableMapping
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any

import pytest

_CONFIG_SOURCE_INSTALLS: list[tuple[ModuleType, ModuleType, set[str]]] = []


def install_config_source(monkeypatch):
    """Import a real selected source, preserving existing participant registries."""
    import tldw_chatbook
    from tldw_chatbook import config

    existing_consumers = {
        name for name in sys.modules if name.startswith("tldw_chatbook.")
    }
    spec = importlib.util.spec_from_file_location(config.__name__, config.__file__)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    monkeypatch.setattr(tldw_chatbook, "config", module)
    _CONFIG_SOURCE_INSTALLS.append((config, module, existing_consumers))
    spec.loader.exec_module(module)
    return module


def select_config_source(
    monkeypatch: pytest.MonkeyPatch,
    path: str | Path | None,
    *namespaces: MutableMapping[str, Any],
) -> ModuleType:
    """Select one independent test source and bind only explicit consumers.

    Call this before patching config behavior or creating its consumers. Later
    environment changes still hit the real source-selection guard. No live
    participant registry or unrelated imported module is reset.

    Args:
        monkeypatch: Owns environment, module and explicit binding restoration.
        path: Config file to select, or None to use the environment's default.
        namespaces: Explicit test or adapter namespaces whose config imports
            should bind to the fresh source.

    Returns:
        The actual config module imported under the selected profile.

    Raises:
        RecoveryRequired: If the selected profile fails config admission.
        PrivatePathError: If the config file or parent fails private-path checks.
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


def restore_config_source_consumers() -> None:
    """Restore exact config module, function and class imports from selections.

    Run after fixture teardown restores monkeypatches, so explicit consumer
    undo cannot reinstall a retired source. Reverse installation order handles
    nested selections; only new application module namespaces are examined.
    Live resources must have completed their existing teardown first.
    """
    installs = _CONFIG_SOURCE_INSTALLS[:]
    _CONFIG_SOURCE_INSTALLS.clear()
    for previous, selected, existing_consumers in reversed(installs):
        replacements = {id(selected): (selected, previous)}
        for name, value in vars(selected).items():
            if (
                isinstance(value, (FunctionType, type))
                and value.__module__ == selected.__name__
                and name in vars(previous)
            ):
                replacements[id(value)] = (value, vars(previous)[name])
        for name, consumer in tuple(sys.modules.items()):
            if (
                not name.startswith("tldw_chatbook.")
                or name in existing_consumers
                or not isinstance(consumer, ModuleType)
                or consumer is selected
            ):
                continue
            for binding, value in tuple(vars(consumer).items()):
                replacement = replacements.get(id(value))
                if replacement is not None and value is replacement[0]:
                    setattr(consumer, binding, replacement[1])
