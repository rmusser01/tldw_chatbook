# tldw_chatbook/Agents/builtin_packs/__init__.py
"""Registry of built-in tool packs.

A pack groups tools the user enables together. Packs bound the catalog so
it stays near ``DIRECT_DISCLOSE_THRESHOLD``, and they give the permission
gate a coarse consent surface above the per-tool one.

Each pack module exports ``TOOLS`` (tool classes, catalog order) and
``REQUIRES`` (optional-dependency feature names). A pack whose
dependencies are unmet is ABSENT from the catalog rather than present and
failing at invoke -- the model must not spend turns discovering a tool is
broken.
"""

from __future__ import annotations

from types import ModuleType

from loguru import logger

from . import files

#: pack name -> module. Add new packs here.
PACKS: dict[str, ModuleType] = {"files": files}


def pack_available(pack: ModuleType) -> bool:
    """Whether every optional dependency this pack declares is installed.

    Args:
        pack: A pack module exporting ``REQUIRES``.

    Returns:
        True when ``REQUIRES`` is empty or every named feature resolves.
    """
    requires = getattr(pack, "REQUIRES", ())
    if not requires:
        return True
    from tldw_chatbook.Utils.optional_deps import check_dependency

    return all(check_dependency(name) for name in requires)


def pack_tool_classes(enabled: frozenset[str]) -> tuple[type, ...]:
    """Tool classes contributed by the enabled, available packs.

    Unknown names are ignored rather than raising: a config naming a pack
    from a newer release must not break the run.

    Args:
        enabled: Pack names the user has switched on.

    Returns:
        Tool classes in ``PACKS`` iteration order, then pack order.
    """
    classes: list[type] = []
    for name, pack in PACKS.items():
        if name not in enabled:
            continue
        if not pack_available(pack):
            logger.info("Built-in pack {name} hidden: missing dependencies", name=name)
            continue
        classes.extend(pack.TOOLS)
    return tuple(classes)
