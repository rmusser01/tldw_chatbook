# tldw_chatbook/Agents/builtin_pack_config.py
"""Which built-in tool packs (and, for the legacy path, which tools) are enabled.

Reads ``[agent_tools] enabled_packs``. Uses ``get_cli_setting``'s 3-arg
form deliberately: the 2-arg form's second positional slot carries the
DEFAULT, not a key, and mis-slotting it there is exactly the defect
TASK-547 records in ``get_tool_executor()``.

Back-compat: TASK-584 shipped per-tool ``[tools] read_file_enabled`` /
``list_directory_enabled`` flags. Those keep working as a deprecated
fallback so an existing user is never silently switched off, but an
explicit ``enabled_packs`` list always wins -- including an explicitly
empty one, which means "no packs" rather than "fall back".

The owner's ruling on the legacy fallback (post task-545 P2, which grew
the ``files`` pack to four tools): the legacy flags name individual
tools, not a whole pack, so they must grant exactly the tools they name
and nothing more. A user who only ever set ``read_file_enabled = true``
must not silently gain ``list_directory``/``glob_files``/``grep_files``
just because task-545 P2 happened to group all four under one pack. The
modern ``[agent_tools] enabled_packs`` path has no such restriction --
it is pack-level config by design and always grants every tool in a
named pack.

That is a tool-level restriction the modern, pack-level config shape has
no room for, so ``resolve_enabled_packs()`` returns it as a separate,
explicit field (``PackResolution.only_tools``) rather than smuggling it
into the pack set somehow. ``None`` means "no restriction, take every
tool in the resolved packs" -- the modern path always returns this.  A
(possibly empty) ``frozenset`` means "restrict to exactly these tool
names" -- only the legacy fallback ever returns this, and callers must
test for ``None`` explicitly (never truthiness) so "no restriction" can
never be confused with "restrict to nothing".

``BuiltinToolProvider`` -- and therefore this module's
``resolve_enabled_packs()`` -- is rebuilt fresh every agent turn (see
``console_agent_bridge``'s per-run builder docstring), so the legacy-flag
deprecation warning uses the same once-per-process log idiom as
``Internal_Prompts.resolver`` (see ``_warn_once`` there): without it, a
user still on the legacy flags would get the warning on every single turn
for the life of a conversation.
"""

from __future__ import annotations

from typing import NamedTuple

from loguru import logger

from tldw_chatbook.config import get_cli_setting

#: `[tools]` flag name -> the single tool name it grants under the legacy
#: fallback. Order here is preserved in the deprecation warning's message.
_LEGACY_FLAG_TOOLS: dict[str, str] = {
    "read_file_enabled": "read_file",
    "list_directory_enabled": "list_directory",
}

#: `[tools]` flags that used to enable individual file tools. Kept as its
#: own tuple (rather than re-deriving it from `_LEGACY_FLAG_TOOLS` at each
#: use) so the warning message's flag list has a stable, documented name.
_LEGACY_FILE_FLAGS = tuple(_LEGACY_FLAG_TOOLS)

_MISSING = object()

#: Dedup keys already warned about (see `_warn_once`). Mirrors
#: `Internal_Prompts.resolver._warned_ids`; tests clear this directly the
#: same way that module's tests clear its set (see `cli_setting` in
#: `Tests/Agents/test_builtin_pack_config.py`), so one test's fallback
#: warning can never suppress another test's assertion that it fired.
_warned_ids: set[str] = set()


def _warn_once(key: str, message: str, **kwargs: object) -> None:
    """Log ``message`` once per ``key`` per process.

    Same idiom as ``Internal_Prompts.resolver._warn_once``: a per-run
    provider that calls the guarded path every turn must not re-log a
    warning that already ran the first time.

    Args:
        key: Dedup key; ``message`` logs at most once per key per process.
        message: The loguru-style message, with ``{name}`` placeholders.
        **kwargs: Values for ``message``'s placeholders.
    """
    if key in _warned_ids:
        return
    _warned_ids.add(key)
    logger.warning(message, **kwargs)


class PackResolution(NamedTuple):
    """Resolved built-in tool-pack configuration for one lookup.

    Attributes:
        packs: Pack names to activate.
        only_tools: ``None`` means no tool-level restriction -- every tool
            contributed by an activated pack is available. A (possibly
            empty) frozenset means "restrict to exactly these tool names,
            across all activated packs". Only the deprecated per-tool
            ``[tools]`` fallback ever produces the restricted form, since
            it names individual tools rather than a whole pack; the
            modern ``[agent_tools] enabled_packs`` path always returns
            ``None`` here. Callers MUST branch on ``is None`` rather than
            truthiness -- an empty frozenset is a real, deliberate
            "grant nothing" instruction, not the same as "unrestricted".
    """

    packs: frozenset[str]
    only_tools: frozenset[str] | None = None


def resolve_enabled_packs() -> PackResolution:
    """Resolve active built-in packs, and any tool-level restriction.

    Returns:
        A `PackResolution`. The modern `[agent_tools] enabled_packs` path
        (including an explicitly empty list, and the ignored-non-list and
        nothing-configured cases) always returns `only_tools=None` --
        pack-level config never restricts within a pack. Only the
        deprecated per-tool `[tools]` fallback sets `only_tools`, to
        exactly the tool name(s) its flag(s) name -- never the whole
        `files` pack.
    """
    configured = get_cli_setting("agent_tools", "enabled_packs", _MISSING)
    if configured is not _MISSING:
        if not isinstance(configured, list):
            logger.warning(
                "[agent_tools] enabled_packs must be a list; ignoring {value!r}",
                value=configured,
            )
            return PackResolution(frozenset(), None)
        return PackResolution(frozenset(str(name) for name in configured), None)

    granted_tools = frozenset(
        tool_name
        for flag, tool_name in _LEGACY_FLAG_TOOLS.items()
        if get_cli_setting("tools", flag, False)
    )
    if granted_tools:
        _warn_once(
            "legacy_tools_flags",
            "[tools] {flags} are deprecated; set [agent_tools] enabled_packs = "
            '["files"] instead. Granting only the tool(s) each flag names.',
            flags=", ".join(_LEGACY_FILE_FLAGS),
        )
        return PackResolution(frozenset({"files"}), granted_tools)

    return PackResolution(frozenset(), None)


def enabled_packs() -> frozenset[str]:
    """Pack names the user has switched on.

    Back-compat wrapper over `resolve_enabled_packs()` for callers that
    only need which packs are active and do not care about any
    tool-level restriction within them (see `PackResolution.only_tools`).

    Returns:
        The configured pack names; an empty set when nothing is enabled.
        Defaults to empty -- built-in file tools ship disabled (TASK-584)
        and this function must not change that posture.
    """
    return resolve_enabled_packs().packs
