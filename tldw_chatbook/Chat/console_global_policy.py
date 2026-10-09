"""Global Console context-policy reads from the installed config (TASK-33620.15.1).

Live on 46c3959526, the Console sync read these nine ``[console]`` keys on
every 0.2 s tick (and again at run start), each time inside a full checked
config operation: storage admission, pause probes and pinned-directory opens
to read an unchanged in-memory config. Main-thread samples put about half of
the transcript sync there. The operation predates warm reads: it was one
handshake for nine reads (TASK-32628, 2026-09-16), and since TASK-32804.1
(2026-09-19) a warm ``get_cli_setting`` read needs none.

A read of the same installed config now reuses its parsed result. Any miss
goes through the checked operation exactly as before: a save or reload
(every publish installs a new dict), an external edit or a changed config
selector (no warm hit), or a local storage pause (so a pause still refuses).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: (installed config dict, reader, parser, parsed overrides) of the last read.
_LAST: tuple[Any, Any, Any, Any] | None = None


def read_global_context_policy_overrides(
    read_setting: Callable[..., Any],
    parse: Callable[[dict[str, Any]], Any],
    keys: tuple[str, ...],
) -> Any:
    """Return the parsed global overrides, reusing them for the same config.

    Args:
        read_setting: ``get_cli_setting``-shaped reader.
        parse: Builds the overrides from the nine raw values.
        keys: The ``[console]`` keys to read, in order.

    Returns:
        The parsed overrides (immutable).

    Raises:
        RecoveryRequired: From the checked operation on a miss (a storage
            pause or a changed selector), as before.
    """
    global _LAST
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Backup_Recovery.config_participants import operation

    last = _LAST
    if (
        last is not None
        and storage_admission._pause is None
        and last[1] is read_setting
        and last[2] is parse
        and config._warm_config_cache_hit() is last[0]
    ):
        return last[3]
    with operation(config):
        values = {key: read_setting("console", key, None) for key in keys}
        installed = config._CONFIG_CACHE
    result = parse(values)
    if installed is not None:
        _LAST = (installed, read_setting, parse, result)
    return result
