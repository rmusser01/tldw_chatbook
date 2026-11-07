"""Original native configuration definitions for deferred first-send persistence."""

import sys
from types import FunctionType

from tldw_chatbook import config


def _native_config_current():
    if sys.modules.get("tldw_chatbook.config") is not config:
        return False
    namespace, rows = config._FIRST_SEND_PERSISTENCE_SOURCE
    if namespace is not vars(config):
        return False
    for owner, name, original, records in rows:
        if owner is not namespace or owner.get(name) is not original or not records:
            return False
        for (
            fn,
            code,
            globals_,
            defaults,
            kwdefaults,
            entries,
            closure,
            cells,
            wrapped,
        ) in records:
            if (
                type(fn) is not FunctionType
                or fn.__code__ is not code
                or fn.__globals__ is not globals_
                or fn.__defaults__ is not defaults
                or fn.__kwdefaults__ is not kwdefaults
                or fn.__closure__ is not closure
                or vars(fn).get("__wrapped__") is not wrapped
                or (
                    kwdefaults is not None
                    and (
                        len(kwdefaults) != len(entries)
                        or any(
                            kwdefaults.get(key) is not value for key, value in entries
                        )
                    )
                )
                or any(cell.cell_contents is not value for cell, value in cells)
            ):
                return False
    return True


from .console_ordinary_tree import _function_record

_NATIVE_CONFIG_CHECK = _function_record(_native_config_current)
