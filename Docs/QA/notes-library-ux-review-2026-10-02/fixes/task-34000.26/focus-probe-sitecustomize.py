"""Append one line per focus write onto a pruning/detached widget (TASK-34000.26 AC#5)."""
import os
from textual.screen import Screen

_LOG = os.environ.get("FOCUS_PROBE_LOG")
_real = Screen.set_focus

def _probe(self, widget, *args, **kwargs):
    if _LOG and widget is not None and (getattr(widget, "_pruning", False) or not widget.is_attached):
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{getattr(widget, 'id', None)} pruning={getattr(widget, '_pruning', None)} attached={widget.is_attached}\n")
    return _real(self, widget, *args, **kwargs)

Screen.set_focus = _probe
