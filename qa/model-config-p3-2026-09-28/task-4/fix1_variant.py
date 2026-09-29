"""TASK-33003.4 fix round 1: write one negative-control variant of the head modal.

Usage: fix1_variant.py <scratch tree at c959f8b594> {no-grow-watch|no-scroll-watch|old-rule}
Never point it at a real checkout; restore with `git checkout HEAD -- <modal>`.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "tldw_chatbook/Widgets/Console/console_settings_modal.py"
src = p.read_text()
SCROLL = '        self.watch(body, "scroll_y", self._sync_fold_hint, init=False)\n'
GROW = (
    "        # Content can grow with no caller (a focused picker's results); sync\n"
    "        # after the refresh, once container_size has caught up too.\n"
    "        self.watch(\n"
    "            body,\n"
    '            "virtual_size",\n'
    "            lambda: self.call_after_refresh(self._sync_fold_hint),\n"
    "            init=False,\n"
    "        )\n"
)
RULE = "        hint.display = body.scroll_y < body.max_scroll_y\n"
OLD_RULE = "        hint.display = body.virtual_size.height > body.container_size.height\n"
edits = {
    "no-grow-watch": [(GROW, "")],
    "no-scroll-watch": [(SCROLL, "")],
    "old-rule": [(RULE, OLD_RULE)],
}[sys.argv[2]]
for old, new in edits:
    assert src.count(old) == 1, (sys.argv[2], old)
    src = src.replace(old, new)
p.write_text(src)
print("variant", sys.argv[2], "written")
