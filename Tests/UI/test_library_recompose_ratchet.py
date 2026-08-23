"""TASK-21116: ratchet on whole-screen recompose statements in library_screen.py.

The Library screen is the largest screen in the app (34k+ lines). A
statement-level ``self.refresh(recompose=True)`` there tears down and
remounts the nav bar, footer, rail, and canvas -- measured repeatedly as
the screen's dominant per-click cost (Docs/Design/2026-08-11-input-latency-
audit.md; Docs/Design/2026-08-22-holistic-perf-review.md finding 21116).
Task-281, task-252, task-15457, and task-21116 each converted per-click
sites to targeted seams, and each time the count regrew between fixes.

This test pins the count so regrowth fails CI instead of accruing silently.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Maximum allowed statement-level whole-screen recompose sites in
#: library_screen.py. Recorded at the TASK-21116 conversion (107 before,
#: 97 after -- 15 per-click statements removed, 5 sanctioned fallback /
#: structural-boundary arms added inside the new targeted seams). LOWER
#: this pin when you remove sites; never raise it for a per-click path --
#: see the failure message for the sanctioned seams.
LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX = 97

_LIBRARY_SCREEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "UI"
    / "Screens"
    / "library_screen.py"
)

_RATCHET_GUIDANCE = """
library_screen.py gained statement-level whole-screen recompose sites:
{count} found, {maximum} allowed.

A `self.refresh(recompose=True)` / `await self.recompose()` on the Library
screen remove/remounts the nav bar, footer, rail, and the entire canvas --
the app's most expensive screen rebuild. Before adding one, use the
sanctioned targeted seams instead:

- `_sync_library_canvas(self, kind, then=...)` -- the mounted canvas
  rebuilds only its own children (conversations/media/media-trash/notes/
  prompts/skills/ingest/search/export/landing/handoff).
- `_apply_library_row_toggle(...)` -- in-place row checkbox/count patches.
- `self._apply_library_open_item_surface(build)` -- rail/header sync + a
  canvas-child swap for open-item / open-canvas transitions (media opens,
  Search/RAG result opens, Export entry, viewer exits).
- `self._sync_library_media_viewer_or_recompose()` -- viewer-scoped rebuild
  for in-viewer sub-state flips.
- Patch the mounted widget directly (`Static.update`, `Button.label`,
  `disabled`) for single-control changes.

A whole-screen recompose is only sanctioned as the FALLBACK arm of a
targeted seam, or for true structural route changes the canvas host cannot
express. If your site genuinely needs one, document why at the call site
and lower/raise the pin in the same reviewed change -- never silently.
""".strip()


def count_library_whole_screen_recompose_statements(source: str) -> int:
    """Count statement-level whole-screen recompose calls in ``source``.

    Counts, via AST (robust to formatting/line-wrapping, blind to comments
    and docstrings):

    - ``self.refresh(... recompose=True ...)`` and
      ``screen.refresh(... recompose=True ...)`` (the module-level helper
      fallbacks target the screen instance);
    - ``self.recompose()`` / ``screen.recompose()`` awaited direct calls.

    Args:
        source: The module source text to scan.

    Returns:
        The number of matching call statements.
    """
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        target = func.value
        if not (isinstance(target, ast.Name) and target.id in ("self", "screen")):
            continue
        if func.attr == "refresh":
            if any(
                keyword.arg == "recompose"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                count += 1
        elif func.attr == "recompose" and not node.args and not node.keywords:
            count += 1
    return count


def test_library_screen_whole_screen_recompose_count_is_ratcheted() -> None:
    """The whole-screen recompose statement count must not regrow."""
    source = _LIBRARY_SCREEN_PATH.read_text(encoding="utf-8")
    count = count_library_whole_screen_recompose_statements(source)
    assert count > 0, (
        "The census found zero whole-screen recompose statements -- the "
        "counter is no longer measuring its subject (renamed receiver or "
        "moved module?). Fix the counter before trusting the ratchet."
    )
    assert count <= LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX, _RATCHET_GUIDANCE.format(
        count=count, maximum=LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX
    )


def test_ratchet_counter_measures_its_subject() -> None:
    """The AST counter recognizes every counted spelling (and only those)."""
    counted = """
class S:
    def a(self):
        self.refresh(recompose=True)
    def b(self):
        self.refresh(repaint=True, recompose=True)
    async def c(self):
        await self.recompose()

def helper(screen):
    screen.refresh(recompose=True)

async def helper2(screen):
    await screen.recompose()
"""
    assert count_library_whole_screen_recompose_statements(counted) == 5

    not_counted = """
class S:
    def a(self):
        # self.refresh(recompose=True) in a comment
        self.refresh()
    def b(self):
        '''self.refresh(recompose=True) in a docstring'''
        self.refresh(recompose=False)
    def c(self, canvas, viewer):
        canvas.refresh(recompose=True)
        viewer.refresh(recompose=True)
"""
    assert count_library_whole_screen_recompose_statements(not_counted) == 0
