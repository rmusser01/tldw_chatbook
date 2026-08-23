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

Every site the census currently sees, in line order. The ones your change
added are in this list -- diff it against the same listing on your merge
base to find them:

{inventory}

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


def find_library_whole_screen_recompose_statements(
    source: str,
) -> list[tuple[int, str, str]]:
    """Locate statement-level whole-screen recompose calls in ``source``.

    Counts, via AST (robust to formatting and line-wrapping, and blind to
    comments and docstrings -- a grep is not):

    - ``self.refresh(... recompose=True ...)`` /
      ``screen.refresh(... recompose=True ...)`` (the module-level helper
      fallbacks take the screen as an explicit ``screen`` argument), plus
      the ``self.screen.refresh(...)`` spelling;
    - ``self.recompose()`` / ``screen.recompose()`` / ``self.screen.
      recompose()`` direct calls.

    KNOWN BLIND SPOTS (deliberate -- each would need a type inference this
    test has no business doing). None is currently used in
    ``library_screen.py``; they are listed so a future author does not
    mistake the pin for a proof of absence:

    - an aliased receiver (``s = self; s.refresh(recompose=True)``);
    - a non-literal flag (``recompose=SOME_CONST`` / ``recompose=flag``);
    - a splatted call (``self.refresh(**kwargs)``);
    - an indirect dispatch (``getattr(self, "refresh")(...)``,
      ``partial(self.refresh, recompose=True)``, a stored bound method);
    - a recompose reached through another object that happens to BE this
      screen (``self.app.screen.refresh(recompose=True)``).

    Args:
        source: The module source text to scan.

    Returns:
        ``(lineno, receiver, spelling)`` for each match, in line order.
    """

    def _receiver(node: ast.expr) -> str | None:
        """Return the receiver name when it denotes this screen."""
        if isinstance(node, ast.Name) and node.id in ("self", "screen"):
            return node.id
        # ``self.screen`` / ``screen.screen`` -- still this screen.
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "screen"
            and isinstance(node.value, ast.Name)
            and node.value.id in ("self", "screen")
        ):
            return f"{node.value.id}.screen"
        return None

    tree = ast.parse(source)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        receiver = _receiver(func.value)
        if receiver is None:
            continue
        if func.attr == "refresh":
            if any(
                keyword.arg == "recompose"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            ):
                found.append(
                    (node.lineno, receiver, f"{receiver}.refresh(recompose=True)")
                )
        elif func.attr == "recompose" and not node.args and not node.keywords:
            found.append((node.lineno, receiver, f"{receiver}.recompose()"))
    return sorted(found)


def count_library_whole_screen_recompose_statements(source: str) -> int:
    """Return how many whole-screen recompose statements ``source`` holds."""
    return len(find_library_whole_screen_recompose_statements(source))


def _enclosing_function(source: str, lineno: int) -> str:
    """Return the innermost function enclosing ``lineno``, or "<module>"."""
    tree = ast.parse(source)
    best_name = "<module>"
    best_start = -1
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        # Innermost == the containing definition that starts latest.
        if node.lineno <= lineno <= end and node.lineno > best_start:
            best_name = node.name
            best_start = node.lineno
    return best_name


def test_library_screen_whole_screen_recompose_count_is_ratcheted() -> None:
    """The whole-screen recompose statement count must not regrow."""
    source = _LIBRARY_SCREEN_PATH.read_text(encoding="utf-8")
    sites = find_library_whole_screen_recompose_statements(source)
    count = len(sites)
    assert count > 0, (
        "The census found zero whole-screen recompose statements -- the "
        "counter is no longer measuring its subject (renamed receiver or "
        "moved module?). Fix the counter before trusting the ratchet."
    )
    if count > LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX:
        # Name the sites. A bare count tells an author that they broke the
        # ratchet but not where, on a 34k-line file (review round, m5).
        inventory = "\n".join(
            f"  library_screen.py:{lineno}  {function}()  ->  {spelling}"
            for lineno, _receiver, spelling in sites
            for function in (_enclosing_function(source, lineno),)
        )
        raise AssertionError(
            _RATCHET_GUIDANCE.format(
                count=count,
                maximum=LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX,
                inventory=inventory,
            )
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

class T:
    def d(self):
        self.screen.refresh(recompose=True)
    async def e(self):
        await self.screen.recompose()
"""
    assert count_library_whole_screen_recompose_statements(counted) == 7

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


def test_ratchet_counter_blind_spots_are_the_documented_ones() -> None:
    """Pin the counter's KNOWN misses so they stay known, not discovered.

    None of these spellings is used in ``library_screen.py`` today. This
    test exists so the docstring's blind-spot list cannot quietly drift
    from the counter's real behaviour: if someone widens the matcher, this
    test fails and the list gets updated with it (review round, m5).
    """
    blind = """
class S:
    def alias(self):
        s = self
        s.refresh(recompose=True)
    def non_literal(self, flag):
        self.refresh(recompose=flag)
    def splat(self, kwargs):
        self.refresh(**kwargs)
    def indirect(self):
        getattr(self, "refresh")(recompose=True)
    def through_app(self):
        self.app.screen.refresh(recompose=True)
"""
    assert count_library_whole_screen_recompose_statements(blind) == 0


def test_ratchet_failure_message_names_the_offending_sites() -> None:
    """A broken ratchet reports file:line and function, not just a count."""
    source = _LIBRARY_SCREEN_PATH.read_text(encoding="utf-8")
    sites = find_library_whole_screen_recompose_statements(source)
    assert sites, "no sites found -- the counter stopped measuring"

    lineno, _receiver, spelling = sites[0]
    function = _enclosing_function(source, lineno)
    assert function != "<module>" or "screen." in spelling
    rendered = _RATCHET_GUIDANCE.format(
        count=len(sites),
        maximum=0,
        inventory=f"  library_screen.py:{lineno}  {function}()  ->  {spelling}",
    )
    assert f"library_screen.py:{lineno}" in rendered
    assert function in rendered
    # The guidance must still name the sanctioned alternatives.
    assert "_sync_library_canvas" in rendered
    assert "_apply_library_open_item_surface" in rendered
