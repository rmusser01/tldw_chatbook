"""TASK-21116: ratchet on whole-screen recompose statements across the Library surface.

The Library screen is the largest screen in the app (34k+ lines). A
statement-level ``self.refresh(recompose=True)`` there tears down and
remounts the nav bar, footer, rail, and canvas -- measured repeatedly as
the screen's dominant per-click cost (Docs/Design/2026-08-11-input-latency-
audit.md; Docs/Design/2026-08-22-holistic-perf-review.md finding 21116).
Task-281, task-252, task-15457, and task-21116 each converted per-click
sites to targeted seams, and each time the count regrew between fixes.

This test pins the count so regrowth fails CI instead of accruing silently.

TASK-3 (2026-09, ``Docs/superpowers/specs/2026-09-01-library-screen-
decomposition-design.md``, "PR 0b") widened the scanned surface from
``library_screen.py`` alone to that file PLUS every module under
``tldw_chatbook/UI/Library_Modules/``, and counts them as ONE census. PR 0a
moved ``_sync_library_canvas`` and its sanctioned whole-screen-recompose
fallback arms out of ``library_screen.py`` into
``Library_Modules/canvas_sync.py`` -- a single-file census would have
silently drained by exactly the sites that moved, so the ratchet could
regrow undetected on the next decomposition step without this widening. A
class-scoped exemption list (``_CENSUS_EXEMPT_CLASS_SITES``) keeps the
widened glob from also picking up recompose calls that belong to an
unrelated class (e.g. a standalone modal) living in the same directory --
see that constant's docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

#: Maximum allowed statement-level whole-screen recompose sites across the
#: Library surface (library_screen.py + every UI/Library_Modules/*.py file,
#: exemption-filtered -- see the module docstring and
#: ``_CENSUS_EXEMPT_CLASS_SITES``). Recorded at the TASK-21116 conversion
#: (107 before, 97 after -- 15 per-click statements removed, 5 sanctioned
#: fallback / structural-boundary arms added inside the new targeted
#: seams). LOWER this pin when you remove sites; never raise it for a
#: per-click path -- see the failure message for the sanctioned seams.
#:
#: TASK-22228 (item 6) re-based it: the 2026-08-24 reader burn-down had
#: already taken the census to 80 without lowering the pin (23 sites of
#: silent headroom -- a ratchet that cannot bite), and routing the six
#: Reader sub-state presses through
#: ``_sync_library_media_viewer_or_recompose`` took it to 74.
#:
#: TASK-3 (2026-09, surface widening) re-based it again, for the same
#: reason TASK-22228 named: the single-file pin of 74 had already drifted
#: 11 sites above the true single-file count of 63 (measured against the
#: PR-0a merge base) -- more silent headroom. The widened,
#: exemption-filtered census (screen 59 + canvas_sync.py 4 + 0 from the one
#: exempted modal class = 63) confirms PR 0a moved sites without gaining or
#: losing any, so the pin is re-based down to that measured 63 -- see
#: ``Docs/superpowers/specs/2026-09-01-library-screen-decomposition-
#: design.md`` ("PR 0b").
LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX = 63

#: TASK-27019: how far the pin may sit above the measured count before the
#: anti-slack guard (``test_census_pin_is_not_left_slack`` below) fires.
#: Mirrors ``Tests/Architecture/test_screen_size_ratchet.py``'s
#: ``test_budget_is_not_left_slack_after_a_wave`` -- a ceiling-only ratchet
#: lets a wave's gain silently buy the next feature headroom the wave never
#: earned it, which is exactly what happened to THIS pin twice: 107 -> 80
#: (drifted 23 sites before TASK-22228 caught it) and 74 -> 63 (drifted 11
#: sites before TASK-3's surface-widening audit caught it).
#:
#: The size ratchet's own tolerance (200 lines / 10 methods on a ~44,000
#: line / ~1,300 method budget) does NOT transfer here by scaling -- applied
#: proportionally to a census of 63 it rounds to noise (<1). That ratio was
#: never the point: the size ratchet's 200/10 is sized to absorb *ordinary,
#: unrelated in-file edits* that grow or shrink a screen's line/method count
#: for reasons that have nothing to do with a decomposition wave. This
#: census has no equivalent noise floor -- a `self.refresh(recompose=True)`
#: / `self.recompose()` statement is never an incidental byproduct of an
#: unrelated edit, so the count essentially only moves when someone
#: deliberately adds, removes, or relocates a whole-screen recompose site
#: (via a targeted-seam conversion, exactly the change this ratchet exists
#: to police).
#:
#: The number is instead sized against this census's OWN documented drift
#: history: the smallest single silent step on record is the 6-site drop
#: from "routing the six Reader sub-state presses through
#: `_sync_library_media_viewer_or_recompose`" (80 -> 74, comment above).
#: A tolerance of 5 sits strictly below that smallest observed step, so the
#: guard would have fired on every drift increment ever recorded against
#: this pin, including its smallest one, while still forgiving a couple of
#: sites of legitimate same-PR churn (e.g. sites shuffling across files
#: during the widened multi-file scan without a net change) that isn't
#: itself an unlowered wave.
_CENSUS_SLACK_TOLERANCE = 5

_LIBRARY_SCREEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "UI"
    / "Screens"
    / "library_screen.py"
)

_LIBRARY_MODULES_DIR = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "UI"
    / "Library_Modules"
)

#: The whole Library surface the widened census scans: the screen itself
#: plus every module PR 0a's decomposition can move code into or out of. A
#: file added under Library_Modules/ later is picked up automatically by
#: the glob -- no test edit required to stay covered.
_LIBRARY_SURFACE_PATHS = sorted(
    [_LIBRARY_SCREEN_PATH] + list(_LIBRARY_MODULES_DIR.glob("*.py"))
)

#: (filename, class name) pairs whose whole-screen-recompose-shaped sites
#: are EXCLUDED from the census because the receiver does not denote
#: LibraryScreen. Sites in unlisted classes, in module-level functions, and
#: in LibraryScreen itself all still count -- a new file or class dropped
#: into Library_Modules/ bites by default, and adding an exemption here is
#: a deliberate, reviewable act, never a silent one.
_CENSUS_EXEMPT_CLASS_SITES: frozenset[tuple[str, str]] = frozenset(
    {
        # PromptCollectionManagerModal is a standalone ModalScreen -- its
        # own small compose tree for the collection-picker dialog -- not
        # LibraryScreen. Its self.refresh(recompose=True)
        # (prompt_collection_manager_modal.py:297, inside _refresh())
        # rebuilds the modal, never the Library screen's nav bar/footer/
        # rail/canvas this ratchet exists to police. It predates the
        # Library_Modules surface widening (TASK-3, 2026-09): the file
        # already lived here, unrelated to PR 0a's move, before this
        # census ever looked at the directory.
        ("prompt_collection_manager_modal.py", "PromptCollectionManagerModal"),
    }
)

_RATCHET_GUIDANCE = """
The Library surface (library_screen.py + UI/Library_Modules/*.py) gained
statement-level whole-screen recompose sites: {count} found, {maximum}
allowed.

Every site the census currently sees, in file:line order. The ones your
change added are in this list -- diff it against the same listing on your
merge base to find them:

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


def _enclosing_class(source: str, lineno: int) -> str | None:
    """Return the innermost class enclosing ``lineno``, or ``None`` at module scope.

    Used to apply ``_CENSUS_EXEMPT_CLASS_SITES``: an exemption key is
    ``(filename, class_name)``, so a site written directly in a
    module-level function (``class_name`` is ``None``) can never match an
    exemption -- only a site textually inside a named class can be
    exempted.
    """
    tree = ast.parse(source)
    best_name: str | None = None
    best_start = -1
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        end = getattr(node, "end_lineno", node.lineno)
        # Innermost == the containing class that starts latest.
        if node.lineno <= lineno <= end and node.lineno > best_start:
            best_name = node.name
            best_start = node.lineno
    return best_name


def _non_exempt_sites(filename: str, source: str) -> list[tuple[int, str, str]]:
    """Return ``(lineno, function, spelling)`` for ``source``'s sites, minus exemptions.

    A site is dropped when ``(filename, enclosing class)`` is listed in
    ``_CENSUS_EXEMPT_CLASS_SITES``. Everything else -- unlisted classes,
    module-level functions, and library_screen.py's own LibraryScreen
    methods -- still counts.
    """
    sites: list[tuple[int, str, str]] = []
    for lineno, _receiver, spelling in find_library_whole_screen_recompose_statements(
        source
    ):
        class_name = _enclosing_class(source, lineno)
        if (filename, class_name) in _CENSUS_EXEMPT_CLASS_SITES:
            continue
        function = _enclosing_function(source, lineno)
        sites.append((lineno, function, spelling))
    return sites


def _widened_library_surface_sites() -> list[tuple[str, int, str, str]]:
    """Return ``(filename, lineno, function, spelling)`` across the whole surface.

    Scans every path in ``_LIBRARY_SURFACE_PATHS`` (library_screen.py +
    UI/Library_Modules/*.py), in path order, applying
    ``_CENSUS_EXEMPT_CLASS_SITES`` per file.
    """
    sites: list[tuple[str, int, str, str]] = []
    for path in _LIBRARY_SURFACE_PATHS:
        source = path.read_text(encoding="utf-8")
        for lineno, function, spelling in _non_exempt_sites(path.name, source):
            sites.append((path.name, lineno, function, spelling))
    return sites


def test_library_screen_whole_screen_recompose_count_is_ratcheted() -> None:
    """The whole-screen recompose statement count must not regrow.

    Scans the WHOLE Library surface -- library_screen.py plus every module
    under UI/Library_Modules/ (see the module docstring) -- as one census,
    so moving code between these files can neither drain nor inflate the
    count for free.
    """
    sites = _widened_library_surface_sites()
    count = len(sites)
    assert count > 0, (
        "The census found zero whole-screen recompose statements -- the "
        "counter is no longer measuring its subject (renamed receiver, "
        "moved module, or an over-broad exemption?). Fix the counter "
        "before trusting the ratchet."
    )
    if count > LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX:
        # Name the sites. A bare count tells an author that they broke the
        # ratchet but not where, across a dozen-plus files (review round, m5).
        inventory = "\n".join(
            f"  {filename}:{lineno}  {function}()  ->  {spelling}"
            for filename, lineno, function, spelling in sites
        )
        raise AssertionError(
            _RATCHET_GUIDANCE.format(
                count=count,
                maximum=LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX,
                inventory=inventory,
            )
        )


def test_census_pin_is_not_left_slack() -> None:
    """TASK-27019: the recorded pin should track reality, not drift above it.

    A ceiling-only ratchet lets a wave that lowers the real count without
    also lowering the pin quietly buy the next feature headroom it was
    never meant to have -- this happened to this exact pin twice (107 -> 80,
    then 74 -> 63; see the drift history in ``LIBRARY_WHOLE_SCREEN_RECOMPOSE_
    MAX``'s docstring). Mirrors ``Tests/Architecture/test_screen_size_
    ratchet.py``'s ``test_budget_is_not_left_slack_after_a_wave``, with the
    tolerance re-derived for this census rather than copied -- see
    ``_CENSUS_SLACK_TOLERANCE`` for why the size ratchet's absolute number
    does not transfer and how this one was chosen instead.
    """
    count = len(_widened_library_surface_sites())
    slack = LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX - count

    assert slack <= _CENSUS_SLACK_TOLERANCE, (
        f"The Library recompose census pin is {slack} sites above the "
        f"measured count ({count} vs pin {LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX}), "
        f"more than the {_CENSUS_SLACK_TOLERANCE}-site tolerance "
        f"(Tests/UI/test_library_recompose_ratchet.py, "
        f"_CENSUS_SLACK_TOLERANCE). A wave landed without lowering the "
        f"ratchet -- set LIBRARY_WHOLE_SCREEN_RECOMPOSE_MAX to {count} in "
        f"the same PR so the gain is locked in, per this file's module "
        f"docstring and TASK-27019."
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


def test_census_exemption_matches_only_the_documented_pair() -> None:
    """The (filename, class) exemption is precise, not a class-name-only match.

    Guards the exemption mechanism itself: the SAME class name in a
    different file, and a DIFFERENT class in the exempted file, must both
    still count. Only the exact pair in ``_CENSUS_EXEMPT_CLASS_SITES`` is
    dropped. Both classes below share an identically-named ``_refresh``
    method on purpose, so a function-name check couldn't tell them apart --
    only the (filename, class) exemption can.
    """
    source = """
class PromptCollectionManagerModal:
    def _refresh(self):
        self.refresh(recompose=True)

class OtherModal:
    def _refresh(self):
        self.refresh(recompose=True)

def module_level(self):
    self.refresh(recompose=True)
"""
    raw_sites = find_library_whole_screen_recompose_statements(source)
    assert len(raw_sites) == 3

    modal_lineno = next(
        lineno
        for lineno, _receiver, _spelling in raw_sites
        if _enclosing_class(source, lineno) == "PromptCollectionManagerModal"
    )
    other_linenos = {lineno for lineno, _receiver, _spelling in raw_sites} - {
        modal_lineno
    }
    assert len(other_linenos) == 2

    exempt_file = "prompt_collection_manager_modal.py"
    other_file = "some_other_file.py"

    # In the documented file: the modal's site is dropped; OtherModal's
    # identically-named method and the module-level function still count.
    exempt_linenos = {
        lineno for lineno, _function, _spelling in _non_exempt_sites(exempt_file, source)
    }
    assert exempt_linenos == other_linenos
    assert modal_lineno not in exempt_linenos

    # Same class name, different filename -- the pair doesn't match, so
    # nothing is dropped: all 3 sites count, including the modal's.
    all_linenos = {
        lineno for lineno, _function, _spelling in _non_exempt_sites(other_file, source)
    }
    assert all_linenos == other_linenos | {modal_lineno}


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
