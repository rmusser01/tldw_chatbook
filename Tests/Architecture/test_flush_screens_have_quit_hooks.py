"""A Screen that flushes pending work on navigation also answers the quit walk.

TASK-34000.1 (review N-01). ``flush_pending_work`` is the app's awaited
"persist or veto" seam, but only screen NAVIGATION awaits it
(``app_navigation.py``). Quitting is a different walk: Ctrl+Q asks the
active screen's ``confirm_quit`` / ``prepare_for_quit`` and nothing else
(``quit_confirmation_screens`` in ``Widgets/confirmation_dialog.py``). A
screen that defines the first without the second therefore saves on a tab
switch and silently drops the very same edit on Ctrl+Q -- which is exactly
how Library ▸ Notes lost the text typed since its last autosave, reproduced
four times by the review and once by its verifier.

So this scans, statically: every class under ``tldw_chatbook/`` that is a
Screen (a base whose name ends in ``Screen``) and defines
``flush_pending_work`` must also define a zero-argument ``confirm_quit`` --
the hook the quit walk calls. Widgets (Library's Folder files workspace) are
not walked by the quit flow; their owning screen answers for them.

``_KNOWN_GAPS`` freezes the screens that already had the gap when this guard
landed. It may only shrink: an entry whose screen now defines
``confirm_quit``, or no longer exists, fails ``test_known_gaps_are_not_stale``
so it is removed in the same change.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "tldw_chatbook"

#: Screens that defined ``flush_pending_work`` without ``confirm_quit`` on
#: 2026-10-03 (TASK-34000.1). Each still drops its pending work on Ctrl+Q,
#: and each has its own task. Two of them prompt from their flush through
#: ``push_screen_wait``, which a quit hook may not do (TASK-33622.10, use
#: ``await_quit_prompt``). The third pushes nothing at all.
_KNOWN_GAPS: dict[str, str] = {
    "tldw_chatbook/UI/Screens/research_workspace_screen.py:ResearchWorkspaceScreen": (
        "TASK-34000.44: quick-note draft; its flush asks through "
        "push_screen_wait (ResearchNoteSwitchRecoveryModal, in a worker)"
    ),
    "tldw_chatbook/UI/Screens/stts_screen.py:STTSScreen": (
        "TASK-34000.45: Studio preference draft; its flush asks through "
        "push_screen_wait (StudioTTSLeaveModal)"
    ),
    "tldw_chatbook/UI/Screens/workflows_screen.py:WorkflowsScreen": (
        "TASK-34000.46: its flush pushes nothing; it flushes the drafts and "
        "vetoes on DraftWriteFailed. The quit flow calls the app's "
        "workflow-authoring owner's prepare_quit; the controller-only "
        "draft path has no quit hook"
    ),
}

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):  # ModalScreen[bool]
        return _base_name(base.value)
    return ""


def _takes_no_arguments(function: _FunctionNode) -> bool:
    args = function.args
    return (
        len(args.posonlyargs) + len(args.args) == 1
        and not args.kwonlyargs
        and args.vararg is None
        and args.kwarg is None
    )


def scan_source(source: str, label: str) -> tuple[list[str], list[str]]:
    """Scan one module's classes.

    Args:
        source: The module's source text.
        label: How findings name the module.

    Returns:
        ``(flushing_screens, offences)`` as ``label:Class``: every Screen
        that defines ``flush_pending_work``, and those of them without a
        zero-argument ``confirm_quit``.
    """
    flushing: list[str] = []
    offences: list[str] = []
    for cls in (node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ClassDef)):
        if not any(_base_name(base).endswith("Screen") for base in cls.bases):
            continue
        methods = {
            node.name: node
            for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if "flush_pending_work" not in methods:
            continue
        flushing.append(f"{label}:{cls.name}")
        confirm = methods.get("confirm_quit")
        if confirm is None or not _takes_no_arguments(confirm):
            offences.append(f"{label}:{cls.name}")
    return flushing, offences


@lru_cache(maxsize=1)  # one tree walk per run; the tests only read it
def _scan_tree() -> tuple[list[str], list[str]]:
    flushing: list[str] = []
    offences: list[str] = []
    for path in sorted(PACKAGE.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "flush_pending_work" not in source:
            continue
        found_flushing, found_offences = scan_source(
            source, str(path.relative_to(REPO_ROOT))
        )
        flushing.extend(found_flushing)
        offences.extend(found_offences)
    return flushing, offences


def test_a_flushing_screen_defines_a_quit_hook() -> None:
    _flushing, offences = _scan_tree()
    new = sorted(set(offences) - set(_KNOWN_GAPS))
    assert not new, (
        "These Screens persist pending work before navigation "
        "(flush_pending_work) but never answer the quit walk, so Ctrl+Q "
        "drops the same edit a tab switch would save (TASK-34000.1). Define "
        "a zero-argument confirm_quit that flushes, and asks through "
        "await_quit_prompt when the flush is vetoed (see "
        "LibraryScreen.confirm_quit):\n  " + "\n  ".join(new)
    )


def test_the_scan_reaches_the_screens_it_guards() -> None:
    """A scan that finds no flushing screens would pass vacuously."""
    flushing, _offences = _scan_tree()
    assert "tldw_chatbook/UI/Screens/library_screen.py:LibraryScreen" in flushing
    assert "tldw_chatbook/UI/Screens/settings_screen.py:SettingsScreen" in flushing
    # A widget's flush is its owning screen's business, not the quit walk's.
    assert not any("LibraryFileNotesWorkspace" in entry for entry in flushing)


def test_known_gaps_are_not_stale() -> None:
    """The frozen gap list only shrinks: a fixed or removed screen leaves it."""
    flushing, offences = _scan_tree()
    stale = sorted(
        entry for entry in _KNOWN_GAPS if entry not in flushing or entry not in offences
    )
    assert not stale, (
        "These screens now define confirm_quit (or are gone); remove them "
        "from _KNOWN_GAPS:\n  " + "\n  ".join(stale)
    )


_FIXTURE = """
class Gap(BaseAppScreen):
    async def flush_pending_work(self):
        return True

class Covered(BaseAppScreen):
    async def flush_pending_work(self):
        return True

    async def confirm_quit(self):
        return await self.flush_pending_work()

class Unreachable(ModalScreen[bool]):
    async def flush_pending_work(self):
        return True

    async def confirm_quit(self, controller):
        return True

class Widget(Vertical):
    async def flush_pending_work(self):
        return True
"""


def test_negative_control_the_scan_catches_a_flush_without_a_quit_hook() -> None:
    flushing, offences = scan_source(_FIXTURE, "fixture")
    assert flushing == ["fixture:Gap", "fixture:Covered", "fixture:Unreachable"]
    # The quit walk calls confirm_quit() with no arguments, so a hook that
    # needs one is not a quit hook at all.
    assert offences == ["fixture:Gap", "fixture:Unreachable"]
