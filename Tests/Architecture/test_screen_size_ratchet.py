"""A one-way ratchet on the screens being decomposed.

**Why this test exists.** Between 2026-08-02 and 2026-08-06 the Console
decomposition extracted ~4,900 lines out of `chat_screen.py` across two
reviewed waves — and the file ended up *larger* than when the work started,
because ~5,500 lines of concurrent feature work landed in it over the same
window. Every one of those lines went into the screen because the screen was
the path of least resistance: `UI/Console_Modules/` did not exist yet, or was
not yet the obvious place to put things.

Extraction alone therefore cannot win. This test makes the screen's current
size a *ceiling* rather than a waypoint, so a wave's gain cannot be silently
re-consumed by the next feature.

**This is a ratchet, not a limit.** The budgets below may only ever go DOWN.
When a wave lands, lower them to the new measurement in the same PR. If you
are here because CI failed, do NOT raise a number to make it pass — that
defeats the entire mechanism and re-opens the hole this test was written to
close.

**What to do when this test fails.** Your new Console code belongs in
`tldw_chatbook/UI/Console_Modules/`, next to `workspace.py`, `session.py`,
`hands_free.py` and `dictation.py`. `DESIGN.md` §7 states the rule and the
binding contract those controllers follow. A region that owns pixels becomes
a widget; behaviour and state with no region become a controller. Both take
their dependencies as named constructor callables rather than reaching back
through the screen.

Method count is tracked alongside line count deliberately: a screen can be
made shorter without being made simpler (by compressing bodies), and it is
the number of responsibilities the class holds — not its character count —
that made it hard to change.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: path -> (class name, max lines, max methods in that class).
#: LOWER these when a decomposition wave lands. Never raise them.
#: Lowered 2026-08-06 at the wave-3 close (message + transcript + prompts):
#: 20,964/612 -> 18,905/598. First recorded immediately after wave 2 (PR #1381).
#: The odd trailing 5 is real and worth leaving: the wave earned 18,904, and
#: `c1c9146b7` on dev added a net line during the merge-day rebase. Measured
#: after the rebase, not before -- a budget set against a stale base is a
#: budget that fails the moment it lands.
_BUDGETS: dict[str, tuple[str, int, int]] = {
    "tldw_chatbook/UI/Screens/chat_screen.py": ("ChatScreen", 18905, 598),
}


def _measure(rel_path: str, class_name: str) -> tuple[int, int]:
    """Line count of a module and method count of one class inside it.

    Args:
        rel_path: Repo-relative path to the module.
        class_name: The dominant class whose methods are counted.

    Returns:
        tuple[int, int]: ``(module line count, class method count)``.

    Raises:
        AssertionError: If the module or the named class is missing — either
            means the budget entry is stale and must be updated deliberately
            rather than silently skipped.
    """
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} not found; the budget entry is stale."
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert classes, f"class {class_name} not found in {rel_path}; budget stale."
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return len(source.splitlines()), len(methods)


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_screen_does_not_grow_past_its_budget(rel_path: str) -> None:
    class_name, max_lines, max_methods = _BUDGETS[rel_path]
    lines, methods = _measure(rel_path, class_name)

    guidance = (
        f"\n\n{rel_path} is under a one-way size ratchet "
        f"(Tests/Architecture/test_screen_size_ratchet.py).\n"
        f"New Console code belongs in tldw_chatbook/UI/Console_Modules/ — see "
        f"DESIGN.md section 7 for the region-vs-controller rule and the "
        f"dependency-binding contract.\n"
        f"Do NOT raise the budget to make this pass. Lower it when a "
        f"decomposition wave lands."
    )

    assert lines <= max_lines, (
        f"{rel_path} grew to {lines} lines (budget {max_lines}, "
        f"+{lines - max_lines}).{guidance}"
    )
    assert methods <= max_methods, (
        f"{class_name} grew to {methods} methods (budget {max_methods}, "
        f"+{methods - max_methods}).{guidance}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_budget_is_not_left_slack_after_a_wave(rel_path: str) -> None:
    """The recorded budget should track reality, not drift above it.

    A ratchet with slack silently permits regrowth up to the stale number, so
    a wave that forgets to lower its budget quietly buys the next feature
    headroom it was never meant to have. The tolerance is deliberately loose
    (200 lines / 10 methods) so ordinary in-file edits do not fail CI — it
    only fires when a decomposition landed and its budget was not updated.
    """
    class_name, max_lines, max_methods = _BUDGETS[rel_path]
    lines, methods = _measure(rel_path, class_name)

    assert max_lines - lines <= 200, (
        f"{rel_path} is {max_lines - lines} lines under its budget "
        f"({lines} vs {max_lines}). A wave landed without lowering the "
        f"ratchet — set it to {lines} so the gain is locked in."
    )
    assert max_methods - methods <= 10, (
        f"{class_name} is {max_methods - methods} methods under its budget "
        f"({methods} vs {max_methods}). Set it to {methods}."
    )
