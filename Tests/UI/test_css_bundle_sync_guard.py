"""TASK-395: the CSS-bundle reproducibility guard.

`check_bundle_sync` rebuilds the bundle and fails when the committed one does not
reproduce from its sources (ignoring the `Generated:` timestamp), naming the
drifted module. Also asserts the repo's own committed bundle is currently in sync.
"""

from tldw_chatbook.css import check_bundle_sync as guard


def _bundle(overrides_body: str, timestamp: str = "2026-01-01 00:00:00") -> str:
    return (
        f"/* GENERATED */\n * Generated: {timestamp}\n\n"
        "/* ===== MODULE: core/_variables.tcss ===== */\n$x: 1;\n"
        f"/* ===== MODULE: utilities/_overrides.tcss ===== */\n{overrides_body}\n"
    )


def test_drifted_modules_ignores_the_generated_timestamp():
    """A faithful rebuild differing only in the timestamp reports no drift."""
    committed = _bundle("Tooltip { border: none; }", timestamp="2026-01-01 00:00:00")
    rebuilt = _bundle("Tooltip { border: none; }", timestamp="2026-07-24 06:31:20")
    assert guard.drifted_modules(committed, rebuilt) is None


def test_drifted_modules_names_the_changed_module():
    """A source edited without regenerating names the drifted MODULE block."""
    committed = _bundle("Tooltip { border: none; }")
    rebuilt = _bundle("Tooltip { border: round $primary; }")
    assert guard.drifted_modules(committed, rebuilt) == ["utilities/_overrides.tcss"]


def test_committed_bundle_reproduces_from_sources():
    """The repo's committed bundle is in sync (and the guard passes end-to-end)."""
    assert guard.main() == 0
