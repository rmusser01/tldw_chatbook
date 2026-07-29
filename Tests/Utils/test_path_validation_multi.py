"""Multi-root path validation (spec 2026-07-26 settings-workspaces §3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Utils.path_validation import (
    ROOT_DENIAL_RECOVERY_HINT,
    ROOT_DENIAL_RECOVERY_POINTER,
    validate_path_multi,
)


def test_accepts_path_inside_any_root(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    (root_b / "sub").mkdir(parents=True)
    root_a.mkdir()
    target = root_b / "sub" / "file.txt"
    target.write_text("x")

    assert validate_path_multi(target, [root_a, root_b]) == target.resolve()


def test_relative_paths_resolve_against_first_root(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "note.txt").write_text("x")

    resolved = validate_path_multi("note.txt", [sandbox, tmp_path / "other"])
    assert resolved == (sandbox / "note.txt").resolve()


def test_rejection_names_all_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(outside, [root_a, root_b])
    message = str(excinfo.value)
    assert str(root_a) in message and str(root_b) in message


def test_empty_roots_rejected() -> None:
    with pytest.raises(ValueError):
        validate_path_multi("anything", [])


def test_rejection_teaches_the_recovery_route(tmp_path: Path) -> None:
    """TASK-1231/F3 AC1: the denial must not be a dead end.

    Pre-fix, this message named every consulted root and stopped there --
    a first-run user on the (unbound) Default workspace had no route to the
    fix. Both the short pointer and the fuller explanation must actually be
    present, verbatim.
    """
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(outside, [root])
    message = str(excinfo.value)
    assert ROOT_DENIAL_RECOVERY_POINTER in message
    assert ROOT_DENIAL_RECOVERY_HINT in message
    assert "Settings > Workspaces" in message
    assert "Default workspace cannot hold folder bindings" in message


def test_recovery_pointer_is_accurate_for_a_non_default_workspace(
    tmp_path: Path,
) -> None:
    """Qodo PR #1074 finding 3: this denial fires identically whether the
    run's workspace is Default or an already-named workspace that simply
    has no folder bound yet -- `validate_path_multi` has no workspace
    context to tell the two apart (see `ROOT_DENIAL_RECOVERY_POINTER`'s own
    docstring for why threading it in would be invasive). Pre-fix, the
    pointer unconditionally said "create a workspace + bind a folder",
    which is actively wrong advice for a run already in a normal
    workspace -- it only needs a folder bound to the one it already has.

    The pointer must therefore read correctly on its own regardless of
    which workspace the run is in (just "bind a folder"); the
    Default-specific advice is confined to the HINT and phrased as an
    explicit conditional ("if this run is in Default"), never asserted as
    fact.
    """
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(outside, [root])
    message = str(excinfo.value)

    assert "bind a folder" in ROOT_DENIAL_RECOVERY_POINTER
    assert "create a workspace" not in ROOT_DENIAL_RECOVERY_POINTER.lower()
    assert "if this run is in default" in ROOT_DENIAL_RECOVERY_HINT.lower()
    # The pointer alone (before any Default-specific caveat) must already
    # be present and correct in the composed message.
    assert message.index(ROOT_DENIAL_RECOVERY_POINTER) < message.index(
        ROOT_DENIAL_RECOVERY_HINT
    )


def test_recovery_pointer_precedes_everything_else(tmp_path: Path) -> None:
    """TASK-1231/F3 AC1 (round 1 review CRITICAL 2): the SHORT pointer must
    be the first thing after the bare denial acknowledgement -- before the
    path is even repeated, before the fuller explanation, and before the
    (open-ended-length) consulted-root list. Any of those three could push
    a LATER recovery route past the transcript's 160-char tool-step-marker
    truncation (``console_agent_bridge._STEP_MARKER_RESULT_LIMIT``); only
    "comes first" is safe regardless of how long any of them is.
    """
    root = tmp_path / "sandbox"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("x")

    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(outside, [root])
    message = str(excinfo.value)
    pointer_index = message.index(ROOT_DENIAL_RECOVERY_POINTER)
    path_index = message.index(str(outside))
    hint_index = message.index(ROOT_DENIAL_RECOVERY_HINT)
    roots_index = message.index(str(root.resolve()))
    assert pointer_index < path_index
    assert pointer_index < hint_index
    assert pointer_index < roots_index


@pytest.mark.parametrize(
    "tool_prefix",
    [
        "ERROR: Failed to read file: ",
        "ERROR: Failed to write file: ",
        "ERROR: Failed to list directory: ",
    ],
)
@pytest.mark.parametrize(
    "path_len_label,path_factory",
    [
        ("~44 chars", lambda tmp_path: str(tmp_path / ("a" * 20) / "secret.txt")),
        ("~60 chars", lambda tmp_path: str(tmp_path / ("a" * 36) / "secret.txt")),
    ],
)
def test_recovery_pointer_survives_real_transcript_truncation(
    tmp_path: Path, tool_prefix: str, path_len_label: str, path_factory
) -> None:
    """TASK-1231/F3 AC1 (round 1 review CRITICAL 2, regression).

    Runs the REAL composed error -- exactly the string a file tool's
    ``except`` handler builds (``f"Failed to {op}: {validate_path_multi's
    ValueError}"``), wrapped in ``agent_runtime``'s own ``"ERROR: "``
    prefix -- through the ACTUAL Console transcript truncation
    (``console_agent_bridge._truncate_step_text`` at
    ``_STEP_MARKER_RESULT_LIMIT``, 160 chars) for a realistic 40-60-char
    path. Pre-fix, "Settings > Workspaces" never survived this for ANY
    path in this range -- the recovery text existed but was entirely
    invisible in the transcript a user or the model actually sees.
    """
    from tldw_chatbook.Chat.console_agent_bridge import (
        _STEP_MARKER_RESULT_LIMIT,
        _truncate_step_text,
    )

    root = tmp_path / "sandbox"
    root.mkdir()
    outside = path_factory(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        validate_path_multi(outside, [root])
    composed = tool_prefix + str(excinfo.value)
    truncated = _truncate_step_text(composed, limit=_STEP_MARKER_RESULT_LIMIT)
    visible = truncated.split("…")[0]

    assert "Settings > Workspaces" in visible, (
        f"recovery pointer did not survive truncation for {path_len_label} "
        f"path with prefix {tool_prefix!r}: {truncated!r}"
    )
    assert ROOT_DENIAL_RECOVERY_POINTER in visible
