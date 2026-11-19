"""TASK-18927: fs_* local-tool self-recovery wave.

Every recovery message is bounded, labeled as a tool-generated hint, and
changes no permission behavior: admission (``resolve_workspace_path`` and
the sensitive-path exclusions) runs exactly as before, and the near-miss
probes only ever name files the filtered walk was already willing to
disclose.
"""

from pathlib import Path

import pytest
import tempfile

from tldw_chatbook.Tools import local_tool_impls
from tldw_chatbook.Tools.local_tool_impls import (
    HINT_LABEL,
    LocalToolError,
    _MAX_HINT_CHARS,
    _glob_relative_files,
    _grep_relative_files,
    edit_file,
    glob_files,
    grep_files,
    write_file,
)
from tldw_chatbook.Tools.patch_tool_impls import (
    FilesystemPatchError,
    apply_patch_to_text,
    parse_patch_targets,
    patch_files,
)
from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

# ---------------------------------------------------------------------------
# fs_edit (AC#1: already-applied no-op; AC#2: whitespace/nearest diagnosis)
# ---------------------------------------------------------------------------


def _ws(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


def test_fs_edit_already_applied_returns_success_noop(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text(
        "alpha REPLACED_MARKER gamma\n", encoding="utf-8"
    )

    result = edit_file(
        "f.txt", "beta", "REPLACED_MARKER", workspace_root=ws
    )

    assert "already applied" in result
    assert "0 replacements" in result
    # No change made: the file is byte-identical.
    assert (ws / "f.txt").read_text(encoding="utf-8") == (
        "alpha REPLACED_MARKER gamma\n"
    )


def test_fs_edit_deletion_without_positive_marker_keeps_error(tmp_path):
    # A deletion edit (empty new_string) has no positive "already applied"
    # marker, so a failed match must stay an error.
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("alpha beta gamma\n", encoding="utf-8")
    with pytest.raises(LocalToolError, match="old_string not found"):
        edit_file("f.txt", "zzz", "", workspace_root=ws)


def test_fs_edit_no_match_diagnoses_whitespace_mismatch(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("def foo():\n    return 1\n", encoding="utf-8")
    # The same text with a tab instead of the four-space indent.
    with pytest.raises(LocalToolError) as caught:
        edit_file("f.txt", "\treturn 1", "x", workspace_root=ws)
    message = str(caught.value)
    assert HINT_LABEL in message
    assert "whitespace" in message


def test_fs_edit_no_match_names_nearest_line_with_visible_whitespace(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("one\ndef foo(x):\n    return x\nfour\n", encoding="utf-8")
    with pytest.raises(LocalToolError) as caught:
        edit_file("f.txt", "def foo(y):", "x", workspace_root=ws)
    message = str(caught.value)
    assert HINT_LABEL in message
    # The closest candidate line is named by number, whitespace-visible.
    assert "line 2:" in message
    assert "def·foo(x):" in message


def test_fs_edit_no_match_without_candidates_keeps_plain_error(tmp_path):
    # Pins the exact plain-error contract other suites rely on (e.g. the
    # workspace executor's payload equality test): nothing similar in the
    # file means byte-identical legacy message.
    ws = _ws(tmp_path)
    (ws / "note.txt").write_text("before", encoding="utf-8")
    with pytest.raises(LocalToolError) as caught:
        edit_file("note.txt", "missing", "after", workspace_root=ws)
    assert str(caught.value) == "old_string not found in note.txt"


def test_fs_edit_hint_is_bounded_on_a_huge_file(tmp_path):
    ws = _ws(tmp_path)
    (ws / "big.txt").write_text(
        "filler line that keeps repeating\n" * 5_000, encoding="utf-8"
    )
    with pytest.raises(LocalToolError) as caught:
        edit_file("big.txt", "no such marker at all", "x", workspace_root=ws)
    message = str(caught.value)
    assert len(message) < 100 + _MAX_HINT_CHARS + len(HINT_LABEL)
    if HINT_LABEL in message:
        hint = message.split(f"{HINT_LABEL}: ", 1)[1]
        assert len(hint) <= _MAX_HINT_CHARS


def test_fs_edit_already_applied_still_passes_the_same_admission(tmp_path):
    # AC#6: recovery never bypasses admission. The no-op path must resolve
    # the path with intent="write" exactly like a real edit.
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text(
        "finished-marker\n", encoding="utf-8"
    )
    calls: list[tuple[str, str]] = []
    real = local_tool_impls.resolve_workspace_path

    def spy(path, workspace, intent, **kwargs):
        calls.append((str(path), intent))
        return real(path, workspace, intent=intent, **kwargs)

    monkeypatch_target = local_tool_impls
    original = monkeypatch_target.resolve_workspace_path
    monkeypatch_target.resolve_workspace_path = spy
    try:
        result = edit_file(
            "f.txt", "old", "finished-marker", workspace_root=ws
        )
    finally:
        monkeypatch_target.resolve_workspace_path = original
    assert "already applied" in result
    assert calls == [("f.txt", "write")]


# ---------------------------------------------------------------------------
# fs_write (AC#3: post-write verification)
# ---------------------------------------------------------------------------


def test_fs_write_verifies_on_disk_bytes_and_reports_mismatch(tmp_path, monkeypatch):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("before", encoding="utf-8")

    def torn_write(target, data, **kwargs):
        # Simulate a torn/concurrent write: different bytes land on disk.
        target.write_bytes(b"torn!")
        return None

    monkeypatch.setattr(local_tool_impls, "_atomic_write_target", torn_write)
    with pytest.raises(LocalToolError, match="write verification failed"):
        write_file("f.txt", "after", workspace_root=ws)


def test_fs_write_success_summary_is_unchanged(tmp_path):
    ws = _ws(tmp_path)
    result = write_file("new.txt", "content!", workspace_root=ws)
    assert result == "wrote 8 characters to new.txt"
    assert (ws / "new.txt").read_bytes() == b"content!"


# ---------------------------------------------------------------------------
# fs_glob / fs_grep (AC#4: zero-match near-miss probing)
# ---------------------------------------------------------------------------

_NO_EXCLUSIONS: tuple[SensitiveExclusion, ...] = ()


def test_fs_glob_zero_match_suggests_closest_existing_names(tmp_path):
    ws = _ws(tmp_path)
    (ws / "src" / "utils").mkdir(parents=True)
    (ws / "src" / "utils" / "a.py").write_text("x = 1\n", encoding="utf-8")

    result = glob_files("src/utilz/*.py", workspace_root=ws)

    assert result.startswith("(no files matching 'src/utilz/*.py')")
    assert HINT_LABEL in result
    assert '"utils"' in result
    assert "src/utils/a.py" in result


def test_fs_glob_zero_match_without_close_names_stays_bare(tmp_path):
    ws = _ws(tmp_path)
    (ws / "a.txt").write_text("x\n", encoding="utf-8")
    result = glob_files("*.qzywx", workspace_root=ws)
    assert result == "(no files matching '*.qzywx')"


def test_fs_glob_near_miss_pool_only_contains_admissible_names(tmp_path):
    # AC#6: the probe walks the SAME filtered iteration, so a sensitive name
    # can never surface in a hint.
    ws = _ws(tmp_path)
    (ws / "secret.txt").write_text("s\n", encoding="utf-8")
    (ws / "secrets.py").write_text("s\n", encoding="utf-8")
    exclusions = (("name", "secret.txt"),)
    result = _glob_relative_files(
        "secred.txt",
        workspace=ws,
        max_results=10,
        sensitive_exclusions=exclusions,
    )
    assert result.startswith("(no files matching 'secred.txt')")
    if HINT_LABEL in result:
        assert "secret.txt" not in result
        assert "secrets.py" in result


def test_fs_grep_zero_match_reports_case_insensitive_near_misses(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("has Alpha inside\nplain line\n", encoding="utf-8")

    result = grep_files("alpha", workspace_root=ws)

    assert result.startswith("(no matches for 'alpha')")
    assert HINT_LABEL in result
    assert "case-insensitively" in result
    assert "f.txt:1" in result


def test_fs_grep_zero_match_without_near_misses_stays_bare(tmp_path):
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("plain line\n", encoding="utf-8")
    result = grep_files("zzqqxx", workspace_root=ws)
    assert result == "(no matches for 'zzqqxx')"


def test_fs_grep_ci_probe_respects_sensitive_exclusions(tmp_path):
    ws = _ws(tmp_path)
    (ws / "secret.txt").write_text("Alpha\n", encoding="utf-8")
    (ws / "ok.txt").write_text("Alpha\n", encoding="utf-8")
    exclusions = (("name", "secret.txt"),)
    result = _grep_relative_files(
        "alpha",
        workspace=ws,
        mode="content",
        max_results=10,
        sensitive_exclusions=exclusions,
    )
    assert result.startswith("(no matches for 'alpha')")
    assert HINT_LABEL in result
    assert "secret.txt" not in result
    assert "ok.txt:1" in result


# ---------------------------------------------------------------------------
# fs_patch (AC#7: context mismatch names the nearest candidate)
# ---------------------------------------------------------------------------

MODIFY_DIFF_WRONG_CONTEXT = """\
--- a/notes.txt
+++ b/notes.txt
@@ -1,3 +1,3 @@
 alpha
-beta
+BETA
 gamma
"""


def test_fs_patch_context_mismatch_names_nearest_candidate(tmp_path):
    ws = _ws(tmp_path)
    # beta moved to line 4: the patch expects it at line 2.
    (ws / "notes.txt").write_text("alpha\ngamma\ndelta\nbeta\n", encoding="utf-8")

    with pytest.raises(LocalToolError) as caught:
        patch_files(MODIFY_DIFF_WRONG_CONTEXT, workspace_root=ws)

    message = str(caught.value)
    assert "patch_context_mismatch" in message
    assert HINT_LABEL in message
    assert "the expected text is at line 4" in message
    # Nothing was written.
    assert (ws / "notes.txt").read_text(encoding="utf-8") == "alpha\ngamma\ndelta\nbeta\n"


def test_fs_patch_context_mismatch_reason_code_unchanged(tmp_path):
    # The reason-code contract is shared with tldw_server's reference; the
    # diagnosis is additive only.
    ws = _ws(tmp_path)
    (ws / "notes.txt").write_text("nothing similar here\n", encoding="utf-8")
    with pytest.raises(LocalToolError) as caught:
        patch_files(MODIFY_DIFF_WRONG_CONTEXT, workspace_root=ws)
    assert caught.value.args[0].startswith("fs_patch failed [patch_context_mismatch]")


def test_patch_mismatch_detail_on_a_text_with_no_candidates_is_none():
    parsed = parse_patch_targets(MODIFY_DIFF_WRONG_CONTEXT)
    with pytest.raises(FilesystemPatchError) as caught:
        apply_patch_to_text("unrelated content\n", parsed[0])
    exc = caught.value
    assert exc.reason_code == "patch_context_mismatch"
    # "beta" vs "unrelated content": nothing close enough — the legacy
    # bare code path (no detail) survives.
    assert exc.detail is None


# ---------------------------------------------------------------------------
# Bounds and labeling (AC#5)
# ---------------------------------------------------------------------------


def test_hint_helpers_bound_and_label():
    from tldw_chatbook.Tools.local_tool_impls import (
        _bound_hint as bound_hint,
        _visualize_whitespace as visualize,
    )

    assert bound_hint("x" * 10_000) == "x" * (_MAX_HINT_CHARS - 1) + "…"
    assert visualize("a b\tc") == "a·b\\tc"
    assert HINT_LABEL == "hint (tool-generated)"


def test_fs_edit_binary_file_keeps_clean_error(tmp_path):
    # Adversarial: binary content must produce the existing clean refusal,
    # not a crash or a hint that replays binary bytes.
    ws = _ws(tmp_path)
    (ws / "bin.dat").write_bytes(b"\x00\x01\x02\xff\xfe")
    with pytest.raises(LocalToolError, match="not valid UTF-8"):
        edit_file("bin.dat", "x", "y", workspace_root=ws)


def test_fs_edit_whitespace_only_old_string_gets_no_hint(tmp_path):
    # Pathological old_string: nothing diagnosable, plain error.
    ws = _ws(tmp_path)
    (ws / "f.txt").write_text("content\n", encoding="utf-8")
    with pytest.raises(LocalToolError) as caught:
        edit_file("f.txt", "   \n\t", "x", workspace_root=ws)
    assert str(caught.value) == "old_string not found in f.txt"


# --- review fix (PR #3086): adversarial coincidental-short cases --------------


def test_coincidental_short_new_string_is_not_reported_as_already_applied():
    """PR-review regression: presence of a SHORT new_string alone must not
    earn the "edit already applied" success (the reviewer reproduced
    "x" inside "item_x" returning a false no-op success). Weak evidence
    falls to the ambiguity error path instead."""
    import pytest

    from tldw_chatbook.Tools.local_tool_impls import LocalToolError

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "one.txt").write_text("prefix item_x suffix content")
        with pytest.raises(LocalToolError) as excinfo:
            edit_file(
                "one.txt",
                "FIXME_MARKER",
                "x",
                workspace_root=root,
                replace_all=True,
            )
        assert "old_string not found" in str(excinfo.value)
        assert "ambiguous" in str(excinfo.value)


def test_coincidental_single_short_new_string_is_not_reported_applied():
    """The single-occurrence variant of the same false-success shape."""
    import pytest

    from tldw_chatbook.Tools.local_tool_impls import LocalToolError

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "two.txt").write_text("prefix item_x suffix content")
        with pytest.raises(LocalToolError) as excinfo:
            edit_file(
                "two.txt", "MISSING_OLD", "ok", workspace_root=root
            )
        assert "old_string not found" in str(excinfo.value)


def test_distinctive_new_string_presence_still_reports_already_applied():
    """The gate must not over-correct: a genuinely distinctive replacement
    present exactly once with the old text absent keeps the no-op success."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "three.txt").write_text(
            "the distinctive replacement landed here"
        )
        result = edit_file(
            "three.txt",
            "original text",
            "the distinctive replacement",
            workspace_root=root,
        )
        assert "already applied" in result
