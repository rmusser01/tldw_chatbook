"""Tests for the turn-file-card annotate loop's pure diff-hunk helpers.

TASK-16800 (spec `Docs/superpowers/specs/2026-08-17-console-turn-file-annotate-
design.md` §2/§4). Covers `DiffHunk`, `split_unified_diff`, `hunk_excerpt`,
`render_diff_feedback_block`, and `format_diff_feedback_disclosure` in
`tldw_chatbook/Chat/console_display_state.py`.

Segmentation is exercised against REAL `git diff -M` output captured from a
tmp repo with two commits (multi-hunk file, single-hunk file, a clean
rename, and a binary file) -- not hand-written diff fixtures. The whole
module is skipped when git is unavailable, matching the precedent in
`Tests/Tools/test_git_tool_impls.py`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from rich.cells import cell_len

from tldw_chatbook.Chat.console_display_state import (
    DiffHunk,
    format_diff_feedback_disclosure,
    hunk_excerpt,
    middle_elide_path,
    render_diff_feedback_block,
    split_unified_diff,
)

GIT_AVAILABLE = shutil.which("git") is not None
pytestmark = pytest.mark.skipif(
    not GIT_AVAILABLE, reason="git is not available on this system"
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _diff_text(repo: Path, sha1: str, sha2: str, *paths: str) -> str:
    result = subprocess.run(
        ["git", "diff", "-M", sha1, sha2, "--", *paths],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.fixture
def diff_fixture(tmp_path: Path) -> dict:
    """Real `git diff -M` text for four cases: multi-hunk, single-hunk,
    clean rename (no content change), and binary."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")

    multi_lines = [f"line{i}\n" for i in range(40)]
    (repo / "multi.py").write_text("".join(multi_lines), encoding="utf-8")
    (repo / "single.py").write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
    (repo / "old_name.txt").write_text(
        "unchanged content\nline two\n", encoding="utf-8"
    )
    (repo / "image.bin").write_bytes(bytes(range(10)) * 5)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "commit1")
    sha1 = _git_output(repo, "rev-parse", "HEAD")

    multi_lines[5] = "line5-CHANGED\n"
    multi_lines[30] = "line30-CHANGED\n"
    (repo / "multi.py").write_text("".join(multi_lines), encoding="utf-8")
    (repo / "single.py").write_text("a\nb\nc-changed\nd\ne\n", encoding="utf-8")
    (repo / "image.bin").write_bytes(bytes(range(9, -1, -1)) * 5)
    _git(repo, "mv", "old_name.txt", "new_name.txt")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "commit2")
    sha2 = _git_output(repo, "rev-parse", "HEAD")

    return {
        "multi": _diff_text(repo, sha1, sha2, "multi.py"),
        "single": _diff_text(repo, sha1, sha2, "single.py"),
        # Both the old and new path must be in the pathspec for `-M` to
        # correlate the deletion with the addition and report a clean
        # rename instead of "new file" -- verified live against real git
        # before writing this fixture.
        "rename": _diff_text(repo, sha1, sha2, "old_name.txt", "new_name.txt"),
        "binary": _diff_text(repo, sha1, sha2, "image.bin"),
    }


# --------------------------------------------------------------------------
# split_unified_diff
# --------------------------------------------------------------------------


def test_multi_hunk_diff_segments_into_two_hunks_with_verbatim_headers(diff_fixture):
    text = diff_fixture["multi"]
    hunks = split_unified_diff(text)
    assert len(hunks) == 2

    all_lines = text.splitlines()
    expected_header_1 = next(l for l in all_lines if l.startswith("@@ -3,7 +3,7 @@"))
    expected_header_2 = next(l for l in all_lines if l.startswith("@@ -28,7 +28,7 @@"))
    assert hunks[0].header == expected_header_1
    assert hunks[1].header == expected_header_2

    # Prelude is the "diff --git"/"index"/"---"/"+++" block, shared by
    # every hunk of this file.
    expected_prelude = "\n".join(all_lines[:4])
    assert hunks[0].file_prelude == expected_prelude
    assert hunks[1].file_prelude == expected_prelude
    assert all_lines[0].startswith("diff --git")
    assert all_lines[2] == "--- a/multi.py"
    assert all_lines[3] == "+++ b/multi.py"


def test_single_hunk_diff_segments_into_one_hunk(diff_fixture):
    text = diff_fixture["single"]
    hunks = split_unified_diff(text)
    assert len(hunks) == 1
    assert hunks[0].header.startswith("@@ -1,5 +1,5 @@")
    assert hunks[0].file_prelude == "\n".join(text.splitlines()[:4])


@pytest.mark.parametrize("key", ["multi", "single"])
def test_body_reassembly_equals_original_diff_lines(diff_fixture, key):
    """prelude + header + body per file reassembles the original diff,
    line for line, for text-file diffs."""
    text = diff_fixture[key]
    hunks = split_unified_diff(text)

    reassembled: list[str] = list(hunks[0].file_prelude.splitlines())
    for hunk in hunks:
        reassembled.append(hunk.header)
        reassembled.extend(hunk.body_lines)

    assert reassembled == text.splitlines()


def test_rename_only_diff_yields_single_fallback_hunk(diff_fixture):
    text = diff_fixture["rename"]
    assert "@@" not in text  # sanity: this really is a no-hunk diff
    hunks = split_unified_diff(text)
    assert hunks == [
        DiffHunk(header="", body_lines=tuple(text.splitlines()), file_prelude="")
    ]


def test_binary_diff_yields_single_fallback_hunk(diff_fixture):
    text = diff_fixture["binary"]
    assert "@@" not in text
    assert "Binary files" in text
    hunks = split_unified_diff(text)
    assert hunks == [
        DiffHunk(header="", body_lines=tuple(text.splitlines()), file_prelude="")
    ]


def test_empty_text_yields_single_fallback_hunk_with_empty_body():
    assert split_unified_diff("") == [
        DiffHunk(header="", body_lines=(), file_prelude="")
    ]


# --------------------------------------------------------------------------
# hunk_excerpt
# --------------------------------------------------------------------------


def test_hunk_excerpt_under_cap_has_no_elision_tail(diff_fixture):
    hunks = split_unified_diff(diff_fixture["single"])
    hunk = hunks[0]
    excerpt = hunk_excerpt(hunk, cap=40)
    assert excerpt == hunk.header + "\n" + "\n".join(hunk.body_lines)
    assert "more lines" not in excerpt


def test_hunk_excerpt_caps_body_and_adds_honest_tail(diff_fixture):
    hunks = split_unified_diff(diff_fixture["multi"])
    hunk = hunks[0]
    assert len(hunk.body_lines) == 8  # fixture invariant: 6 context + 1 remove + 1 add

    excerpt = hunk_excerpt(hunk, cap=3)
    lines = excerpt.splitlines()
    assert lines[0] == hunk.header
    assert lines[1:4] == list(hunk.body_lines[:3])
    assert lines[4] == "… 5 more lines"
    assert len(lines) == 5


def test_hunk_excerpt_default_cap_is_40():
    hunk = DiffHunk(
        header="@@ -1,2 +1,2 @@",
        body_lines=tuple(f"l{i}" for i in range(50)),
        file_prelude="",
    )
    excerpt = hunk_excerpt(hunk)
    lines = excerpt.splitlines()
    assert lines[1:41] == [f"l{i}" for i in range(40)]
    assert lines[41] == "… 10 more lines"


def test_hunk_excerpt_fallback_hunk_omits_empty_header_line(diff_fixture):
    hunks = split_unified_diff(diff_fixture["rename"])
    hunk = hunks[0]
    assert hunk.header == ""
    excerpt = hunk_excerpt(hunk, cap=2)
    lines = excerpt.splitlines()
    # No leading blank line for the empty header.
    assert lines[0] == hunk.body_lines[0]
    assert lines == [hunk.body_lines[0], hunk.body_lines[1], "… 2 more lines"]


# --------------------------------------------------------------------------
# hunk_excerpt byte cap (Qodo #5, PR #1779 fix round)
# --------------------------------------------------------------------------


def test_hunk_excerpt_default_byte_cap_is_4096():
    """A hunk whose (line-capped) rendering exceeds 4096 UTF-8 bytes is
    further truncated by the DEFAULT byte_cap, with no explicit argument
    needed -- the line cap alone (40 lines) does not bound bytes."""
    # 200 lines of 40 bytes each easily clears both the 40-line cap (so
    # only the first 40 lines would ever be line-cap-eligible) but this
    # exercises the byte cap on a body that's short enough in LINE COUNT
    # (well under 40) yet still oversized in BYTES -- one enormous single
    # line, the exact "minified file" shape the fix targets.
    huge_line = "x" * 20_000
    hunk = DiffHunk(header="@@ -1,1 +1,1 @@", body_lines=(huge_line,), file_prelude="")
    excerpt = hunk_excerpt(hunk)
    assert len(excerpt.encode("utf-8")) <= 4096
    assert excerpt.endswith("… truncated")
    assert excerpt.startswith("@@ -1,1 +1,1 @@\n" + "x" * 10)


def test_hunk_excerpt_under_byte_cap_is_unaffected():
    hunk = DiffHunk(
        header="@@ -1,1 +1,1 @@", body_lines=("short line",), file_prelude=""
    )
    excerpt = hunk_excerpt(hunk)
    assert excerpt == "@@ -1,1 +1,1 @@\nshort line"
    assert "truncated" not in excerpt


def test_hunk_excerpt_custom_byte_cap_truncates_at_line_boundary():
    """Whole lines are kept for as long as they fit; the ONE line that
    finally overflows the budget is included as a partial (byte-safe)
    prefix rather than dropped outright -- so the surviving content is
    exactly a prefix of the original header+body text, never content from
    a later line jumping ahead of a dropped one."""
    body = tuple(f"line{i}-{'y' * 20}" for i in range(20))  # ~25 bytes/line
    hunk = DiffHunk(header="@@ -1,1 +1,1 @@", body_lines=body, file_prelude="")
    excerpt = hunk_excerpt(hunk, byte_cap=120)
    assert len(excerpt.encode("utf-8")) <= 120
    assert excerpt.endswith("… truncated")

    tail = "\n… truncated"
    assert excerpt.endswith(tail)
    kept_text = excerpt[: -len(tail)]
    full_text = "\n".join([hunk.header, *body])
    assert full_text.startswith(kept_text)
    assert kept_text != full_text  # genuinely truncated, not the whole text


def test_hunk_excerpt_single_line_wider_than_byte_cap_hard_truncates_within_it():
    """The line-boundary preference degrades gracefully: when even the
    FIRST body line alone overflows byte_cap (no earlier newline to stop
    at), the excerpt still respects byte_cap via a hard truncation inside
    that one line, rather than either overflowing the cap or dropping the
    line's content entirely."""
    hunk = DiffHunk(header="", body_lines=("z" * 500,), file_prelude="")
    excerpt = hunk_excerpt(hunk, byte_cap=50)
    assert len(excerpt.encode("utf-8")) <= 50
    assert excerpt.endswith("… truncated")
    assert excerpt.startswith("z")


def test_hunk_excerpt_byte_cap_and_line_cap_both_apply():
    """The line cap's own "… N more lines" tail is itself subject to the
    byte cap when the surviving (line-capped) text is still oversized."""
    body = tuple(f"l{i}-{'a' * 30}" for i in range(60))
    hunk = DiffHunk(header="@@ -1,1 +1,1 @@", body_lines=body, file_prelude="")
    excerpt = hunk_excerpt(hunk, cap=40, byte_cap=200)
    assert len(excerpt.encode("utf-8")) <= 200
    assert excerpt.endswith("… truncated")


# --------------------------------------------------------------------------
# render_diff_feedback_block / format_diff_feedback_disclosure
# --------------------------------------------------------------------------


def _note(
    *,
    id: int = 1,
    run_id: str = "12345678-abcd-4a4a-9a9a-abcdefabcdef",
    root: str = "/ws",
    path: str = "a.py",
    hunk_index: int = 0,
    hunk_header: str = "@@ -1,4 +1,6 @@",
    hunk_excerpt: str = "some excerpt",
    note: str = "use the cached value here",
    created_at: str = "2026-08-17T00:00:00Z",
    delivered_at: str | None = None,
) -> dict:
    """Build a `change_notes` row dict, matching the columns Task 1's
    `AgentRunsDB.notes_for_run`/`pending_notes_for_conversation` return."""
    return {
        "id": id,
        "run_id": run_id,
        "root": root,
        "path": path,
        "hunk_index": hunk_index,
        "hunk_header": hunk_header,
        "hunk_excerpt": hunk_excerpt,
        "note": note,
        "created_at": created_at,
        "delivered_at": delivered_at,
    }


def _expected_entry(note: dict) -> str:
    """Independently reproduce spec §4's per-note entry format (heading
    excluded) -- duplicated here rather than imported from the module
    under test, so an exact-format test can't be satisfied by a change to
    the implementation's own formatting helper.

    Fenced with FOUR backticks (final-review fix wave): a three-backtick
    fence breaks when the excerpt itself contains a triple-backtick line
    (a markdown-file diff), so the implementation escapes one level up.
    """
    short_id = note["run_id"][:8]
    return (
        f"### {note['path']} — {note['hunk_header']}   [run {short_id}]\n"
        f"> {note['note']}\n"
        f"````\n{note['hunk_excerpt']}\n````"
    )


def test_render_diff_feedback_block_empty_notes_returns_empty_and_no_ids():
    assert render_diff_feedback_block([]) == ("", [])


def test_render_diff_feedback_block_exact_format_for_one_note():
    note = _note(
        id=7,
        run_id="12345678-abcd-4a4a-9a9a-abcdefabcdef",
        path="a.py",
        hunk_header="@@ -1,4 +1,6 @@",
        hunk_excerpt="some excerpt",
        note="use the cached value here",
    )
    block, included_ids = render_diff_feedback_block([note])
    expected = (
        "## Diff feedback from the user (on your earlier file changes)\n"
        "### a.py — @@ -1,4 +1,6 @@   [run 12345678]\n"
        "> use the cached value here\n"
        "````\n"
        "some excerpt\n"
        "````"
    )
    assert block == expected
    assert included_ids == [7]


def test_render_diff_feedback_block_four_backtick_fence_survives_triple_backtick_excerpt():
    """A hunk excerpt containing its OWN triple-backtick line (a diff of a
    markdown file with a fenced code block) must not break out of the
    entry's own fence -- the fence must be four backticks precisely so a
    three-backtick line inside the excerpt stays inert content rather than
    closing the fence early.
    """
    excerpt = "+```python\n+def foo():\n+    pass\n+```"
    note = _note(
        id=9,
        path="README.md",
        hunk_header="@@ -1,2 +1,4 @@",
        hunk_excerpt=excerpt,
        note="fenced code block added here",
    )
    block, included_ids = render_diff_feedback_block([note])
    assert included_ids == [9]
    expected_entry = _expected_entry(note)
    assert f"````\n{excerpt}\n````" in block
    assert block == (
        "## Diff feedback from the user (on your earlier file changes)\n"
        + expected_entry
    )
    # The excerpt's own ``` lines must survive verbatim, unescaped, inside
    # the four-backtick fence -- not stripped, not escaped.
    assert "+```python" in block
    assert "+```" in block.splitlines()[-2]


def test_render_diff_feedback_block_short_id_is_first_8_chars_of_run_id():
    note = _note(run_id="deadbeef-0000-0000-0000-000000000000")
    block, _ = render_diff_feedback_block([note])
    assert "[run deadbeef]" in block


def test_render_diff_feedback_block_includes_multiple_notes_oldest_first_under_cap():
    notes = [
        _note(id=1, path="a.py", note="first note"),
        _note(id=2, path="b.py", note="second note"),
        _note(id=3, path="c.py", note="third note"),
    ]
    block, included_ids = render_diff_feedback_block(notes, cap_bytes=1_000_000)
    assert included_ids == [1, 2, 3]
    assert block.index("a.py") < block.index("b.py") < block.index("c.py")
    assert "more notes held" not in block


def _equal_length_note(i: int) -> dict:
    """A `_note` whose rendered entry is the SAME byte length for every
    `i` (same-length path/header/note/excerpt) -- lets a test hand-derive
    exact block byte counts (139 / 217 / 295 bytes for 1 / 2 / 3 such
    notes, +78 bytes per note -- 76 bytes of text plus the 2 extra fence
    backticks the final-review fix wave added per entry) without
    depending on `render_diff_feedback_block` to compute its own expected
    values."""
    return _note(
        id=i,
        run_id="12345678-abcd-4a4a-9a9a-abcdefabcdef",
        path=f"p{i}.py",
        hunk_header="@@ -1,4 +1,6 @@",
        hunk_excerpt="excerptX",
        note="edit here",
    )


def test_render_diff_feedback_block_holdover_line_bytes_are_included_in_cap():
    """Regression: the holdover line's own bytes must fit under cap_bytes
    too -- appending it unconditionally after the per-note check let the
    final block exceed the cap (reviewer reproduced 441 bytes against a
    398-byte cap on the pre-fix version).

    Byte arithmetic (independent of the function under test, re-derived by
    hand from `_equal_length_note`'s fixed entry size after the
    final-review fix wave's four-backtick fence): 1/2/3-note blocks (no
    holdover) are 139 / 217 / 295 bytes; the holdover suffix "\\n\\n… N
    more notes held for the next message" is 44 bytes for any
    single-digit N (unaffected by the fence change -- it carries no
    backticks). At cap_bytes=218, two notes fit on their own (217 <
    218) -- the PRE-FIX (pre-holdover-fix) code would have stopped there
    and appended the 44-byte holdover unconditionally, producing a
    261-byte block that blows the 218-byte cap. The fix must instead
    notice 217 + 44 > 218, evict the second note, and land on the
    one-note-plus-holdover block (139 + 44 = 183 bytes), which fits.
    """
    notes = [_equal_length_note(1), _equal_length_note(2), _equal_length_note(3)]

    block, included_ids = render_diff_feedback_block(notes, cap_bytes=218)

    assert len(block.encode("utf-8")) <= 218, (
        "the rendered block, holdover line included, must never exceed cap_bytes"
    )
    assert included_ids == [1]
    entry_one = _expected_entry(notes[0])
    expected = (
        "## Diff feedback from the user (on your earlier file changes)\n"
        + entry_one
        + "\n\n… 2 more notes held for the next message"
    )
    assert block == expected
    assert len(expected.encode("utf-8")) == 183


def test_render_diff_feedback_block_holdover_never_exceeds_cap_across_boundary_caps():
    """Property sweep (not tied to one hand-picked cap): for every cap in
    the range where at least one note is excluded, the returned block
    (holdover line included) must stay within cap_bytes, and included_ids
    must remain an oldest-first prefix."""
    notes = [_equal_length_note(1), _equal_length_note(2), _equal_length_note(3)]
    for cap in range(145, 300, 7):
        block, included_ids = render_diff_feedback_block(notes, cap_bytes=cap)
        assert len(block.encode("utf-8")) <= cap or included_ids == [], (
            f"cap={cap} block exceeded cap_bytes with a non-empty inclusion set"
        )
        assert included_ids == [n["id"] for n in notes[: len(included_ids)]]
        if len(included_ids) < len(notes):
            held = len(notes) - len(included_ids)
            assert block.endswith(f"… {held} more notes held for the next message")


def test_render_diff_feedback_block_no_holdover_reserved_when_nothing_excluded():
    """Boundary: a cap that exactly fits all notes' own bytes (no note
    excluded) must not be short by the holdover line's size -- the reserve
    must only apply once exclusion is actually happening."""
    notes = [_equal_length_note(1), _equal_length_note(2), _equal_length_note(3)]
    full_block, full_ids = render_diff_feedback_block(notes, cap_bytes=1_000_000)
    full_bytes = len(full_block.encode("utf-8"))
    assert full_bytes == 295

    # One byte above the exact all-notes size: a naive "always reserve the
    # holdover budget" implementation would still evict the last note here
    # (295 + 44 > 296); the correct implementation must not.
    block, included_ids = render_diff_feedback_block(notes, cap_bytes=full_bytes + 1)

    assert included_ids == [1, 2, 3]
    assert block == full_block
    assert "more notes held" not in block
    assert len(block.encode("utf-8")) <= full_bytes + 1


def test_render_diff_feedback_block_excludes_all_notes_when_cap_too_small():
    notes = [_note(id=1, path="a.py", note="first note")]
    block, included_ids = render_diff_feedback_block(notes, cap_bytes=1)
    assert included_ids == []
    assert block.startswith(
        "## Diff feedback from the user (on your earlier file changes)"
    )
    assert block.endswith("… 1 more notes held for the next message")
    assert "a.py" not in block


# --------------------------------------------------------------------------
# render_diff_feedback_block queue-blocker guard (Qodo #5, PR #1779 fix
# round): the OLDEST pending note must always be deliverable, even one
# whose captured excerpt (a legacy row predating hunk_excerpt's own byte
# cap) is bigger than the whole block cap on its own.
# --------------------------------------------------------------------------


def test_render_diff_feedback_block_oversized_oldest_note_is_truncated_not_dropped():
    """Pre-fix, a note whose rendered entry alone exceeds cap_bytes made
    the loop `break` on the very first iteration with NOTHING included
    (included_ids == []) -- that note (and every note behind it, since it
    is always the oldest/first considered on every subsequent call) was
    never delivered, permanently blocking the whole queue. This is the
    regression pin: the oversized note's id MUST appear in included_ids,
    truncated to fit, rather than being silently excluded forever.
    """
    huge_excerpt = "x" * 20_000  # single "line" -- the minified-file shape
    oversized = _note(
        id=1, path="minified.js", hunk_excerpt=huge_excerpt, note="huge one"
    )
    normal = _note(id=2, path="b.py", note="normal one")

    block, included_ids = render_diff_feedback_block([oversized, normal])

    assert len(block.encode("utf-8")) <= 16384
    # The queue-blocker regression: id 1 must be deliverable.
    assert 1 in included_ids
    assert included_ids[0] == 1
    assert "minified.js" in block
    assert "huge one" in block
    assert "… excerpt truncated to fit" in block
    # The oldest note's own truncated excerpt is a genuine (long) prefix
    # of the original -- not reduced to nothing.
    assert "x" * 100 in block


def test_render_diff_feedback_block_oversized_oldest_note_leaves_later_notes_pending():
    """The note behind the truncated oldest one follows the ordinary
    break-at-cap behavior -- held for the next send, not lost, not
    force-included alongside it."""
    huge_excerpt = "x" * 20_000
    oversized = _note(
        id=1, path="minified.js", hunk_excerpt=huge_excerpt, note="huge one"
    )
    normal = _note(id=2, path="b.py", note="normal one")

    block, included_ids = render_diff_feedback_block([oversized, normal])

    assert included_ids == [1]
    assert "b.py" not in block
    assert block.endswith("… 1 more notes held for the next message")


def test_render_diff_feedback_block_oversized_note_still_excluded_when_metadata_alone_overflows():
    """When the cap is so small even the note's fixed metadata (path,
    header, note text -- never touched by the truncation guard) can't
    fit, the note is excluded exactly like the pre-fix floor case -- the
    guard only ever shrinks the EXCERPT, never other fields."""
    notes = [_note(id=1, path="a.py", hunk_excerpt="x" * 20_000, note="first note")]
    block, included_ids = render_diff_feedback_block(notes, cap_bytes=1)
    assert included_ids == []
    assert block.endswith("… 1 more notes held for the next message")


def test_render_diff_feedback_block_only_note_oversized_and_alone_still_delivered():
    """A single-note queue: no holdover line is needed once the oldest
    (only) note is truncated to fit, since nothing is left pending."""
    huge_excerpt = "y" * 20_000
    note = _note(id=1, path="solo.js", hunk_excerpt=huge_excerpt, note="alone")
    block, included_ids = render_diff_feedback_block([note])
    assert included_ids == [1]
    assert len(block.encode("utf-8")) <= 16384
    assert "more notes held" not in block
    assert "… excerpt truncated to fit" in block


def test_render_diff_feedback_block_embeds_real_hunk_excerpt(diff_fixture):
    """End-to-end: a real segmented hunk's excerpt lands verbatim in the
    fenced block."""
    hunks = split_unified_diff(diff_fixture["single"])
    excerpt = hunk_excerpt(hunks[0])
    note = _note(
        id=1,
        path="single.py",
        hunk_header=hunks[0].header,
        hunk_excerpt=excerpt,
        note="fix this",
    )
    block, included_ids = render_diff_feedback_block([note])
    assert included_ids == [1]
    assert f"````\n{excerpt}\n````" in block


def test_format_diff_feedback_disclosure_exact_format_for_one_note():
    note = _note(
        path="a.py", hunk_header="@@ -1,4 +1,6 @@", note="use the cached value here"
    )
    text = format_diff_feedback_disclosure([note])
    assert (
        text
        == '\U0001f4dd Diff feedback attached — a.py @@ -1,4 +1,6 @@: "use the cached value here"'
    )


def test_format_diff_feedback_disclosure_one_line_per_note():
    notes = [
        _note(path="a.py", hunk_header="@@ -1,2 +1,2 @@", note="first"),
        _note(path="b.py", hunk_header="@@ -3,4 +3,4 @@", note="second"),
    ]
    text = format_diff_feedback_disclosure(notes)
    lines = text.splitlines()
    assert len(lines) == 2
    assert lines[0].endswith('a.py @@ -1,2 +1,2 @@: "first"')
    assert lines[1].endswith('b.py @@ -3,4 +3,4 @@: "second"')


def test_format_diff_feedback_disclosure_empty_notes_returns_empty_string():
    assert format_diff_feedback_disclosure([]) == ""


# --------------------------------------------------------------------------
# Kind-aware rendering (TASK-18060 Task 8, spec §5): `file` and `diff_line`
# anchor kinds get their own block-entry and disclosure-line shapes; `hunk`
# stays byte-unchanged (every test above this section is the byte-parity
# proof and must stay green UNMODIFIED).
# --------------------------------------------------------------------------


def _file_note(**over) -> dict:
    """A `file`-kind note row: `hunk_index=-1`, `hunk_header=''`,
    `hunk_excerpt=''` sentinels (spec §4) so it can never match a real
    hunk."""
    note = _note(hunk_index=-1, hunk_header="", hunk_excerpt="", **over)
    note["anchor_kind"] = "file"
    return note


def _diff_line_note(
    *, diff_line_index: int = 6, diff_line_text: str = "+line6", **over
) -> dict:
    """A `diff_line`-kind note row: hunk fields ALSO populated (spec §4 --
    the hunk the line falls in), plus the line-specific fields."""
    note = _note(**over)
    note.update(
        anchor_kind="diff_line",
        diff_line_index=diff_line_index,
        diff_line_text=diff_line_text,
    )
    return note


def test_render_diff_feedback_block_exact_format_for_file_note():
    note = _file_note(
        id=3,
        run_id="12345678-abcd-4a4a-9a9a-abcdefabcdef",
        path="c.py",
        note="please clean this whole file",
    )
    block, included_ids = render_diff_feedback_block([note])
    expected = (
        "## Diff feedback from the user (on your earlier file changes)\n"
        "### c.py — whole file   [run 12345678]\n"
        "> please clean this whole file"
    )
    assert block == expected
    assert included_ids == [3]
    assert "@@" not in block
    assert "````" not in block


def test_render_diff_feedback_block_exact_format_for_diff_line_note():
    note = _diff_line_note(
        id=4,
        run_id="12345678-abcd-4a4a-9a9a-abcdefabcdef",
        path="b.py",
        hunk_header="@@ -5,3 +5,4 @@",
        hunk_excerpt="+line5\n+line6",
        note="fix line 6",
        diff_line_index=6,
        diff_line_text="+line6",
    )
    block, included_ids = render_diff_feedback_block([note])
    expected = (
        "## Diff feedback from the user (on your earlier file changes)\n"
        "### b.py — @@ -5,3 +5,4 @@   [run 12345678]\n"
        "> on line: +line6\n"
        "> fix line 6\n"
        "````\n"
        "+line5\n+line6\n"
        "````"
    )
    assert block == expected
    assert included_ids == [4]


def test_render_diff_feedback_block_mixed_kinds_all_render_correctly_in_one_block():
    hunk_note = _note(
        id=1,
        path="a.py",
        hunk_header="@@ -1,2 +1,2 @@",
        hunk_excerpt="+x",
        note="hunk note",
    )
    file_note = _file_note(id=2, path="c.py", note="file note")
    line_note = _diff_line_note(
        id=3,
        path="b.py",
        hunk_header="@@ -5,3 +5,4 @@",
        hunk_excerpt="+line5\n+line6",
        note="line note",
    )

    block, included_ids = render_diff_feedback_block(
        [hunk_note, file_note, line_note], cap_bytes=1_000_000
    )

    assert included_ids == [1, 2, 3]
    assert block.count("## Diff feedback from the user") == 1
    assert "### a.py — @@ -1,2 +1,2 @@   [run" in block
    assert "### c.py — whole file   [run" in block
    assert "### b.py — @@ -5,3 +5,4 @@   [run" in block
    assert "> on line: +line6" in block
    assert block.index("a.py") < block.index("c.py") < block.index("b.py")


def test_render_diff_feedback_block_file_note_empty_excerpt_never_gets_truncated_to_fit_tail():
    """A `file` note's excerpt is always `''` (spec §4) -- with a huge cap
    (nothing excluded), it must never grow a phantom '... truncated to
    fit' tail or a fence, since there is no excerpt to truncate."""
    note = _file_note(id=1, path="notes.md", note="please tidy this whole file")
    block, included_ids = render_diff_feedback_block([note], cap_bytes=1_000_000)
    assert included_ids == [1]
    assert "excerpt truncated to fit" not in block
    assert "````" not in block


def test_render_diff_feedback_block_file_note_empty_excerpt_cap_never_produces_negative_truncation():
    """Cap sanity (brief): a `file` note has `hunk_excerpt == ''` by
    construction. When it's the oldest (only) note and doesn't fit under a
    tiny cap, the queue-blocker excerpt-truncation guard must handle an
    EMPTY excerpt sanely -- no exception, no negative-length truncation
    artifact -- and fall back to the ordinary clean exclusion with the
    holdover line, exactly like the pre-existing metadata-alone-overflows
    case for hunk notes."""
    note = _file_note(id=1, path="notes.md", note="please tidy this whole file")
    block, included_ids = render_diff_feedback_block([note], cap_bytes=1)
    assert included_ids == []
    assert block.startswith(
        "## Diff feedback from the user (on your earlier file changes)"
    )
    assert block.endswith("… 1 more notes held for the next message")
    assert "notes.md" not in block


def test_format_diff_feedback_disclosure_exact_format_for_file_note():
    note = _file_note(path="c.py", note="please clean this whole file")
    text = format_diff_feedback_disclosure([note])
    assert (
        text
        == '\U0001f4dd Diff feedback attached — c.py (whole file): "please clean this whole file"'
    )


def test_format_diff_feedback_disclosure_exact_format_for_diff_line_note():
    note = _diff_line_note(
        path="b.py", hunk_header="@@ -5,3 +5,4 @@", note="fix line 6"
    )
    text = format_diff_feedback_disclosure([note])
    assert (
        text
        == '\U0001f4dd Diff feedback attached — b.py @@ -5,3 +5,4 @@ line: "fix line 6"'
    )


def test_format_diff_feedback_disclosure_mixed_kinds_one_line_each():
    hunk_note = _note(
        id=1, path="a.py", hunk_header="@@ -1,2 +1,2 @@", note="hunk note"
    )
    file_note = _file_note(id=2, path="c.py", note="file note")
    line_note = _diff_line_note(
        id=3, path="b.py", hunk_header="@@ -5,3 +5,4 @@", note="line note"
    )

    text = format_diff_feedback_disclosure([hunk_note, file_note, line_note])
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0].endswith('a.py @@ -1,2 +1,2 @@: "hunk note"')
    assert lines[1].endswith('c.py (whole file): "file note"')
    assert lines[2].endswith('b.py @@ -5,3 +5,4 @@ line: "line note"')


# -- middle_elide_path (Task 7, spec §5: row-label path elision) ----------


def test_middle_elide_path_fits_returns_unchanged():
    assert middle_elide_path("a/b/c.py", 100) == "a/b/c.py"


def test_middle_elide_path_exact_budget_returns_unchanged():
    path = "a/b/c.py"
    assert middle_elide_path(path, len(path)) == path


def test_middle_elide_path_loose_elides_the_middle_keeping_ends():
    path = "root/sub1/sub2/sub3/file.py"
    elided = middle_elide_path(path, 15)
    assert elided == "root/…/file.py"
    assert len(elided) <= 15
    assert elided.startswith("root/")
    assert elided.endswith("/file.py")


def test_middle_elide_path_many_components_still_yields_one_ellipsis():
    """Regardless of how many components sit between the first and last,
    only ONE "…" placeholder replaces all of them -- never one per
    dropped component.

    Qodo round (TASK-17611): the naive "<first>/…/<last>" candidate here
    ("a/…/deep_file_name.py", 21 cells) itself overflows a 20-cell budget
    by one cell -- pre-fix, that overflow went unmeasured and unchecked
    entirely. The last component's head is now trimmed by exactly the one
    cell needed to fit.
    """
    path = "a/b/c/d/e/f/g/h/deep_file_name.py"
    elided = middle_elide_path(path, 20)
    assert elided == "a/…/eep_file_name.py"
    assert elided.count("…") == 1
    assert len(elided) < len(path)
    assert cell_len(elided) <= 20


def test_middle_elide_path_degenerate_one_component_returns_unchanged():
    """A bare filename with no "/" at all has no middle to drop -- eliding
    it would mangle the one meaningful fragment left, so it is returned
    unchanged even though it doesn't fit the budget."""
    path = "a_very_long_filename_with_no_directory_component.py"
    assert middle_elide_path(path, 10) == path


def test_middle_elide_path_two_components_returns_unchanged():
    """Both ends already ARE the whole path -- there is no middle
    component to remove, and the elided 3-part form would not even be
    shorter than the original two-part path."""
    path = "dir/file.py"
    assert middle_elide_path(path, 5) == path


def test_middle_elide_path_empty_string_returns_unchanged():
    assert middle_elide_path("", 10) == ""


def test_middle_elide_path_shrinks_endpoints_when_the_elided_form_still_overflows():
    """Qodo round (TASK-17611): when even the 3-part "<first>/…/<last>"
    form overflows the budget, the endpoint components themselves are now
    cell-trimmed (not just the middle) so the final result actually fits
    -- pre-fix, this case silently returned the still-overflowing 3-part
    form unchanged (documented then as "best effort"; that was the exact
    bug this round fixes, not an intentional limit)."""
    path = "a_very_long_first_component/mid/another_very_long_last_component.py"
    elided = middle_elide_path(path, 10)
    assert elided == "a_v/…/t.py"
    assert cell_len(elided) <= 10


def test_middle_elide_path_budgets_by_terminal_cell_width_not_char_count():
    """TASK-17611 (AC#5): a path carrying double-width (CJK) characters must
    be budgeted by actual terminal CELL width, not raw character count.

    This 3-component path is exactly 10 *characters* long -- pre-fix, the
    old ``len(path) <= budget`` check reported it as already fitting a
    10-char budget and returned it unchanged, even though its real
    on-screen width is 15 cells (each CJK character paints 2 cells) --
    silently overflowing the row by 5 cells. Budgeting by
    ``rich.cells.cell_len`` catches this and elides it correctly.

    Qodo round (same task): the naive "<first>/…/<last>" candidate here
    ("根/…/文件.py", 12 cells) STILL overflows a 10-cell budget -- the
    original AC#5 fix measured the ORIGINAL path but never the
    CONSTRUCTED elided candidate, so this exact case kept overflowing by
    2 cells even after that fix. The last component's head is now
    trimmed (cell-aware, dropping "文" and keeping the extension) so the
    final result actually respects the budget.
    """
    path = "根/目录/文件.py"
    assert len(path) == 10, "the character-count budget this bug hides behind"
    elided = middle_elide_path(path, 10)
    assert elided == "根/…/件.py", (
        "a wide-character path within the CHAR budget but over the CELL "
        "budget must still be elided, with the endpoint itself trimmed "
        "when the plain first/…/last candidate still overflows"
    )
    assert cell_len(elided) <= 10, (
        "the CONSTRUCTED candidate must itself fit the cell budget, not "
        "just be shorter than the original path"
    )


def test_middle_elide_path_tiny_budget_degrades_honestly_without_overflowing():
    """Qodo round (TASK-17611): even a budget far too small for any real
    content must never produce a result wider than the budget, as long as
    the budget is at least the ellipsis's own cell width (1) -- there is
    always at least the bare "…" placeholder to fall back to. Only a
    budget narrower than the ellipsis itself (0, or negative) is allowed
    to overflow, since there is nothing left to offer.
    """
    path = "根/目录/文件.py"
    for budget in (1, 2, 3, 4, 5):
        elided = middle_elide_path(path, budget)
        assert cell_len(elided) <= budget, (
            f"budget={budget}: {elided!r} is wider than its own budget"
        )
    # Below the ellipsis's own width, overflow is the documented
    # exception -- but it must still degrade to the smallest possible
    # placeholder, not the full unelided path.
    assert middle_elide_path(path, 0) == "…"
