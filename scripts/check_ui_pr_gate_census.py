#!/usr/bin/env python3
"""Guard: the Tests/UI slice that gates pull requests can only grow.

TASK-32908. Before this, ``Tests/UI`` -- 25,917 collected tests, the largest
single directory in the tree -- gated **nothing** on a pull request:

* ``.github/workflows/test.yml`` does have a 12-shard ``ui-tests`` job, but
  that workflow is ``on: push: branches: ["main"]`` plus
  ``workflow_dispatch``. It does not accept ``pull_request`` events at all,
  so neither its core shards nor its UI shards are a PR gate.
* ``.github/workflows/nightly-deep.yml`` runs ``pytest ./Tests/`` on a
  schedule, which is where the UI suite was assumed to be covered. It is
  not: with no ``--continue-on-collection-errors`` a single bad import
  aborts the whole session, and it has been doing exactly that -- measured
  on run 35706024071 (2026-09-22), ``collected 101851 items / 1 error``
  then ``Interrupted: 1 error during collection``, **zero tests executed**,
  six consecutive nights.

So the only PR gate is ``derived-artifacts.yml``, and the cost of that gap
is not theoretical. ``Tests/UI/test_console_library_tool_setting.py``
asserted ``service._collections is app.local_library_collections_service``
against a ``LocalLibraryToolService`` that has had no ``_collections``
attribute since 5dd1077df6 retired it. The assertion could not pass. It sat
red and blocked nothing -- and the sibling copy of that same factory in
``Chat/console_runtime.py`` went on passing a now-rejected
``collections_service=`` keyword, i.e. a live ``TypeError`` on the
Console-direct Library tool path, for three weeks.

The full directory cannot go in the PR gate: measured at 25,917 tests and
~5.5 CPU-hours, against a fast lane budgeted in minutes. What goes in is a
**verified-green subset**, listed one path per line in
``scripts/ui_pr_gate_census.txt`` and run by the ``ui-fast-lane`` job.

This checker is what stops that subset from quietly evaporating. The tests
themselves are the ratchet on *behaviour* -- a censused file that goes red
turns the job red, which is the whole point. What tests cannot catch is the
census being edited instead of the bug:

* a listed path that no longer exists (renamed or deleted) makes pytest
  collect fewer files while still exiting 0 -- a gate that silently tests
  less than it claims;
* deleting lines is the path of least resistance when a censused file goes
  red, and nothing else would notice.

Hence: every listed path must exist, no duplicates, and the census may
never fall below ``MINIMUM_FILES``. Growing it is free; shrinking it
requires editing this file, which is the review checkpoint.

Exits 0 when the census is intact, 1 otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CENSUS_PATH = REPO_ROOT / "scripts" / "ui_pr_gate_census.txt"

#: Ratchet floor. Raise it when the census grows; never lower it without
#: saying why in the commit message. This is deliberately a literal rather
#: than "len(census) at HEAD" -- a floor derived from the file it guards
#: guards nothing.
# TASK-32908 set this to 120 (120 files / 851 tests, verified green).
# TASK-32899 lowered it to 118: Tests/UI/test_media_handoffs.py and
# Tests/UI/test_media_v88_simple.py were removed because their SUBJECT was
# removed -- both import tldw_chatbook.UI.MediaWindow_v2 (and V88), which
# that task deleted as dead code. This is the one shrink the floor is not
# meant to stop: no behaviour went uncovered, the covered thing is gone.
# TASK-33621.15 raised it to 119: Tests/UI/test_console_session_tab_close.py
# (8 private-profile tests, ~90 s serial) guards the Console tab close that
# stayed broken for three weeks because no PR gate ran its routing tests.
# TASK-33622.10 raised it to 120: Tests/UI/test_quit_prompt_vanish.py gates
# the Ctrl+Q orphaned-prompt hang, which no other PR-gated test would catch.
# TASK-33661 raised it to 121: Tests/UI/test_console_turn_resend_ui.py pins
# the Resend action row, its `r` key, the in-flight guard, and a real-Console
# click and keypress through both Resend paths.
# Roleplay frame B0 raised it to 125: test_adaptive_pane_shell.py,
# test_destination_rail_row.py and test_base_app_screen_tab_region.py gate the
# shared pane shell, rail-row fitting and the behaviour-neutral Tab region.
# The mounted every-route Tab test (test_base_app_screen_tab_region_routes.py,
# ~65 s) stays out of the fast lane; the B0 gate run executes it on both arms.
# TASK-33006 raised it to 130: the five Chat settings files (core-first,
# disclosures, hidden fields, model change, saved defaults; ~5.3 min serial)
# gate the redesigned modal, now inside TASK-34353 sharded lane.
# Retain the existing Console pending-kind/compact approval floor increment
# (+3 over dev), alongside every incoming settings and existing UI census entry.
# TASK-33622.15 raised it to 135: test_close_under_quit_question.py and
# test_modal_quit_in_flight_hooks.py pin Ctrl+Q's still-working answer and a
# dialog that finishes under "Quit while still working?" closing after Wait
# (~35 s serial together). The PR's two real-app quit files
# (test_app_quit_in_flight_modals.py ~2.7 min, test_console_video_picker_cancel.py
# ~1 min locally) stay out until the lane has a shard with room for them.
# TASK-33003.20 adds the production approval batch geometry guard.
# TASK-34000.1 raised it to 139: the census held 138 files on dev (two past
# the 136 floor), and Tests/UI/test_library_quit_guard.py joins it. It is a
# deliberately lean core: 3 real-app boots, about 35-45 s locally; the typed
# tail and a new note reach the DB before exit; a refused save keeps the
# typist in place and Ctrl+Q asks. The other quit variants live in
# test_library_quit_guard_extended.py, outside this lane: it was near its
# 20-minute cap when they were written (TASK-34353 has since sharded it).
# TASK-34000.2 raised it to 140: Tests/UI/test_library_notes_sync_attention.py
# is one real-app boot over a real lasting-sync root wedged the way review
# finding N-02 left it: the tree row, the Notes list and the editor say "needs
# attention", and the real Recovery button heals the folder with no
# "RuntimeError". Measured 24 s wall (14 s call) alone and 56 s wall under
# local load, against the lane's 60 s per-file rule -- keep it a SINGLE test;
# further attention variants go in non-gated files.
# Re-measured at 142 after rebasing onto dev (138 files there + this branch's
# four): TASK-34000.3's test_library_export_replace_confirm.py (two real-app
# boots: a note and a prompt export ask before replacing) and the TASK-32633
# slice's test_library_notes_sync_delete_restore.py (one boot: Delete holds the
# synced folder, Undo returns it) joined the census without a floor bump of
# their own, which left them free to be deleted unnoticed.
# The wave's final review raised it to 143 (C1):
# Tests/UI/test_library_note_autosave_recovers.py is two real-app boots, about
# 20 s locally. A burst whose max wait had run out armed a 0 s timer, which
# Textual never fires, so autosave stayed dead after Keep editing on a refused
# quit and after a rail switch away and back. The unit test that should have
# caught it asserted the 0.0 against a fake ``set_timer``; these read the row.
# TASK-34100.5 raised it to 146 (dev's 143 plus its three files): the
# first-run handoff's mounted guards -- Console's real first mount warns
# nothing (test_console_first_chat_first_mount.py), toasts clear the nav and
# chips at 120x40 (test_console_toast_clears_header.py), and a turn finishing
# in the visible tab raises no hidden notice (test_console_visible_turn_
# attention.py). About 3.5 min serial under load average 30 locally.
# TASK-33620.9 adds the private-profile mounted rename publication regressions.
# TASK-31245 adds the private-profile hydration and handle-ownership regressions.
# TASK-31966 adds the mounted recovery-bar idempotence/geometry regressions.
# TASK-31966 adds the shared Send-reason size idempotence/resize regressions.
# TASK-34100.16 raised it to 151 (dev's floor plus its one file):
# Tests/UI/test_backup_restore_setup_entry.py -- setup's Restore entry opens
# on Inspect, names the format, and explains a settings file, a folder and a
# disabled Create.
# TASK-33620.5 raised it to 155: dev's census held 154 files (three above the
# 151 floor) and Tests/UI/test_console_send_acknowledgement.py joins it. It is
# the lean core -- Enter's "Sending..." frame ordering at 80x24 plus the
# acknowledgement's own rules, 9 tests, about 10 s locally. The mounted
# variants stay in test_console_send_acknowledgement_extended.py, outside the
# lane: the whole set ran 58-110 s serially, over the 60 s per-file rule. An
# earlier cut of that PR kept the file out, saying it pushed shard 3 past the
# 20-minute cap. It did not: shard 3 timed out again with the file removed,
# and none of that shard's 581 tests ran the acknowledgement. The shard was
# at capacity, which the fourth shard (TASK-34353, a920bfe149) fixed.
# TASK-34000.7 (2026-10-08) raised it to 156:
# Tests/UI/test_library_notes_tree_scrolls_wide.py -- the wide Library Notes
# tree is a scroll owner and a Down walk keeps the focused row in view
# (120x36/160x45 plus the 80x24/100x30 compact pins), 6 tests, about
# 26 s locally. The slower variants (200x50/235x52, wheel events, the
# breakpoint round trip, the Trash opener) stay in
# test_library_notes_tree_scrolls_wide_extended.py, outside the lane.
# TASK-34000.13 (2026-10-08) raised it to 157:
# Tests/UI/test_library_note_delete_prompt_visible.py -- Info > Delete shows
# the whole prompt (copy, Cancel, Delete) inside the Info box at 120x36 and
# 80x24, focus lands on a visible Cancel, Tab/Shift+Tab stay inside, and
# Cancel restores Info's scroll; 4 tests, about 14 s locally. The slower
# variants (160x30/160x45/235x52, the resize while open, the real-DB
# "Linked from" case and the DB-level Cancel/Delete checks) stay in
# test_library_note_delete_prompt_visible_extended.py, outside the lane.
# TASK-34000.8 (2026-10-09) raised it to 158:
# Tests/UI/test_library_note_header_fits_pane.py -- the note editor's Save
# and the whole "Use in Console" have a region inside the editor pane at
# 120x36 and 160x45, Discard new note is whole and its hiding never moves
# the mode row, widening 119 -> 235 never hides a header control, the save
# state is never a one-column strip, and F6/Tab/the footer only name a Save
# the user can see; 6 tests, about 22 s locally. The slower variants (140x40,
# 200x50 and 235x52 with the rail open, the long title, 200x24, the
# 235 -> 120 -> 235 round trip, the delete prompt while stacked) stay in
# test_library_note_header_fits_pane_extended.py, outside the lane.
# TASK-34000.25 (2026-10-09) raised it to 159:
# Tests/UI/test_library_rail_switch_keeps_reader_and_note.py -- a rail round
# trip keeps the open Media item (tab, scroll, the row's loaded marker) and a
# New-note note with autosaved text; a dirty note switched inside the
# debounce comes back with its text, the next autosave saves it and the DB
# version equals the snapshot's at every step; `n` in the Reader creates a
# note titled after the document whose first line is the media:// source
# link; 4 tests, about 29 s locally. The slower arms (the Info tab, the real
# external-edit conflict, `n` inside Find, the server-item refusal, the
# untouched-blank GC, the vetoed title, Back at 100x30, 120x36) stay in
# test_library_rail_switch_keeps_reader_and_note_extended.py, outside the lane.
# Roleplay frame B1 raised it to 163: test_roleplay_frame_state.py,
# test_roleplay_stylesheet.py and test_workbench_fitted_text.py gate the pure
# header state, the lazy Roleplay sheet's ownership and FittedText; TASK-34400's
# test_roleplay_hostile_text_surfaces.py (no lane ran it; B1 re-pins one of
# its tests) gates the widget-level hostile-text sinks. B1's mounted Roleplay
# files are bootstrap-profile and run in the PR Fast Lane's
# admission-sensitive step instead (TASK-32873).
# TASK-33007.5 raised it to 136: test_settings_model_defaults_rows.py (~75 s
# serial) gates Settings' Model defaults rows, the one-row Select rows on
# Providers & Models and Console Behavior, and a real save of a blanked field.
# TASK-33007.6 raised it to 137: test_settings_advanced_disclosures.py (~75 s
# serial) gates the Advanced fold -- order, one-row titles that say their
# state, state words on every discovered and catalog row, one frame level,
# and '/' opening the closed disclosure it lands in.
# TASK-33007.7 raised it to 138: test_settings_console_fallback_rows.py (~45 s
# serial) gates Console Behavior's global fallbacks -- Model defaults' rows,
# the On/Off streaming Select, the config-key disclosure and a real save that
# a new chat and an inheriting model default both follow.
# TASK-33007.9 raised it to 139: test_settings_connect_rows.py (23 mounted
# cases, ~6.7 min serial measured at load average 35; 13-25 s a case) gates
# Connect's one-row rows -- the Provider control's name after a choice or
# Revert, painted from its head, its open list's box at both full-screen
# sizes, the key's source words, the Key check row and the Tab budget to Model.
# TASK-33007.9 (review round 2) raised it to 140:
# test_settings_default_model_picker.py (12 mounted cases, ~2.1 min serial)
# gates the Default model picker -- one row, ids grouped by where they came
# from, Custom ID and its rollback, and a wide id read from its head. The lane
# went to five shards in the same commit: this phase's five Settings files
# cost ~10.8 min serial, and replaying per-file seconds from three-shard job
# logs over dev's census plus them put 19-20.5 min of pytest in one of three
# shards and 16.6-17.5 in one of four; five give 13.3-14.5 at most.
# TASK-33007 (final review) raised it to 142: test_settings_save_reach.py
# (6 cases; the real D1 reach save and Ctrl+T in Console) and
# test_settings_providers_models_card_geometry.py (2 cases; the card's
# hit-test net from the region-module move) were left out of the lane.
# The capture review's items 9-11 (p7cf-b) raised it to 143:
# test_settings_console_behavior_grammar.py (5 cases) pins Console
# Behavior's one frame, one control column, its labels and prose inset.
# TASK-33007 (CI on #3057) raised it to 170: test_settings_connect_rows.py
# ran 15+ min of one 20-min shard on CI (61 cases, each mounting the whole
# Settings screen) and timed the shard out. It is split by topic into itself
# (the Provider control), _tab_budget.py, _key_rows.py and _paint.py, listed
# together so round-robin puts each part in a different shard; the lane went
# to six shards in the same commit. The four sit after
# test_settings_default_model_picker.py: replaying per-file seconds from
# #3057's five shard logs, that spot gives at most 12.8 min of pytest in one
# of six shards, where right after test_settings_anthropic_auth_source.py
# stacked three of the slowest Settings files in one shard (15.6 min).
MINIMUM_FILES = 174


def read_census(path: Path) -> list[str]:
    """Read the census, dropping comments and blank lines.

    Args:
        path: The census file, one `Tests/UI/...` path per line.

    Returns:
        The listed paths in file order, comments and blanks removed.
    """
    """Return the census entries, in file order, ignoring blanks/comments."""
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def shard(entries: list[str], index: int, total: int) -> list[str]:
    """Pick one UI Fast Lane shard: every `total`-th entry from `index`.

    Round-robin, not contiguous halves (TASK-34353): the census's slow files
    sit together (the Console cluster), so a contiguous split measured 14.2
    vs 2.3 min where round-robin gives 6.8 vs 9.7. Each shard is still a
    subsequence of the census, so census order holds inside it.

    Args:
        entries: The census, in file order.
        index: This shard, 0-based (`strategy.job-index`).
        total: The shard count (`strategy.job-total`).

    Returns:
        This shard's paths, in census order.

    Raises:
        ValueError: When `index` is not in `range(total)`.
    """
    if not 0 <= index < total:
        raise ValueError(f"shard index {index} is not in range({total})")
    return entries[index::total]


def main(argv: list[str] | None = None) -> int:
    """Verify the PR-gate census is intact, or print one shard of it.

    Args:
        argv: Command-line arguments; ``--shard INDEX TOTAL`` prints that
            shard's paths, one per line, instead of checking the census.

    Returns:
        0 when every listed path exists, is unique, sits under `Tests/UI/`, and
        the census has not shrunk below its floor; 1 otherwise.
    """
    if not CENSUS_PATH.exists():
        print(f"FAIL: census file is missing: {CENSUS_PATH}", file=sys.stderr)
        return 1

    entries = read_census(CENSUS_PATH)
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--shard"]:
        print("\n".join(shard(entries, int(args[1]), int(args[2]))))
        return 0
    problems: list[str] = []

    seen: set[str] = set()
    for entry in entries:
        if entry in seen:
            problems.append(f"duplicate entry: {entry}")
        seen.add(entry)
        if not entry.startswith("Tests/UI/"):
            problems.append(f"not a Tests/UI path: {entry}")
            continue
        if not (REPO_ROOT / entry).is_file():
            problems.append(
                f"listed file does not exist: {entry}\n"
                "    A renamed or deleted censused file makes the gate collect "
                "fewer tests while still exiting 0.\n"
                "    Update the census to the new path, or remove the line and "
                "lower MINIMUM_FILES with a reason."
            )

    if len(entries) < MINIMUM_FILES:
        problems.append(
            f"census has shrunk: {len(entries)} files, floor is {MINIMUM_FILES}.\n"
            "    If a censused file genuinely had to leave the gate, lower\n"
            f"    MINIMUM_FILES in {Path(__file__).name} in the SAME commit and say why.\n"
            "    Deleting the line on its own is how a gate rots to nothing."
        )

    if problems:
        print(
            f"FAIL: {CENSUS_PATH.relative_to(REPO_ROOT)} is not intact "
            f"({len(problems)} problem(s)):",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(entries)} Tests/UI files in the PR gate "
        f"(floor {MINIMUM_FILES}); every listed path exists."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
