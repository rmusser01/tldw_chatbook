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

An entry is a whole file or, since TASK-33621.27, one test's pytest node id
(``Tests/UI/test_x.py::test_name`` or ``...::test_name[param]``). Node ids let
a P0 regression test whose file is too slow for the lane be gated on its own,
instead of the whole file staying ungated. A node id entry must name a test
its file still defines (a renamed test would otherwise make pytest refuse the
whole shard), may not contain whitespace (the lane reads the census one entry
per line), and may not sit beside its own file as a whole entry (pytest
collapses overlapping arguments, ADR-103).

Exits 0 when the census is intact, 1 otherwise.
"""

from __future__ import annotations

import ast
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
# TASK-33621.27 (2026-10-10) raised it by 12, to 175: the Console review's P0
# regression tests (Save .md, Choose folder, the keep-alive's Ctrl+Q, Stop and
# the composer buttons) had run in no PR lane since they merged. Their files
# measured 148-942 s each under load, so they come in as 12 node-id entries --
# the first entries that are not whole files; see the module docstring. A
# node id counts as one entry toward this floor, like a file.
# TASK-34785 (2026-10-10) raised it by three, to 178: the transcript
# detach/lock/composition regressions enter as three node IDs (13 cases),
# preserving every existing file/node and its shard placement.
# TASK-34786 (2026-10-10) adds the actual native-owner retirement regression
# as one node, preserving all earlier shard placements.
# TASK-34797 adds the native Notes status context/shutdown module after green qualification.
# TASK-25907.23 / 34800 adds seven bounded privacy/currentness entries.
# PR2921: production Library Dreams reader and native modal layout/owner checks.
# PR3010 adds two bounded native rule phase/Stop event node IDs.
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
# TASK-33007 (CI on #3057) raised it by three: test_settings_connect_rows.py
# ran 15+ min of one 20-min shard on CI (61 cases, each mounting the whole
# Settings screen) and timed the shard out. It is split by topic into itself
# (the Provider control), _tab_budget.py, _key_rows.py and _paint.py; the
# lane went to six shards in the same commit. Where the four sit in the
# census is decided with the other Phase 7 files, below.
# Placement (measured on #3057's six shard logs, every file timed): the eleven
# Phase 7 Settings files are 35.2 of the lane's 71.4 pytest minutes, and where
# they fall decides the split as much as the shard count does -- listed where
# they were, one of six shards ran 16.0 min (17.4 with setup, of 20). They now
# sit at the end of the census, so no earlier file's shard depends on them,
# ordered by scripts/ui_lane_shard_replay.py for the flattest split. Per-file
# seconds move about 40% between runs on the hosted runners (one file took
# 204 s in one run of #3057 and 295 s in the next), so the replay scores each
# file at the worst of the runs it is given. Over three runs of #3057 and
# dev's 180 entries above the tail, the best tail order replays at 14.2 min
# of pytest at most with six shards, 13.1 with seven and 13.7 with eight
# (dev's own entries fall unevenly over eight), so the lane went to seven. An
# entry added above the tail shifts its shards: re-run the replay then, and
# change the shard count only if the best order still nears the cap -- a new
# count re-deals every entry, and one more shard is not always lighter.
# Phase 7's counts above were taken on its own base; over dev's 179 its eleven
# files make 190, and the node that pins the Study card-list lock (the review
# of #3057) 191.
# The review of #3057 (ruling C11, 2026-10-11) raised it by 18, to 209: one
# new file and 17 node ids that were the only evidence for a ticked Phase 7
# criterion and ran in no PR lane (the census names the criterion beside
# each). They sit above the Phase 7 tail by that ruling, which moves the
# tail's files four shards on (18 mod 7): the replay is re-run after them.
# Integrating that review added one more, 210: the Saved model list's row-words
# test, red in no lane since its rows became Text (five shards on, 19 mod 7).
# Ruling C16 (2026-10-12) raised it by 27, to 237: the Tests/UI files the
# review's fix wave and rounds added, each 3 of 3 serial alone (tail +6 shards).
# C16 for the capture fix wave (2026-10-13): +9 node ids, to 246 (tail +2).
# C16 for capture fix 2 and its rounds (2026-10-13): +12 node ids, to 258 (tail +5).
# C16 for capture fix 3 and its rounds (2026-10-13): +5 node ids, to 263 (tail +5).
# Rebased onto dev (2026-10-13): dev's 190 entries and this PR's 84, 274. The 84
# are one block at the end, re-ordered as a whole: the replay over three runs of
# #3057, with the lanes report's worst local seconds x1.5 for the 53 entries no
# CI log timed, gives 17.6 min of pytest at most as they were listed over seven
# shards, 15.4 re-ordered (about 17 of the job's 20 with setup) and 13.5 over
# eight, so the lane went to eight. Its first CI run there (38458195175) timed
# one shard out: on CI the new entries took up to 1.9x their local seconds.
# Replayed over that run and the three before it (each file at its worst), the
# lane is 125 pytest minutes: 15.6 at most over eight, 13.9 over nine (about 16
# of 20 with setup), 12.5 over ten, so the lane went to nine. The rounds after the
# rebase appended five entries (279); the block of 89 is dealt as a whole.
# Round 4 after the rebase (ruling cf3r1-N1): one node appended after the
# block, a snapshot edit whose Changed has not landed is kept: 275.
# C16 for post-rebase round 1: +2 node ids after the block (leaving and
# Ctrl+Q wait for a snapshot Save still writing): 277.
# Capture 13c: +2 node ids after the block (a kept snapshot edit is
# forgotten once it is saved or reverted): 279.
# Review of #3057, findings 2 and 3: +2 node ids after the block (the
# unsaved-work check reads a kept Settings visit; a pipeline voice edit
# survives a Settings save/restore): 281.
# Their siblings: +1 node id after the block (undoing an audio.cpp review
# merge keeps a pipeline voice edit): 282.
# Rebased onto dev's six-shard lane (TASK-34917, 2026-10-14): no entry added,
# 282. Replayed over #3057's three CI runs and five six-shard runs, each file
# at its slowest, the re-ordered block gives 15.2 pytest min at most over nine
# shards, 13.7 over ten and 12.4 over eleven; ten leaves a hung test's report
# room under the 20-min cap, so the lane runs ten.
# Merge-gate review of #3057's rebased head (2026-10-14): no entry added, 282.
# The census's group notes for those rulings moved here, since the replay's
# --write now moves each entry with the comments above it and would have
# carried a group note with whichever entry sat under it. Ruling C11's 18
# entries and the nineteenth each passed three serial runs alone, at
# pytest's seconds for one run on a loaded machine (load average 40 and up),
# and are node ids where their files are slow or hold reds that are not this
# phase's. Each C16 round's entries are the Tests/UI files or tests it added
# that no lane ran, each 3 of 3 serial alone (seconds in the p7-review-fix
# lanes report). Their notes' "N shards on (N mod 7)" described the
# seven-shard lane of their day; the block has since been re-dealt whole for
# ten shards, so no earlier offset holds. Every block entry keeps its own
# "what it pins" comment, and a blank line ends the block's header.
# TASK-33008 (Phase 8 review, 2026-10-16) raised it by 33, to 315: the
# phase's new Console, Settings-return and Get started card tests ran in no
# PR lane. 33 node ids are appended after the Phase 7 block, so no earlier
# entry changes shard: the ten heaviest first, one to each of the ten shards
# (16-40 s each, local serial at load average ~30), then 23 that take about
# a second or less: about 225 s in all, 13-36 s a shard. Their files are too
# slow whole (the return file runs ~13 min, Use detected's ~4.5 min); the 33
# passed 3 of 3 serial runs together (243 s each).
# Its review round 2 raised it by 2, to 317: two more of the phase's Tests/UI
# tests ran in no lane, appended after the 33 (Use detected keeps the saved
# endpoint, ~5 s; the footer under a dialog, ~10 s), each 3 of 3 serial.
# Its review round 5 raised it by 3, to 320: the round's regression test (a
# card covered at launch runs discovery on the first return) and the two
# discovery-on-return tests it stopped masking ran in no lane. Appended after
# the 35, so they land on shards 8-10 of ten (5-6 s each), each 3 of 3 serial.
# The owner's decisions on its open questions (2026-10-17) raised it by 2, to
# 322: decision D2's keyless-403 hint in Settings' Test rows and in Chat
# settings' Test connection, appended last (1-2 s each), each 3 of 3 serial.
# TASK-35413 raised it by 1, to 323: the project-instruction status row's
# sync-before-compose pin, appended last (no app; well under 1 s).
# TASK-35444 raised it by 1, to 324: the recovery sync's no-wait-on-config-
# rebuild pin, appended last (one mounted Console, about 3 s).
# TASK-34406.28 raised it by two, to 326: ten browser Quit cases and the
# flat-search native-drain regression, appended without moving earlier entries.
# Phase 9 Task 2 (2026-10-17) appends its Tests/UI tests that no lane ran,
# each a node id (their files are slow whole or hold reds that are not this
# task's), each 3 of 3 serial alone. TASK-33005.11's switcher rows: 321.
# TASK-33002.9's unlisted-model lead and toast, and its key-source words: 323.
# TASK-33005.7's Inspector Setup row and send hint, and Settings' identity row: 325.
# TASK-33005.15's three Overview tests, its rewritten no-model pin and the
# NOT RUNNING filter: 330.
# Its review round 1: the identity row follows the Key check row on a poll
# tick (TASK-33005.7): 331; a Test that probes nothing toasts its summary
# literally (TASK-33002.9): 332.
# Phase 9 Task 6 (2026-10-17) appends its Tests/UI node ids that no lane ran,
# each 3 of 3 serial alone: pick mode offers a listing only for the endpoint
# that listed it (TASK-33006.19): 333; a new entry listed under the close
# guard says how to pick its model (TASK-33006.25): 334. Its review round 1:
# pick mode offers a listing only where a pick of it lands (TASK-33006.19): 336.
# Phase 9 Task 7 (2026-10-17) appends three entries no lane ran, each 3 of 3
# serial: the State line count file whole (TASK-33002.10, .11), the new
# fallback search file (TASK-33003.14), and the Test rows after Save
# (TASK-33005.8): 339 (after Task 6's 336, in the Wave 4 integration).
# Its review round 1: a Save that changes no connection field keeps the Key
# check row (TASK-33005.8): 340.
# Phase 9 Task 8 appends its two, each 3 of 3 serial: the mode strip's
# literal provider text (TASK-33002.19, a new file, whole) and the footer
# after Esc closes Chat settings (TASK-33006.20, a node id): 342 (after
# Task 7's 340, in the Wave 4 integration).
# Phase 9 Task 12 appends four, each 3 of 3 serial: the shared-evidence
# readers file whole (TASK-33005.9) and three TASK-33005.10 node ids (the
# late switcher probe, the Model summary reflow, the keyed Settings case): 346.
# Phase 9 Task 14 appends four model-config entries no lane ran, each 3 of 3
# serial in the lane's install set: the field-label file whole
# (TASK-33002.17), the Lab GGUF one-row node (TASK-33641.2), the Skills
# built-in rows file whole (TASK-33003.22) and the legacy [API] key save node
# (TASK-33001.15); it also widens the provider save-scope node to its whole
# file in place (TASK-33002.17, no count change): 350, after Task 12's 346,
# in the Wave 5 integration. Its review round 1 appends the Image Gen
# real-CSS backend-row node (TASK-33641.3), now a private-profile test: 351.
# Phase 9 Task 15 appends the credential round-trip node (TASK-33006.16),
# now a private-profile test, 3 of 3 serial: 352.
# Phase 9 Task 16 appends one node id, 3 of 3 serial: a local server's
# switcher rows are the models its probe listed (TASK-33005.12): 353, after
# Task 15's 352, in the Wave 6 integration. Its integration review round 1
# appends one node id, 3 of 3 serial: pick mode offers a Chat settings
# listing beside the configured server's own listing (TASK-33005.12): 354.
# Round 2 appends one node id, 3 of 3 serial: pick mode offers a Chat
# settings listing beside a loaded catalog (TASK-33005.12): 355.
# Phase 9 Task 18 renames those two nodes in place (TASK-35190: a listing at
# this chat's endpoint replaces the configured server's and the catalog, no
# count change) and appends three node ids, each 3 of 3 serial: a pick keeps
# the chat's session endpoint, a registry entry keeps its own URL, and Switch
# model keeps it and saves none (TASK-35190): 358. Its review round 1
# appends three node ids, each 3 of 3 serial: a pick takes its provider's
# latest endpoint, pick mode reads the session endpoint, and a Settings
# round trip keeps it (TASK-35190): 361. Its review round 2 appends one
# node id, 3 of 3 serial: Ctrl+N reads the configured endpoint, never
# probed on a public host (TASK-35190): 362. Its review round 3 appends one
# node id, 3 of 3 serial: an open's two probers share one in-flight cap
# (TASK-35190): 363. The Phase 9 capture fix wave appends one node id, 3 of
# 3 serial: a dispatched weekly row paints its relative next run
# (TASK-33641.4): 364; and one more, 3 of 3 serial: every Send disabled
# reason fits its strip whole (TASK-33005.7): 365; and one more, 3 of 3
# serial: both generation panels' backend editors are whole and Save is
# reachable (TASK-33641.3): 366. Its review round 1 moves the eleven
# bootstrap_profile node ids Phase 9 appended (TASK-33006.19's three,
# TASK-33006.25's, TASK-33005.12's two, TASK-35190's three, TASK-33641.4's
# and TASK-33641.3's) to the PR Fast Lane's admission-sensitive step, ledger
# R11's rule (TASK-32873: their keep_bootstrap_profile enrollment can poison
# sandboxed suites sharing a process). Each still runs in one PR lane: 355.
# Phase 9's whole-PR review, round 1 moves the four Chat settings geometry
# node ids out the same way (TASK-33006.24's module pytestmark made them
# bootstrap_profile), and four of the phase's private-profile and pure
# Tests/UI evidence nodes take their slots in place (TASK-33003.22's two
# governance ratchets, TASK-33641.4's definition row, TASK-33003.16's theme
# import), so no other entry changes shard. Each still runs in one PR lane: 355.
# Phase 9 was rebased onto dev at 69771c1811 (2026-10-19), whose floor was
# 324: the Phase 9 counts above were taken over its base's 320, so its 35
# entries end at 359 here. The whole-PR review's round 2 marks TASK-33641.4's
# definition-row node bootstrap_profile (its setup imports the app under the
# per-test redirect; it passed in shard 7 only beside files that import the
# app at collection) and moves it to the admission-sensitive step, ledger R11.
# The census's last entry, TASK-33005.7's reason-width node, takes its slot,
# so no other entry changes shard. Each still runs in one PR lane: 358.
# Rebased again onto dev at e3d32ce81a (2026-10-19), whose floor was 326
# (TASK-34406.28's two entries): Phase 9's 34 entries end at 360 here.
# Rebased onto dev 0fef489ece, whose TASK-34982 raised it by 1: the
# poll-timer-stop row-removal pins (a bare transcript host; well under
# 1 s), appended after Phase 9's entries so none of those changes shard
# from its measured runs: 361.
# Rebased onto dev 8e879de639, whose TASK-34751 (2026-10-19) raised it by 2:
# Study's literal-text pins (a markup-shaped workspace name in the scope
# banner; type-to-search over markup-shaped deck and quiz names) and its
# quit-mid-rebuild pin, appended last, after TASK-34982's entry: 363.
# TASK-35010 (S1 T1) adds its saved-lists file: 361.
# TASK-35011 (S1 T2, 2026-10-20) adds its pick-joins-the-list file: 362.
# Its review round 1 (2026-10-20) lists both S1 files as one node per test
# function (12 + 15) instead of two whole files, so no shard takes either
# file whole: 387.
# TASK-35012 (S1 T3, 2026-10-20) appends its nine test functions after
# T2's, one node id each, so nine of the S1 block's ten shard groups gain
# one function each: 396.
# Rebased onto dev 1bbc145ffc, where #3169 merged at 363 (three entries
# over S1's base of 360): S1's 36 nodes end at 399.
# TASK-35676/35677/35678 append six bounded Settings entries after dev
# ccf99a59c4: 405 entries before the parallel-run qualification.
# TASK-35680 appends three bounded parallel-run regressions: 408.
# TASK-35678 rebalances older entries after runs 39447290920/39455910928
# hit the 20-minute cap. Full-run timings and explicit unfinished-work
# reserves guide six further whole-file pairs; S1's 36-node block and the
# nine new entries stay in place at the 399-entry boundary. Coverage,
# shard count, timeout and the inherited floor are preserved.
# TASK-35865..35870 add seven representative mounted regressions and retain
# all 408 prior entries. Measured whole-file slot swaps preserve every node,
# the ten shards and all time limits; TASK-35834 headroom work stays open.
# TASK-35834: lane now runs thirteen. The model-defaults whole-file selector
# becomes all 16 function selectors (+15); see its coverage guard in Tests/CI.
# TASK-35597 raised it further: the five Tools-hub suites were made
# PR-gate-lane safe and re-censused (3 whole files + 112 node ids) on top of
# the 430 above.
# TASK-36036/36037 append four entries without changing dev slots or limits.
# TASK-36097..36100 append thirty-five regression selectors without moving dev slots.
# TASK-34000 clarify batch (2026-10-22, rebased onto dev 7700c63a68) raised it
# by 4, to 588: test_library_rag_answer_names_model.py (.21, the pre-Run line
# names provider and model), test_library_reports_demo_consent.py (.23, the
# paid-demo consent), test_nav_veto_keeps_highlight.py (.27, a vetoed nav
# switch says why) and test_library_media_use_in_console_links.py (.24, link
# on use; 3-4 boots, ~11-13 s quiet). All four are appended at the census
# tail so every existing entry keeps its round-robin shard
# (derived-artifacts.yml: a mid-file insert re-deals every later entry).
# TASK-36130..36132 append fifteen concrete nodes after dev's 588 selectors.
# TASK-33620.15.3 (2026-10-20; rebased onto dev e2cffbed60 2026-10-22)
# appends the idle heap re-freeze's mounted Console file at the tail
# (about 10 s locally): 588 + 1 = 589.
# TASK-33620.15.1.3 (2026-10-20) appends the Environment controller file,
# whose closed-rail test pins the rail guard before the root read. Rebased
# onto dev after the TASK-34000 clarify batch (588): 589.
# TASK-34000 P1-A batch (2026-10-22, rebased onto dev 28d7ab2c9a after a
# second queue conflict) appends 8 files at the census tail so no existing
# entry changes shard: .18's opened-item actions (+ _extended, + Reader
# controls in the list view), .22's dirty-editor Escape (+ _extended exits,
# + Reader-opener refusals), .9's autosave-after-dialog and .10's
# veto-in-place. Dev carried two assignments here (605, then 604; the last
# won) over 605 entries; collapsed to one: 605 + 8 = 613.
# TASK-35835 appends one more, to 614: the keyless-save pin
# (test_settings_keyless_provider_save_keeps_template_env_name_out
# [legacy-env-unset], one private-profile child, ~8 s) proves a quit keeps
# the bytes a Settings save wrote. Appended at the tail, like the others.
# TASK-35648 raised it by 1, to 615: test_sheet_size_ceiling, which ran in no
# PR lane while dev's Library panels sheet sat at 2,004 active lines. A
# static node (about 4-6 s with its private-profile child), appended at the
# tail so no existing entry changes shard.
# TASK-35649 raised it by 1, to 616: the class-level DEFAULT_CSS allowlist
# ratchet, which ran in no PR lane while six unlisted declarations accreted.
# A static node (an AST walk, about 5-10 s with its private-profile child),
# appended at the tail after TASK-35648's.
# TASK-36360..36363 append48 nodes after the complete616-entry dev census.
# 2026-10-23, TASK-34000.20 (P1-C batch, task 1) appends one whole-file entry at
# the tail -- Tests/UI/test_library_ingest_tilde_path_and_enter_waits.py (three
# 160x45 boots, ~15 s locally) -- so no existing entry changes shard: 664 -> 665.
# 2026-10-23, TASK-34000.16 (P1-C batch, task 2) appends one whole-file entry at
# the tail -- Tests/UI/test_library_notes_sync_skips_bad_files_by_name.py (one
# 160x45 boot) -- so no existing entry changes shard: 665 -> 666.
# 2026-10-24, TASK-34000.26 (P1-C batch, task 3) appends one whole-file entry at
# the tail -- Tests/UI/test_library_focus_never_lands_on_a_removed_widget.py
# (three 80x24 boots of the real app, ~15 s locally) -- so no existing entry
# changes shard: 666 -> 667.
# TASK-36512 lists four heavy whole-file entries by function (the MCP
# permissions handoff by parametrize case), so no shard takes one whole:
# every test still runs once, and the floor follows the entry count.
# Current UI lane: 32 shards.
# TASK-36454..36458 append all 58 exact mounted regressions after dev's
# measured split-file tail (TASK-36512), preserving its slots and coverage.
# 2026-10-24, TASK-36479 appends two static node ids at the tail -- the
# focus-edge contrast sweep over the 89 measurable themes and its
# discovery/shortfall-list guard (Tests/UI/test_theme_contrast.py; no
# mount, no config read, ~8 s locally) -- so no existing entry changes
# shard: 762 -> 764.
# TASK-36693 adds all fifteen mounted retirement/publication and Stop cases.
# TASK-36694 adds all thirteen ordinary button/delivery controls.

# Final TASK-36695/36698 review adds four positive/pruning controls.
# TASK-36728/36729 append twenty-two mounted cases; existing shard slots stay fixed.
# TASK-36899 adds eight context lifetime/error controls without moving existing slots.
# TASK-36900 adds eighteen real Reader/DB lifetime and pending-keyboard controls.
# TASK-34000 P1-B batch (2026-10-25, rebased onto dev 7a5e2513c5) appends its
# 6 lane files at the tail so no existing entry changes shard: .17, .14, .12,
# .11 (two files), .71: 893 + 6 = 899.
# TASK-33620.15.8 (2026-10-23) appends 2 at the tail: the composer
# state-restyle file (5 mounted tests, about 14 s locally) and one node, the
# catalog fingerprint's first-read yield (about 4 s); the rest of its file is
# TASK-33620.15.15's to place. Rebased 2026-10-25 onto dev's 899: 899 + 2 = 901.
# TASK-33620.15.7 (2026-10-20) appends its run-tick layout-request file, one
# entry of ~15 s locally, after P1-B's tail: 899 -> 900 (re-based 2026-10-25
# onto dev e554f25812).
# TASK-33620.15.4 (2026-10-23) appends 3 whole-file entries at the tail:
# the projection's off-loop read pins (Library activity, workspace display
# reads, checked display and run-start coalescing), each well under 30 s,
# so no existing entry changes shard: 899 -> 902.
# TASK-33620.15.4's rework (2026-10-23) appends 1 whole-file entry at the
# tail: a send's durable commit keeps the Console's whole syncs quiet
# (about 8 s), so no existing entry changes shard: 902 -> 903.
# TASK-36012 (2026-10-26) appends 1 whole-file entry at the tail: a send's
# start and end leave same-size widgets un-laid-out (20 tests, about 22 s
# locally, under the lane's 30 s per-file budget): dev's 905 + 1 = 906. It
# also drops a dead first `MINIMUM_FILES = 906` assignment dev carried
# above the live 905.
# TASK-35970 (2026-10-26) appends 1 whole-file entry at the tail: a collapsed
# Character section renders nothing while a send refreshes it (three cases,
# about 15 s locally), so no existing entry changes shard: 906 -> 907.
# Character section renders nothing while a send refreshes it (four cases,
# about 18 s locally), so no existing entry changes shard: 906 -> 907.
# TASK-35970 (2026-10-26) appends 1 whole-file entry at the tail: a collapsed
# Character section renders nothing while a send refreshes it (seven cases,
# about 29 s locally), so no existing entry changes shard: 906 -> 907. It also
# drops the dead 906 assignment that stood above dev's effective 905.
# PR #3251 retains all 908 live dev slots plus 56 exact controls.
# TASK-33162.1 gates both ingestion prompt MouseMove cases in one node entry.
# TASK-33620.15.15 (2026-10-27) raised it to 966 (dev's 964 + 2): the
# run-start yield node becomes its whole file in the node's own slot (net
# zero), and the first-send import-cycle and warm-up files join at the tail.
# zero), and the first-send warm-up and import-cycle files join at the tail.
# Replayed over seven green 17-shard runs, each new file charged at its local
# time under load ~38 (CI ran the run-start node in 4.3-6.1 s against 8.8 s
# at that load), shards 13, 14 and 17 come to 15.8, 16.3 and 16.0 pytest min;
# the heaviest shard (11, 17.2) is unchanged.
# TASK-37000 (2026-10-27) appends 3 entries at the tail: the collapsed-rail
# deferral file (four widget cases, about 8 s locally) and the two mounted
# Console cases of its send file as node ids (about 21 s and 30 s locally at
# load ~15), so no existing entry changes shard: 965 -> 968.
# TASK-33620.15.1.6 (2026-10-27) appends 1 whole-file entry at the tail: the
# Environment panel's worker root reads through the real registry (one
# private-profile journey), so no existing entry changes shard: 964 -> 965.
# Environment panel's worker root reads through the real registry (two
# private-profile journeys), so no existing entry changes shard: 964 -> 965.
# TASK-33620.15.1.6 (2026-10-27) appends 2 whole-file entries at the tail: the
# Environment panel's worker root reads through the real registry, and consent
# revoked in the hop before a landing (one private-profile journey each), so
# no existing entry changes shard: 964 -> 966.
# TASK-33620.15.17 (2026-10-27) appends 1 node entry at the tail: keys typed
# before the Console composer's first focus land in the draft, in order (one
# private-profile case), so no existing entry changes shard: dev's 964 + 1
# = 965.
# Textual audit controls retain every dev entry plus 167 tail entries.
# TASK-36611 (2026-10-27) appends 1 whole-file entry at the tail: a tab
# publication holding the session locks finishes while the Console screen's
# pump waits for it (one mounted case, 9-24 s per call locally depending on
# load, most of it the private-profile child's start-up), so no existing
# entry changes shard: 908 -> 909.
# Rebased 2026-10-29 onto dev's 1140 entries: 1140 + 1 = 1141.
# TASK-33620.15.1.1 (2026-10-27) appends 1 node id at the tail: a projecting
# poll tick reads the Agent rail's drilled sub-agent run once (one mounted
# case, about 17 s locally; its resumed-run twin stays in the file), so no
# existing entry changes shard. Rebased 2026-10-27 onto dev's 964: 964 + 1 = 965.
# Rebased 2026-10-29 onto dev's 1140 entries: 1140 + 1 = 1141.
# TASK-33620.15.22 (2026-10-27) appends 1 whole-file entry at the tail: a
# staged attachment gets one Send readiness from either sync site (four
# mounted cases, about 17 s locally), so no existing entry changes shard:
# dev's 964 + 1 = 965.
# TASK-33620.15.20 (2026-10-27) appends 1 whole-file entry at the tail: the
# Conversations tray's run-marker diff composes style no throwaway node (two
# cases, about 8 s locally), so no existing entry changes shard: dev's 1141 + 1
# = 1142.
# Conversations tray's run-marker patch builds no throwaway widget, and its
# A/B parity and drift cases (four tests, three app runs, about 15 s locally),
# so no existing entry changes shard: dev's 1141 + 1 = 1142.
# TASK-33620.15.20 (2026-10-27) appends 2 whole-file entries at the tail: the
# Conversations tray's run-marker patch builds no throwaway widget, with its
# A/B, drift and content-sized-box cases (five tests, four app runs), and its
# structural-sequence case (one test, one app run), each about 18 s locally
# at load 34, so no existing entry changes shard: dev's 1141 + 2 = 1143.
# TASK-33626.2 (2026-10-29) appends 2 node entries at the tail: the Console
# modal literal-colour ratchet and one light-theme rendered-contrast case
# (both under 5 s locally), so no existing entry changes shard: dev's 1141
# + 2 = 1143. Rebased 2026-10-29 onto dev's 1142: 1142 + 2 = 1144.
# Rebased 2026-10-29 onto dev's 1143: 1143 + 2 = 1145.
# Rebased 2026-10-29 onto dev's 1145 entries (dev carried two
# MINIMUM_FILES lines, 1145 and 1144; one kept): 1145 + 2 = 1147.
# TASK-33626.1 (2026-10-29) appends 2 node entries at the tail: the
# Hands-free consent gate and the Esc-during-model-preparation cancel (one
# mounted ChatScreen case each, ~10 s locally), so no existing entry changes
# shard: 1143 + 2 = 1145. Rebased 2026-10-29 onto dev's 1142: 1144 + 2 = 1146.
# Rebased 2026-10-29 onto dev's 1143: 1145 + 2 = 1147.
# Rebased 2026-10-29 onto dev's 1145 entries: 1147 + 2 = 1149.
# TASK-33622.6 (2026-10-29) appends 5 node ids at the tail: the Console rail
# filter's typing, binding-letter, titles-only count, painted-row and
# stale-sync cases, so no existing entry changes shard: dev's 1144 + 5 = 1149.
# Rebased 2026-10-29 onto dev's 1149 entries: 1149 + 5 = 1154.
# TASK-33621.51 (2026-10-29) appends 2 whole-file entries at the tail: the
# Library outcome line on a mounted ChatScreen (four cases, about 24 s
# locally) and its projection helper (16 cases, about 12 s), so no
# existing entry changes shard: dev's 1149 + 2 = 1151.
# TASK-36380 (2026-10-29) appends 1 node entry at the tail: the koboldcpp
# Discover journey (about 23 s locally), so no existing entry changes shard.
# An increment of 1 for that entry. The floor now equals the entry count,
# which also absorbs any lag in dev's floor (tail entries a merge added
# without raising it).
# TASK-36382 appends 9 entries at the tail: four Tests/UI files no lane ran
# (focus contracts, Evals skill eval, the llama.cpp refresh gate, the CSS
# staleness manifest) and five static contracts of test_modal_wide_tier.py
# by node id, whose whole file is over the per-file rule; the anchor check
# among them is one test, ~1 s. The floor now equals the entry count, which
# also absorbs any lag in dev's floor (tail entries a merge added without
# raising it).
# TASK-18925 adds the verified OS-notification Settings journey (+1).
# Console hook-file qualification remains open after native-worker teardown
# errors; no existing census entry is removed or replaced (review receipt).
# PR #3162 (2026-10-31): append the verified archive-card file only.
# The mounted flow remains outside this addition pending qualification.
# PR #3166: two independently verified regression nodes; retain all 1175 entries.
# PR #3173, 2026-10-31: retain all 1177 base entries and append four reviewed nodes.
# PR #3183: preserve current-base prefix and append five reviewed mount/retry/config-capture controls.
# PR #3259 (2026-10-31): retain all 1187 current-base entries and append ten reviewed Notes selectors.
# PR3289 (2026-10-31): retain all1197 current-dev entries and the one mounted rule journey.
# PR #3290 (2026-10-31): retain all1198 current-dev entries and append44 reviewed Textual nodes.

# TASK-34100.5.3 (rebased 2026-10-29 onto dev's 1140) appends 1 node at the
# tail: the Tools & MCP Tool gates pane's fit-to-content pin, which the Full
# agent tools group changed (about 3 s, 3 of 3 serial): 1140 + 1 = 1141. The
# task's other new files are bootstrap_profile and run in the PR Fast Lane's
# admission-sensitive job instead.
# PR3295, 2026-10-31: retain all1242 dev entries plus the Tool gates node.
# PR3299, 2026-10-31: retain all1243 dev entries and append three key-ownership nodes.
# TASK-33620.2 (2026-10-29) appends 2 node entries at the tail: the vision
# block's verbatim refusal through the real composer (AC#4) and the Switch
# model focus hand-back, one mounted Console case each, so no existing entry
# changes shard: dev's 1140 + 2 = 1142.
# Rebased 2026-10-29 onto dev's 1141 entries (TASK-36611 above): 1141 + 2
# = 1143.
MINIMUM_FILES = 1248


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


class _Unresolvable(Exception):
    """A parametrize id that cannot be worked out without importing the test."""


_PRIMITIVE = (str, int, float, bool, type(None))


def _module_values(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level ``NAME = <expr>`` assignments and def/class names."""
    values: dict[str, ast.AST] = {}
    for item in tree.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = item.value
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.value is not None:
                values[item.target.id] = item.value
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            values[item.name] = item
    return values


def _elements(node: ast.AST, scope: dict[str, ast.AST]) -> list[ast.AST]:
    """The argvalues sequence, as AST nodes, in iteration order."""
    if isinstance(node, ast.Name) and node.id in scope:
        return _elements(scope[node.id], scope)
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    if isinstance(node, ast.Dict):  # iterating a dict yields its keys
        if any(key is None for key in node.keys):
            raise _Unresolvable("a ** splat in a dict literal")
        return list(node.keys)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "tuple", "sorted"}
        and len(node.args) == 1
        and not node.keywords
    ):
        items = _elements(node.args[0], scope)
        if node.func.id == "sorted":
            keys = [_literal(item, scope) for item in items]
            if not all(isinstance(key, str) for key in keys):
                raise _Unresolvable("sorted() over non-strings")
            return [ast.Constant(key) for key in sorted(keys)]
        return items
    raise _Unresolvable(f"argvalues built by {ast.unparse(node)!r}")


def _literal(node: ast.AST, scope: dict[str, ast.AST]):
    if isinstance(node, ast.Name) and node.id in scope:
        return _literal(scope[node.id], scope)
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        raise _Unresolvable(f"value {ast.unparse(node)!r}") from None


#: pytest's ``_non_printable_ascii_translate_table`` (``_pytest/compat.py``).
_NON_PRINTABLE = {i: f"\\x{i:02x}" for i in range(128) if i not in range(32, 127)}
_NON_PRINTABLE.update({ord("\t"): "\\t", ord("\r"): "\\r", ord("\n"): "\\n"})


def _ascii_escaped(value: str | bytes) -> str:
    """pytest's ``ascii_escaped``: how a str or bytes value appears in an id."""
    if isinstance(value, bytes):
        text = value.decode("ascii", "backslashreplace")
    else:
        text = value.encode("unicode_escape").decode("ascii")
    return text.translate(_NON_PRINTABLE)


def _scalar_id(value: object) -> str | None:
    """pytest's ``_idval_from_value`` for a value a literal can hold."""
    if isinstance(value, (str, bytes)):
        return _ascii_escaped(value)
    if value is None or isinstance(value, (float, int, bool, complex)):
        return str(value)
    return None


def _value_id(
    node: ast.AST, argname: str, index: int, scope: dict[str, ast.AST]
) -> str:
    """The id pytest gives one parameter value (``_idval`` in pytest 8.4)."""
    if isinstance(node, ast.Name) and isinstance(
        scope.get(node.id), (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return node.id  # pytest uses a function's or class's __name__
    try:
        value = _literal(node, scope)
    except _Unresolvable:
        if isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return f"{argname}{index}"
        raise
    scalar = _scalar_id(value)
    return scalar if scalar is not None else f"{argname}{index}"


def _unique(ids: list[str]) -> list[str]:
    """pytest's ``make_unique_parameterset_ids`` suffixing of repeated ids."""
    counts = {item: ids.count(item) for item in ids}
    suffixes: dict[str, int] = {}
    resolved = list(ids)
    for index, item in enumerate(ids):
        if counts[item] <= 1:
            continue
        sep = "_" if item and item[-1].isdigit() else ""
        number = suffixes.get(item, 0)
        candidate = f"{item}{sep}{number}"
        while candidate in resolved:
            number += 1
            candidate = f"{item}{sep}{number}"
        resolved[index] = candidate
        suffixes[item] = number + 1
    return resolved


def _is_param_call(node: ast.AST) -> bool:
    func = node.func if isinstance(node, ast.Call) else None
    return isinstance(func, ast.Attribute) and func.attr == "param"


def _parametrize_ids(decorator: ast.Call, scope: dict[str, ast.AST]) -> list[str]:
    """The ids one ``@pytest.mark.parametrize`` call generates, in order."""
    args = list(decorator.args)
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    argnames_node = args[0] if args else keywords.get("argnames")
    argvalues_node = args[1] if len(args) > 1 else keywords.get("argvalues")
    if argnames_node is None or argvalues_node is None:
        raise _Unresolvable("parametrize without literal argnames/argvalues")
    argnames = _literal(argnames_node, scope)
    if isinstance(argnames, str):
        argnames = [name.strip() for name in argnames.split(",") if name.strip()]
    argnames = list(argnames)
    explicit = None
    ids_node = args[3] if len(args) > 3 else keywords.get("ids")
    if ids_node is not None:
        if not isinstance(ids_node, (ast.List, ast.Tuple, ast.Name)):
            raise _Unresolvable(f"ids={ast.unparse(ids_node)}")
        explicit = []
        for item in _elements(ids_node, scope):
            value = _literal(item, scope)
            if value is None:
                explicit.append(None)  # pytest falls back to the generated id
                continue
            text = _scalar_id(value)
            if text is None:
                raise _Unresolvable(f"ids entry {ast.unparse(item)!r}")
            explicit.append(text)
    ids: list[str] = []
    for index, item in enumerate(_elements(argvalues_node, scope)):
        given = None
        values = item
        if _is_param_call(item):
            id_keyword = next((k.value for k in item.keywords if k.arg == "id"), None)
            if id_keyword is not None:
                param_id = _literal(id_keyword, scope)
                if param_id is not None:
                    if not isinstance(param_id, str):
                        raise _Unresolvable(f"pytest.param id {param_id!r}")
                    given = _ascii_escaped(param_id)
            values = (
                ast.Tuple(elts=list(item.args)) if len(argnames) > 1 else item.args[0]
            )
        # pytest.param(id=...) wins over ids=[...] (pytest 8.4 _resolve_ids).
        if given is None and explicit is not None and index < len(explicit):
            given = explicit[index]
        if given is not None:
            ids.append(given)
            continue
        if len(argnames) == 1:
            ids.append(_value_id(values, argnames[0], index, scope))
            continue
        if isinstance(values, ast.Name) and values.id in scope:
            values = scope[values.id]
        if not isinstance(values, (ast.Tuple, ast.List)) or len(values.elts) != len(
            argnames
        ):
            raise _Unresolvable(f"multi-argument value {ast.unparse(values)!r}")
        ids.append(
            "-".join(
                _value_id(element, name, index, scope)
                for element, name in zip(values.elts, argnames)
            )
        )
    return _unique(ids)


def _is_parametrize(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parametrize"
    )


def _fixture_call(decorator: ast.AST) -> ast.Call | None:
    """The ``@pytest.fixture(...)`` call of a decorator, or None."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    name = (
        target.attr
        if isinstance(target, ast.Attribute)
        else getattr(target, "id", None)
    )
    if name != "fixture":
        return None
    return decorator if isinstance(decorator, ast.Call) else ast.Call(target, [], [])


def _fixtures(body: list[ast.stmt]) -> dict[str, tuple[bool, bool, list[str]]]:
    """Fixtures defined in `body`: name -> (parametrized, autouse, requests)."""
    found: dict[str, tuple[bool, bool, list[str]]] = {}
    for item in body:
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in item.decorator_list:
            call = _fixture_call(decorator)
            if call is None:
                continue
            keywords = {k.arg: k.value for k in call.keywords}
            name = item.name
            if isinstance(keywords.get("name"), ast.Constant):
                name = str(keywords["name"].value)
            autouse = isinstance(keywords.get("autouse"), ast.Constant) and bool(
                keywords["autouse"].value
            )
            requests = [a.arg for a in item.args.args if a.arg not in {"self", "cls"}]
            found[name] = ("params" in keywords, autouse, requests)
    return found


def _parametrized_fixture(
    path: Path, tree: ast.Module, owners: list[ast.AST]
) -> str | None:
    """A parametrized fixture the test reaches, if the checker can see one.

    Looks at the test module, its classes, and every ``conftest.py`` from the
    file's directory up to the repository root; follows fixture requests
    transitively and counts autouse fixtures. Plugin fixtures are out of
    sight, which is one reason pytest stays the final check.
    """
    available: dict[str, tuple[bool, bool, list[str]]] = {}
    directory = path.resolve().parent
    conftests = []
    while True:
        candidate = directory / "conftest.py"
        if candidate.is_file():
            conftests.append(candidate)
        if directory == REPO_ROOT or directory.parent == directory:
            break
        directory = directory.parent
    for conftest in reversed(conftests):  # nearer conftests override
        try:
            available.update(
                _fixtures(ast.parse(conftest.read_text(encoding="utf-8")).body)
            )
        except (OSError, SyntaxError, ValueError):
            continue
    available.update(_fixtures(tree.body))
    for owner in owners[:-1]:
        available.update(_fixtures(getattr(owner, "body", [])))
    function = owners[-1]
    wanted = [a.arg for a in function.args.args if a.arg not in {"self", "cls"}]
    for decorator in function.decorator_list:
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "usefixtures"
        ):
            wanted += [a.value for a in decorator.args if isinstance(a, ast.Constant)]
    wanted += [name for name, (_, autouse, _) in available.items() if autouse]
    seen: set[str] = set()
    while wanted:
        name = wanted.pop()
        if name in seen or name not in available:
            continue
        seen.add(name)
        parametrized, _, requests = available[name]
        if parametrized:
            return name
        wanted.extend(requests)
    return None


def resolve_node(path: Path, node: str) -> str | None:
    """Check that `path` still defines the test a node id names.

    Static on purpose: the census check runs install-free, so it cannot import
    the test module. A bracketed id (``test_x[enter-80x24]``) is worked out
    from the function's literal ``@pytest.mark.parametrize`` decorators the
    way pytest 8.4 builds ids -- closest decorator first; ``pytest.param(id=)``
    before ``ids=[...]``; str/bytes ascii-escaped; repeated ids suffixed --
    because a renamed id makes pytest exit 4 and the whole shard runs nothing
    (TASK-33621.27 review). This is a fast pre-check, not proof: pytest's own
    collection stays the final word (a ``pytest_make_parametrize_id`` hook, for
    one, can change an id). What it cannot see it refuses rather than accepts:
    ids built by code (a computed list, ``ids=lambda ...``), class- or
    module-level parametrize, and parametrized fixtures, which add their own
    id components. Gate the whole test instead, or give it literal ids.

    Args:
        path: The test file.
        node: The node id after the file, ``test_x``, ``test_x[id]`` or
            ``TestClass::test_x[id]``.

    Returns:
        None when the node resolves; otherwise why it does not.
    """
    name_part, bracket, param = node.partition("[")
    names = name_part.split("::")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as error:
        return f"cannot read {path.name}: {type(error).__name__}"
    body: list[ast.stmt] = tree.body
    owners: list[ast.AST] = []
    match: ast.AST | None = None
    for depth, name in enumerate(names):
        last = depth == len(names) - 1
        kinds = (ast.FunctionDef, ast.AsyncFunctionDef) if last else (ast.ClassDef,)
        match = next(
            (item for item in body if isinstance(item, kinds) and item.name == name),
            None,
        )
        if match is None:
            return f"{path.name} does not define {name_part}"
        owners.append(match)
        body = getattr(match, "body", [])
    if not bracket:
        return None
    if not param.endswith("]"):
        return f"malformed node id {node!r}"
    wanted = param[:-1]
    module_marks = any(
        isinstance(item, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in item.targets)
        and any(_is_parametrize(sub) for sub in ast.walk(item.value))
        for item in tree.body
    )
    class_marks = any(
        _is_parametrize(decorator)
        for owner in owners[:-1]
        for decorator in getattr(owner, "decorator_list", [])
    )
    if module_marks or class_marks:
        return (
            f"parametrize id [{wanted}] cannot be resolved statically: module- or "
            "class-level parametrize. Gate the whole test instead."
        )
    fixture = _parametrized_fixture(path, tree, owners)
    if fixture is not None:
        return (
            f"parametrize id [{wanted}] cannot be resolved statically: the test "
            f"uses the parametrized fixture {fixture!r}, whose ids pytest adds "
            "to the node id. Gate the whole test instead."
        )
    scope = _module_values(tree)
    decorators = [d for d in reversed(match.decorator_list) if _is_parametrize(d)]
    if not decorators:
        return f"{name_part} is not parametrized, so it has no id [{wanted}]"
    try:
        combined = [""]
        for decorator in decorators:  # closest to the function first
            ids = _parametrize_ids(decorator, scope)
            combined = [
                f"{left}-{right}" if left else right
                for left in combined
                for right in ids
            ]
    except _Unresolvable as reason:
        return (
            f"parametrize id [{wanted}] cannot be resolved statically ({reason}). "
            "Gate the whole test, or give its parametrize literal ids."
        )
    if wanted not in combined:
        shown = ", ".join(combined[:8]) + (" ..." if len(combined) > 8 else "")
        return f"{name_part} has no parametrize id [{wanted}] (it has: {shown})"
    return None


def overlapping_targets(targets: list[str] | tuple[str, ...]) -> list[str]:
    """Pytest arguments that overlap another in the same invocation.

    Pytest collapses overlapping arguments (ADR-103): a file listed whole
    beside one of its node ids ran 1 of its 21 tests with everything green
    (TASK-33621.27 review). Shared by the census check and the Tests/CI pins
    on every PR-lane pytest step.

    Args:
        targets: One pytest invocation's targets (directories, files or node
            ids), in order.

    Returns:
        One problem line per overlap; empty when the targets are disjoint.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for target in targets:
        if target in seen:
            problems.append(f"duplicate entry: {target}")
        seen.add(target)
    unique = list(dict.fromkeys(targets))
    for outer in unique:
        outer_path = outer.rstrip("/")
        directory = "::" not in outer and not outer_path.endswith(".py")
        for inner in unique:
            if inner == outer:
                continue
            inner_file = inner.split("::", 1)[0].rstrip("/")
            if inner.startswith(outer_path + "::") or inner.startswith(
                outer_path + "["
            ):
                kind = "whole-file entry" if "::" not in outer else "listed target"
                problems.append(
                    f"node id overlaps a {kind}: {inner}\n"
                    f"    {outer} is also listed and contains it; pytest collapses "
                    "overlapping arguments (ADR-103). Drop one of the two."
                )
            elif directory and inner_file.startswith(outer_path + "/"):
                problems.append(
                    f"{inner} sits under the listed directory {outer_path}; "
                    "pytest collapses overlapping arguments (ADR-103)."
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Verify the PR-gate census is intact, or print one shard of it.

    Args:
        argv: Command-line arguments; ``--shard INDEX TOTAL`` prints that
            shard's paths, one per line, instead of checking the census.

    Returns:
        0 when every entry is unique, sits under `Tests/UI/`, names a file
        that exists (and, for a node id, a test that file defines), and the
        census has not shrunk below its floor; 1 otherwise.
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

    # Duplicates, a node id beside its own whole file, and anything under a
    # listed directory: the same helper pins every PR-lane pytest step.
    problems.extend(overlapping_targets(entries))
    for entry in entries:
        if not entry.startswith("Tests/UI/"):
            problems.append(f"not a Tests/UI path: {entry}")
            continue
        if any(character.isspace() for character in entry):
            problems.append(
                f"entry contains whitespace: {entry!r}\n"
                "    The lane reads one entry per line; gate a node id without "
                "spaces (pick another parametrization, or the whole test)."
            )
            continue
        file_part, _, node = entry.partition("::")
        if not (REPO_ROOT / file_part).is_file():
            problems.append(
                f"listed file does not exist: {entry}\n"
                "    A renamed or deleted censused file makes the gate collect "
                "fewer tests while still exiting 0.\n"
                "    Update the census to the new path, or remove the line and "
                "lower MINIMUM_FILES with a reason."
            )
            continue
        if not node:
            continue
        reason = resolve_node(REPO_ROOT / file_part, node)
        if reason is not None:
            problems.append(
                f"listed test does not resolve: {entry}\n"
                f"    {reason}\n"
                "    A renamed or deleted gated test makes pytest refuse the "
                "whole shard with 'not found' (exit 4, nothing runs).\n"
                "    Update the census to the test's current id, or remove the "
                "line and lower MINIMUM_FILES with a reason."
            )

    if len(entries) < MINIMUM_FILES:
        problems.append(
            f"census has shrunk: {len(entries)} entries, floor is {MINIMUM_FILES}.\n"
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
        f"OK: {len(entries)} Tests/UI entries in the PR gate "
        f"(floor {MINIMUM_FILES}); every listed path exists."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
