# Library Adaptive Reader Programme Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five-destination Library adaptive-reader programme with one hermetic, production-shaped automated and live evidence run tied to an exact subject revision.

**Architecture:** Reuse the existing shared-shell and destination reader suites as the automated matrix. Keep the one-time parent runner, stdlib-only pre-import child, and live scenario declarations task-local inside the retained TASK-23019 evidence directory; add one focused production-CSS UI module only for durable cross-reader resize and route-cycle contracts. Product code changes are not planned; a regression enters this PR only through the bounded repair gate in Task 6.

**Tech Stack:** Python 3.12 from the repository virtualenv, pytest/pytest-asyncio, Textual Pilot with `TldwCli.CSS_PATH`, stdlib `subprocess`, `sys.addaudithook`, `tempfile`, `xml.etree.ElementTree`, `json`, `hashlib`, and `shutil`.

---

## Planning constraints

- Specification: `Docs/superpowers/specs/2026-08-27-library-adaptive-reader-programme-closeout-design.md`
- Task: `backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md`
- Architecture: `backlog/decisions/086-library-adaptive-reader-shell.md`
- Capability ledger: `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md`
- Evidence destination: `Docs/superpowers/reviews/evidence/task-23019/`
- Approved terminal sizes: `160x50`, `120x35`, `100x30`, `80x24`
- No full repository test sweep. Run only the production-shaped cross-reader suites, closeout runner tests, live matrix, and named derived/static checks below.
- No new dependency, schema, service authority, persistence owner, reader abstraction, or Watchlists/global-shell work.

**ADR required:** no

**ADR path:** N/A; verify existing `backlog/decisions/086-library-adaptive-reader-shell.md`.

**Reason:** TASK-23019 verifies and closes the accepted five-destination shell boundary. A finding that changes storage, authority, security, or long-lived application structure must become a separate task with a fresh ADR assessment.

## File structure

| Path | Responsibility |
| --- | --- |
| `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py` | Task-local parent CLI, finite catalogue, test selection, subject-revision checks, scratch environment, result validation, normalization, hashing, cleanup, and bounded evidence promotion. |
| `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py` | Task-local stdlib-only pre-import execution boundary; installs filesystem/network audit tripwires, runs pytest/live scenarios, and writes exact results plus attempted-access facts under scratch. |
| `Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py` | Task-local 20-cell production-CSS matrix and five representative capability journeys. These executable scenario declarations are retained with the evidence and never become a CI API. |
| `Tests/Live/test_library_adaptive_reader_closeout.py` | TDD coverage for catalogue completeness, clean-subject admission, environment/path ownership, tripwire behavior, result validation, normalization, cleanup, collision safety, and promotion rollback. |
| `Tests/UI/test_library_adaptive_reader_closeout.py` | Durable production-CSS five-destination resize-purity, fresh-screen preference reload, and sequential route-cycle regressions. It imports only established `Tests/UI` helpers, never the task-local runner/scenario files, and writes no evidence. |
| `Docs/superpowers/reviews/evidence/task-23019/` | Task-local executable sources plus generated bounded evidence: README, manifest, summary, structured facts, representative text/SVG captures, and hashes. |
| `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md` | Final programme-closeout ledger entry linked to the subject revision and TASK-23019 evidence. |
| `backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md` | Plan, checked acceptance criteria, implementation notes, exact verification, and final status. |

No production file is scheduled for modification. If Task 6 admits a localized repair, use the existing contract owner named there; do not add another shared layer.

## Bounded contract mapping

The parent runner owns this exact catalogue. Every row must name at least one automated pytest node and one live fact/capture before the runner can return success.

| ID | Existing automated source(s) | Fresh live source |
| --- | --- | --- |
| SH-01 | `test_library_adaptive_reader_shell.py`; mount/identity cases in each destination reader module | every common matrix cell plus `single_app_route_cycle` |
| SH-02 | `test_library_adaptive_reader_state.py`; shared-shell and destination geometry cases | all 20 `common_matrix` cells |
| SH-03 | shared preference cases in `test_library_media_reader_shell.py`; new fresh-screen all-destination reload case | `single_app_route_cycle` preference facts |
| SH-04 | shared-shell focus cases; Media footer; Conversations/Skills F6 cases | common matrix focus/footer facts plus `single_app_route_cycle` |
| SH-05 | Media, Conversations, Prompts, and Skills stale-settlement reader cases; Notes session fences | five capability journeys |
| SH-06 | pure layout-state cases and Media resize-purity case | `resize_purity` for all five destinations |
| SH-07 | existing route/unmount fencing plus the new closeout route test | `single_app_route_cycle` |
| ME-01 | `test_library_media_reader_flow.py` | `media_capability` |
| ME-02 | `test_library_multiselect_media.py` | `media_capability` |
| CO-01 | progressive Find and Read/Info cases in `test_library_conversation_reader.py` | `conversations_capability` |
| CO-02 | stale/retry/deletion/handoff cases in the same module | `conversations_capability` |
| NO-01 | `test_library_notes_reader.py` | `notes_capability` |
| NO-02 | Notes conflict/session and `test_library_multiselect_notes.py` | `notes_capability` |
| PR-01 | Basic/Advanced draft and validation cases in `test_library_prompts_reader.py` | `prompts_capability` |
| PR-02 | Prompt history/import/retry and `test_library_prompts_canvas.py` focused nodes | `prompts_capability` |
| SK-01 | mode/draft/trust identity cases in `test_library_skills_reader.py` | `skills_capability` |
| SK-02 | Files/stale grant/delete plus focused Skills service cases | `skills_capability` |

Earlier TASK-22031/TASK-22033 evidence is lineage only. The runner must reject a catalogue row whose current subject revision has no fresh passing automated node or live result.

### Task 1: Record the approved plan and task state

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-27-library-adaptive-reader-programme-closeout-design.md`
- Create: `Docs/superpowers/plans/2026-08-27-library-adaptive-reader-programme-closeout.md`
- Modify: `backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md`

- [ ] **Step 1: Set the design status to approved**

Change only the status header:

```markdown
**Status:** Approved for implementation planning
```

- [ ] **Step 2: Attach this implementation plan to TASK-23019**

Run:

```bash
backlog task edit 23019 --plan $'1. Add the bounded manifest and hermetic runner contracts\n2. Add pre-import filesystem/network tripwires\n3. Add the missing production-shaped live matrix and sequential route cycle\n4. Run the curated automated matrix and classify any failures\n5. Freeze and verify the exact subject revision\n6. Promote normalized evidence and close the programme\n\nADR required: no\nADR path: N/A\nReason: verifies ADR-086 without changing its storage, service-authority, security, or application-structure boundary.'
```

Expected: TASK-23019 remains `In Progress` and gains `## Implementation Plan`; no acceptance criterion is checked.

- [ ] **Step 3: Verify documentation hygiene**

Run:

```bash
../../.venv/bin/python scripts/check_backlog_task_ids.py
git diff --check
```

Expected: both exit 0.

- [ ] **Step 4: Commit the approved plan**

```bash
git add Docs/superpowers/specs/2026-08-27-library-adaptive-reader-programme-closeout-design.md Docs/superpowers/plans/2026-08-27-library-adaptive-reader-programme-closeout.md 'backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md'
git commit -m 'docs(library): plan adaptive reader closeout'
```

### Task 2: Add the finite catalogue and task-local parent-runner contracts

**Files:**
- Create: `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py`
- Create: `Tests/Live/test_library_adaptive_reader_closeout.py`

- [ ] **Step 1: Write failing catalogue and subject-admission tests**

Add tests that load the runner by file path and assert:

```python
EXPECTED_IDS = {
    "SH-01", "SH-02", "SH-03", "SH-04", "SH-05", "SH-06", "SH-07",
    "ME-01", "ME-02", "CO-01", "CO-02", "NO-01", "NO-02",
    "PR-01", "PR-02", "SK-01", "SK-02",
}

def test_catalogue_is_finite_unique_and_has_both_evidence_kinds():
    module = _load_runner()
    assert set(module.CATALOGUE) == EXPECTED_IDS
    assert all(entry.automated_nodes for entry in module.CATALOGUE.values())
    assert all(entry.live_cases for entry in module.CATALOGUE.values())

def test_subject_revision_requires_exact_clean_head(tmp_path, monkeypatch):
    module = _load_runner()
    monkeypatch.setattr(module, "_git", _fake_git(head="abc", dirty=True))
    with pytest.raises(module.CloseoutError, match="subject_worktree_not_clean"):
        module.admit_subject(tmp_path, "abc")
```

Also cover wrong `HEAD`, missing catalogue IDs, duplicate live keys, unknown pytest selectors against
an injected synthetic collection, and a final-head source change relative to the recorded subject
tree. Do not collect the not-yet-created closeout UI module in Task 2; real collection validation
begins after Task 4 creates it.

Add parser REDs for every declared option and incompatible combination listed in Step 3. Assert
stable `CloseoutError` categories for semantic misuse and ordinary `argparse` usage errors only for
unknown flags or malformed values.

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -q
```

Expected: FAIL because the runner module and catalogue do not exist.

- [ ] **Step 3: Implement the smallest parent-runner model**

Use frozen dataclasses and constants, not a plugin framework:

```python
@dataclass(frozen=True)
class Contract:
    automated_nodes: tuple[str, ...]
    live_cases: tuple[str, ...]

CATALOGUE: dict[str, Contract] = {
    "SH-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_adaptive_reader_shell.py::test_sync_layout_retains_every_mounted_child_identity",
            "Tests/UI/test_library_media_reader_shell.py::test_media_shell_mounts_library_items_reader_and_two_five_column_grips",
            "Tests/UI/test_library_conversation_reader.py::test_conversations_mount_three_retained_roles_once",
            "Tests/UI/test_library_notes_reader.py::test_database_notes_mount_three_retained_roles_once",
            "Tests/UI/test_library_prompts_reader.py::test_prompts_mount_three_retained_roles_once",
            "Tests/UI/test_library_skills_reader.py::test_skills_mount_three_retained_roles_and_default_to_overview",
        ),
        live_cases=("common_matrix", "single_app_route_cycle"),
    ),
    "SH-02": Contract(
        automated_nodes=(
            "Tests/Library/test_library_adaptive_reader_state.py::test_shared_resolution_uses_adaptive_width_classes",
            "Tests/UI/test_library_adaptive_reader_shell.py::test_all_five_regions_remain_inside_representative_media_widths",
        ),
        live_cases=("common_matrix",),
    ),
    "SH-03": Contract(
        automated_nodes=(
            "Tests/UI/test_library_media_reader_shell.py::test_shared_library_pane_choice_round_trips_between_media_and_conversations",
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_preferences_restore_in_fresh_screen",
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle",
        ),
        live_cases=("single_app_route_cycle",),
    ),
    "SH-04": Contract(
        automated_nodes=(
            "Tests/UI/test_library_adaptive_reader_shell.py::test_hiding_focused_pane_moves_focus_to_truthful_restore_grip",
            "Tests/UI/test_library_media_reader_flow.py::test_footer_advertises_only_working_current_actions",
            "Tests/UI/test_library_conversation_reader.py::test_conversations_global_f6_cycles_visible_destination_roles",
            "Tests/UI/test_library_skills_reader.py::test_skills_reader_f6_reaches_items_and_work_regions",
        ),
        live_cases=("common_matrix", "single_app_route_cycle"),
    ),
    "SH-05": Contract(
        automated_nodes=(
            "Tests/UI/test_library_media_reader_flow.py::test_late_completion_for_a_cannot_replace_loaded_b_or_show_error",
            "Tests/UI/test_library_conversation_reader.py::test_late_previous_selection_cannot_overwrite_current_reader",
            "Tests/Library/test_library_notes_session.py::test_stale_open_session_cannot_replace_a_newer_loaded_session",
            "Tests/UI/test_library_prompts_reader.py::test_same_prompt_older_detail_load_cannot_overwrite_newer_generation",
            "Tests/UI/test_library_skills_reader.py::test_same_skill_older_detail_result_cannot_replace_newer_generation",
        ),
        live_cases=(
            "media_capability", "conversations_capability", "notes_capability",
            "prompts_capability", "skills_capability",
        ),
    ),
    "SH-06": Contract(
        automated_nodes=(
            "Tests/Library/test_library_adaptive_reader_state.py::test_resolution_never_mutates_saved_preferences",
            "Tests/UI/test_library_media_reader_shell.py::test_media_shell_resize_uses_resolver_without_reads_or_recompose",
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_resize_is_presentation_only",
        ),
        live_cases=("resize_purity",),
    ),
    "SH-07": Contract(
        automated_nodes=(
            "Tests/UI/test_library_adaptive_reader_closeout.py::test_closeout_single_app_route_cycle",
        ),
        live_cases=("single_app_route_cycle",),
    ),
    "ME-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_media_reader_flow.py::test_reader_defaults_to_read_and_keeps_mode_across_local_items",
            "Tests/UI/test_library_media_reader_flow.py::test_progress_restores_after_loaded_content_mounts",
        ),
        live_cases=("media_capability",),
    ),
    "ME-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_multiselect_media.py::test_confirming_bulk_delete_swaps_toolbar_for_confirm_row",
            "Tests/UI/test_library_multiselect_media.py::test_delete_selection_soft_deletes_via_real_db_and_updates_records_and_counts",
        ),
        live_cases=("media_capability",),
    ),
    "CO-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_conversation_reader.py::test_progressive_reader_paints_first_page_then_completes_find_off_loop",
            "Tests/UI/test_library_conversation_reader.py::test_reader_info_is_explicit_and_truthful",
        ),
        live_cases=("conversations_capability",),
    ),
    "CO-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_conversation_reader.py::test_open_console_requires_final_complete_error_free_match",
            "Tests/UI/test_library_conversation_reader.py::test_authoritative_refresh_marks_selected_conversation_deleted_without_fallback",
        ),
        live_cases=("conversations_capability",),
    ),
    "NO-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_notes_reader.py::test_list_and_work_identity_survive_open_preview_info_and_edit",
        ),
        live_cases=("notes_capability",),
    ),
    "NO-02": Contract(
        automated_nodes=(
            "Tests/Library/test_library_notes_session.py::test_conflict_stops_chaining_and_preserves_the_newest_draft",
            "Tests/UI/test_library_multiselect_notes.py::test_permanent_navigator_tasks_respect_dirty_draft_veto",
        ),
        live_cases=("notes_capability",),
    ),
    "PR-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_prompts_reader.py::test_basic_save_preserves_advanced_only_prompt_fields",
            "Tests/UI/test_library_prompts_reader.py::test_invalid_advanced_block_routes_save_focus_to_its_owner",
        ),
        live_cases=("prompts_capability",),
    ),
    "PR-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_prompts_reader.py::test_import_replaces_only_work_content_and_keeps_list_mounted",
            "Tests/UI/test_library_prompts_reader.py::test_detail_failure_keeps_prior_prompt_locked_and_retry_loads_selection",
        ),
        live_cases=("prompts_capability",),
    ),
    "SK-01": Contract(
        automated_nodes=(
            "Tests/UI/test_library_skills_reader.py::test_skill_modes_preserve_list_work_and_one_live_draft",
            "Tests/UI/test_library_skills_reader.py::test_skills_trust_mode_identifies_exact_review_snapshot",
        ),
        live_cases=("skills_capability",),
    ),
    "SK-02": Contract(
        automated_nodes=(
            "Tests/UI/test_library_skills_reader.py::test_skills_files_mode_is_read_only_and_labels_binary_files",
            "Tests/UI/test_library_skills_reader.py::test_same_skill_older_trust_review_cannot_patch_newer_generation",
            "Tests/UI/test_library_skills_reader.py::test_same_skill_older_delete_cannot_reset_a_newer_work_generation",
        ),
        live_cases=("skills_capability",),
    ),
}
SIZES = ((160, 50), (120, 35), (100, 30), (80, 24))
DESTINATIONS = ("media", "conversations", "notes", "prompts", "skills")

def admit_subject(repo: Path, requested: str) -> Subject:
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    if head != requested:
        raise CloseoutError("subject_revision_mismatch")
    if _git(repo, "status", "--porcelain"):
        raise CloseoutError("subject_worktree_not_clean")
    return Subject(commit=head, tree=tree)
```

Define and unit-test the parent CLI in this task, before any RED command relies on it:

```text
--subject-revision REV   exact clean commit admitted for a promotable run
--development-run        run against the current checkout without subject admission or promotion
--live-case NAME         run one declared live scenario
--live-only              run every declared live scenario and skip the curated pytest matrix
--no-promote             explicitly retain no repository evidence
--promote                validate and promote the complete evidence bundle
--verify-evidence PATH   verify one already-promoted bundle without executing tests
```

Parser tests must prove `--development-run` rejects `--subject-revision`, `--promote`, and
`--verify-evidence`; `--live-case` rejects `--live-only`; promotion requires
`--subject-revision`; and development runs are unconditionally non-promoting even when
`--no-promote` is omitted. Unknown live keys must fail with the stable category
`scenario_not_defined`, not an `argparse` traceback.

Keep the curated pytest file list as one tuple. It must include:

```text
Tests/Library/test_library_adaptive_reader_state.py
Tests/Library/test_library_media_reader_state.py
Tests/Library/test_library_conversation_reader_state.py
Tests/Library/test_library_notes_session.py
Tests/Library/test_library_prompts_seam.py
Tests/Library/test_library_skills_reader_state.py
Tests/UI/test_library_adaptive_reader_shell.py
Tests/UI/test_library_media_reader_shell.py
Tests/UI/test_library_media_reader_flow.py
Tests/UI/test_library_conversation_reader.py
Tests/UI/test_library_notes_reader.py
Tests/UI/test_library_prompts_reader.py
Tests/UI/test_library_skills_reader.py
Tests/UI/test_library_multiselect_media.py
Tests/UI/test_library_multiselect_conversations.py
Tests/UI/test_library_multiselect_notes.py
Tests/UI/test_library_adaptive_reader_closeout.py
Tests/Chat/test_chat_conversation_service.py
Tests/Prompt_Management/test_prompt_preservation.py
Tests/Skills/test_skills_library_flow.py
Tests/Skills/test_skill_trust_service.py
Tests/Skills/test_local_skills_bundle_io.py
Tests/Skills/test_read_skill_file.py
Tests/Skills/test_skill_file_trust_material.py
```

Treat each catalogue node as a pytest selector: it is satisfied by an exact passing node ID or by
one or more passing parameterized node IDs beginning with `selector + "["`. During collection,
reject a selector that matches nothing; during settlement, reject one whose matching cases are not
all `PASS`.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -q
../../.venv/bin/python -m ruff check Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Tests/Live/test_library_adaptive_reader_closeout.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Tests/Live/test_library_adaptive_reader_closeout.py
git commit -m 'test(library): define adaptive reader closeout manifest'
```

### Task 3: Add the pre-import hermetic child boundary

**Files:**
- Create: `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py`
- Modify: `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py`
- Modify: `Tests/Live/test_library_adaptive_reader_closeout.py`

- [ ] **Step 1: Write failing boundary tests**

Cover these observable outcomes with child subprocess tests named
`test_child_immediately_exits_for_each_prohibited_network_api`,
`test_child_immediately_exits_for_each_prohibited_process_api`,
`test_child_records_and_blocks_read_from_real_profile`,
`test_child_records_and_blocks_write_to_checkout`,
`test_child_allows_checkout_and_runtime_reads_but_only_scratch_writes`,
`test_child_environment_redirects_every_writable_owner_before_import`, and
`test_child_result_names_every_collected_and_settled_node`.

Use a tiny synthetic pytest module written inside the test's `tmp_path`; do not start the 629-case
curated matrix while testing the child boundary itself.

Parameterize the network RED across `socket.connect`, `socket.sendto`, `socket.sendmsg`,
`socket.bind`, `socket.listen`,
`socket.getaddrinfo`, `socket.gethostbyname`, `socket.gethostbyname_ex`,
`socket.gethostbyaddr`, and `socket.getnameinfo`. Parameterize the process RED across
`subprocess.Popen`, `os.system`, `os.posix_spawn`, `os.posix_spawnp`, and every available
`os.spawn*` and `os.exec*` variant. Each synthetic test deliberately catches `PermissionError` and attempts to
continue; the child must nevertheless exit with the reserved containment status and the exact
stable attempt category, proving denial cannot be swallowed.

The test fixture supplies fake real-user paths but never creates, reads, inventories, or hashes
their contents. Assert that the denied attempt is durably present in `attempts.jsonl` before the
child returns the reserved status.

- [ ] **Step 2: Run the tests to verify RED**

```bash
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -k 'child or environment or tripwire' -q
```

Expected: FAIL because the child boundary does not exist.

- [ ] **Step 3: Implement scratch environment creation in the parent**

Set these before spawning the child:

```python
owned = {
    "HOME": scratch / "home",
    "XDG_CONFIG_HOME": scratch / "xdg-config",
    "XDG_DATA_HOME": scratch / "xdg-data",
    "XDG_CACHE_HOME": scratch / "xdg-cache",
    "XDG_STATE_HOME": scratch / "xdg-state",
    "TLDW_CONFIG_PATH": scratch / "xdg-config" / "tldw_cli" / "config.toml",
    "TMPDIR": scratch / "tmp",
    "TEMP": scratch / "tmp",
    "TMP": scratch / "tmp",
    "TLDW_TEST_MODE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
```

Create directories explicitly and prove each resolved writable owner is within `scratch`. Pass the original resolved `HOME`/XDG defaults only as denied-root strings; never enumerate those directories. Remove credential-bearing environment variables before the child starts.

- [ ] **Step 4: Implement the stdlib-only child before importing pytest**

The child may import only stdlib until it has:

1. resolved checkout, scratch, Python stdlib, interpreter, installed dependency roots, and the
   exact inert runtime resources required by the interpreter such as `os.devnull`;
2. pre-opened an append-only attempt-log file descriptor inside scratch;
3. installed `sys.addaudithook` plus narrow wrappers for any listed `os.system`/`os.spawn*`/
   `os.exec*` or `socket.bind`/`socket.listen`
   operation that the running interpreter does not expose through an audit event; and
4. captured the thread/task baseline used by the live tests.

Monitor at minimum `open`, `os.open`, mutating `os.*` events, `sqlite3.connect`,
`socket.connect`, `socket.sendto`, `socket.sendmsg`, `socket.bind`, `socket.listen`, every listed DNS operation,
`subprocess.Popen`, `os.system`, `os.posix_spawn*`, and every available `os.spawn*` and `os.exec*`. Allow only the
interpreter-internal local socket-pair creation needed by the event loop; deny outbound connect,
send, DNS, bind/listen, and process launch from the tested process. Reads are allowed only under
declared checkout/runtime/scratch roots and the enumerated inert runtime resources; writes only
under scratch.

Every prohibited event calls one allocation-minimal containment function. It writes a precomputed
stable category (never raw paths or arguments) to the pre-opened descriptor with `os.write`, calls
`os.fsync`, and terminates immediately with `os._exit(CONTAINMENT_EXIT_STATUS)` where the reserved
status is `86`. It must not raise a catchable exception. The parent treats status 86 as a containment
failure, stops the entire run, and can never promote its outputs. Disable pytest cache with
`-p no:cacheprovider` and put `--basetemp` and JUnit/results under scratch.

Use one tiny pytest plugin object to record exact node IDs and terminal outcomes:

```python
class ResultRecorder:
    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.failed):
            self.results[report.nodeid] = report.outcome.upper()
```

The child writes `automated-results.json` or `live-results.json` only under scratch after pytest returns.

Implement and test two explicit child dispatch modes. Pytest mode imports pytest only after the
boundary and uses `ResultRecorder`. Live mode imports an explicitly supplied scenario-module file
only after the boundary, finds an async callable in its finite `SCENARIOS` mapping, and executes that
one callable. Test the live dispatcher with a tiny synthetic scenario module under `tmp_path`, plus
an unknown-key case that returns `scenario_not_defined`.

- [ ] **Step 5: Run the boundary tests to verify GREEN**

```bash
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -k 'child or environment or tripwire' -q
../../.venv/bin/python -m py_compile Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py
```

Expected: PASS. Injected filesystem/network attempts must fail with stable categories and retained attempted-access facts.

- [ ] **Step 6: Commit**

```bash
git add Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Tests/Live/test_library_adaptive_reader_closeout.py
git commit -m 'test(library): isolate adaptive reader closeout runs'
```

### Task 4: Add the missing production-shaped closeout journeys

**Files:**
- Create: `Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py`
- Create: `Tests/UI/test_library_adaptive_reader_closeout.py`
- Modify: `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py`
- Modify: `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py`
- Modify: `Tests/Live/test_library_adaptive_reader_closeout.py`

- [ ] **Step 1: Write the 20-cell common matrix RED**

Declare exact destinations and sizes in the task-local scenario module. The child iterates the
Cartesian product and records 20 independently named results:

```python
async def run_common_cell(destination, terminal_size, context):
    app = await _seed_closeout_app(context.case_root(destination, terminal_size))
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=terminal_size) as pilot:
        screen = _active_library_screen(host)
        shell = await _open_destination(screen, pilot, destination)
        facts = _common_facts(screen, shell, destination, terminal_size)
        _assert_common_contract(facts)
        context.capture(host, f"{destination}-{terminal_size[0]}x{terminal_size[1]}", facts)
```

Facts must include containment, all five pane/grip regions, exact shell/items/work object identities, requested and effective preferences, selected/pending/loaded identity, focus owner, active host worker groups, visible controls, and compositor text. At 160x50 also collapse Library then Items and prove the Items region expands after Library collapses. At 80x24 prove Work remains mounted and both restore grips are painted/focusable.

In **every** cell, not only at 160x50, actively select a second deterministic record and wait for
matching loaded identity; reach every destination mode through the visible wide or compact control;
toggle Library closed/open and Items closed/open through their actual grips; and prove selection,
mode, focus, requested/effective preference truth, and exact shell/items/work identities survive
both restoration paths. At narrow widths where a requested-open pane remains effectively collapsed,
assert the truthful requested/effective distinction and reachable restore control rather than
claiming it painted open.

- [ ] **Step 2: Write the resize-purity and route-cycle REDs**

Add one parameterized resize test for all destinations. Wrap existing service/config/persistence seams with counters, resize `160x50 → 120x35 → 100x30 → 80x24 → 160x50`, and assert no list/detail read, worker start, config read, preference write, or poll occurs solely from effective geometry changes.

Add `test_closeout_preferences_restore_in_fresh_screen`. Using only the existing scratch config
seam, persist a closed shared Library pane and a distinct Items-open value for each of Media,
Conversations, Notes, Prompts, and Skills. Dispose the first screen, force a settings reload, create
a genuinely fresh app and `LibraryScreen`, visit all five destinations, and assert the shared
Library choice reloads everywhere while every destination restores only its own Items choice. This
is the durable restart/reload proof for SH-03; the route cycle below remains the in-process proof.

Add one single-app sequence:

```text
Media → Conversations → Notes → Prompts → Skills →
Media → Conversations → Notes → Prompts → Skills
```

Mutate only permitted session state: shared Library preference, one destination Items preference, Media mode/selection, Notes draft/mode, Prompt draft/mode, Skills mode/selection, and focus. On revisit assert each destination restores only its owned state, shared Library truth is shared, destination Items truth is isolated, no stale worker settles into the current route, and no duplicate shell/items/work owner remains mounted.

- [ ] **Step 3: Write five capability-journey REDs**

Use the established helpers/services from the existing reader tests; do not clone service
implementations. Add the five task-local async scenarios `run_media_capability`,
`run_conversations_capability`, `run_notes_capability`, `run_prompts_capability`, and
`run_skills_capability`.

Each journey asserts the exact catalogue behavior from the design and captures one structured result plus representative text/SVG frames. Use Media at `160x50`, Conversations at `160x50`, Notes at `120x35`, Prompts at `100x30`, and Skills at `80x24`; the common matrix already covers every destination at every size. Destructive actions stop at truthful preview unless the existing deterministic fixture explicitly confirms the operation and verifies recovery/selection truth.

- [ ] **Step 4: Run the new module to verify RED**

Run three separate RED probes so shared setup cannot hide which contract is absent:

```bash
../../.venv/bin/python Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py --development-run --live-case common_matrix --no-promote
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_closeout.py -k resize -q
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_closeout.py -k preferences_restore_in_fresh_screen -q
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_closeout.py -k route_cycle -q
```

Expected: each command fails for its named missing scenario/assertion. Then run each of the five
capability cases separately with `--live-case DESTINATION_capability --no-promote` and confirm its
own RED before implementing it.

- [ ] **Step 5: Implement with existing harnesses and settlement helpers**

Reuse:

- `LibraryProductionCSSHarness`, `_active_library_screen`, `_wait_for_condition`, `_wait_for_library_shell`, and `_wait_for_selector` from `Tests/UI/test_library_shell.py`;
- Media app/rows/services from `test_library_media_side_by_side.py` and `test_library_media_reader_flow.py`;
- Conversations deterministic/progressive/gated services from `test_library_conversation_reader.py`;
- Notes fixtures and editor helpers from `test_library_notes_reader.py` and `test_library_shell.py`;
- Prompt database/service/editor helpers from `test_library_prompts_reader.py` and `test_library_prompts_canvas.py`; and
- Skills local service wiring from `test_library_skills_reader.py`.

Do not use fixed sleeps. Wait for production worker/modal/widget/receipt state. Task-local scenarios
write only through the child-supplied raw evidence directory inside scratch; durable UI tests write
nothing.

The seed fixture owns a registry of every real in-memory/scratch database and service cleanup
callback. Capture the cleanup baseline immediately before each host mounts. After that host exits,
close its registry, settle/cancel only workers owned by that host, and compare threads/tasks
registered after that host-local baseline. The common facts must report the owner counts before and
after cleanup; no assertion claims process-wide ownership or treats framework activity created
between child startup and pytest/scenario import as a leak.

- [ ] **Step 6: Run the new module to verify GREEN**

```bash
../../.venv/bin/python Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py --development-run --live-only --no-promote
../../.venv/bin/python -m pytest Tests/UI/test_library_adaptive_reader_closeout.py -q
../../.venv/bin/python -m ruff check Tests/UI/test_library_adaptive_reader_closeout.py
```

Expected: 25 task-local live cases pass (20 common cells and five capability journeys), and seven
durable UI cases pass (five parameterized resize-purity cases, one fresh-screen preference reload,
and one sequential route cycle).
Record the actual counts rather than hard-coding them in evidence.

- [ ] **Step 7: Commit**

```bash
git add Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Tests/UI/test_library_adaptive_reader_closeout.py Tests/Live/test_library_adaptive_reader_closeout.py
git commit -m 'test(library): add cross-destination closeout journeys'
```

### Task 5: Add validation, cleanup, and atomic evidence promotion

**Files:**
- Modify: `Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py`
- Modify: `Tests/Live/test_library_adaptive_reader_closeout.py`

- [ ] **Step 1: Write failing result/promotion tests**

Add focused tests named
`test_success_requires_every_catalogue_id_to_have_fresh_automated_and_live_pass`,
`test_not_applicable_requires_catalogue_level_reason`,
`test_normalization_replaces_scratch_checkout_and_runtime_paths`,
`test_secret_or_user_path_rejects_the_whole_bundle`,
`test_only_allowlisted_relative_artifacts_are_promoted`,
`test_symlink_or_oversized_capture_is_rejected`,
`test_raw_root_is_absent_before_repository_promotion`,
`test_existing_unrelated_destination_is_never_replaced`, and
`test_owned_destination_replace_rolls_back_on_injected_failure`.

Parameterize promotion failure injection at every swap phase: after complete-stage validation,
after target-to-backup rename, after stage-to-target rename, and before backup removal. For each
phase, construct the post-crash filesystem state and prove the next invocation recovers one complete
valid bundle. Add separate residues for an unrelated lookalike stage/backup name and prove recovery
never deletes or renames it.

- [ ] **Step 2: Run to verify RED**

```bash
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -k 'catalogue or normaliz or secret or promote or cleanup or rollback' -q
```

Expected: FAIL because validation/promotion is incomplete.

- [ ] **Step 3: Implement validation and bounded promotion**

Required retained relative paths are finite:

```text
task23019_closeout.py
task23019_closeout_child.py
task23019_scenarios.py
README.md
manifest.json
summary.json
hashes.json
facts/*.json
captures/*.txt
captures/*.svg
```

The three Python files are pre-existing subject sources, not raw outputs: verify their subject hashes
and preserve them byte-for-byte. Reject symlinks, unknown paths, host/user absolute paths,
credential values, undeclared `NOT_APPLICABLE`, missing contract mappings, JSON above `256 KiB`,
text captures above `128 KiB`, SVG captures above `512 KiB`, or a total promoted output bundle above
`16 MiB`. Normalize output paths to `<checkout>`, `<runtime>`, and `<scratch>` in memory. Exit the raw
`TemporaryDirectory` and assert the raw root no longer exists before any repository write.

Build a **complete sibling directory** such as
`Docs/superpowers/reviews/evidence/.task-23019.stage-<nonce>`, never a stage inside the live evidence
directory. Copy the exact verified subject bytes for all three Python sources into it, add the
normalized managed outputs, write a task/subject identity marker, and validate the entire staged
bundle including hashes and size limits. Then perform one whole-directory swap: rename the current
`task-23019` directory to a sibling managed backup, rename the complete stage to `task-23019`, and
fsync the parent directory. Only after the new target validates may the owned backup be removed.
There is no in-place per-file replacement.

On every invocation, recover interrupted swaps before doing other work. If only an owned backup
exists, restore it; if a valid target and owned backup both exist, validate the target and then
remove the backup; if an owned stage residue exists, remove it only after its marker and subject
identity validate. Any collision lacking the exact TASK-23019 marker, subject identity, and expected
directory role is unrelated and must be left untouched while the operation fails closed. Tests
must prove the three Python files in the promoted target are byte-identical to the admitted subject.

Record SHA-256 hashes after normalization for every retained artifact **except `hashes.json`
itself**; the manifest declares that exclusion explicitly. Generate README/manifest from the
subject commit/tree held in memory, not from the later evidence commit.

- [ ] **Step 4: Run to verify GREEN**

```bash
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -q
../../.venv/bin/python -m ruff check Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py Tests/Live/test_library_adaptive_reader_closeout.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py Tests/Live/test_library_adaptive_reader_closeout.py
git commit -m 'test(library): validate adaptive reader closeout evidence'
```

### Task 6: Run the curated automated matrix and classify failures

**Files:**
- Modify only if a proven regression is admitted: the existing test and existing contract owner named by the failing catalogue ID.
- Modify if needed before repair: `backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md`

- [ ] **Step 1: Run runner self-tests and collect the curated matrix**

```bash
../../.venv/bin/python -m pytest Tests/Live/test_library_adaptive_reader_closeout.py -q
../../.venv/bin/python -m pytest --collect-only -q Tests/Library/test_library_adaptive_reader_state.py Tests/Library/test_library_media_reader_state.py Tests/Library/test_library_conversation_reader_state.py Tests/Library/test_library_notes_session.py Tests/Library/test_library_prompts_seam.py Tests/Library/test_library_skills_reader_state.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_media_reader_shell.py Tests/UI/test_library_media_reader_flow.py Tests/UI/test_library_conversation_reader.py Tests/UI/test_library_notes_reader.py Tests/UI/test_library_prompts_reader.py Tests/UI/test_library_skills_reader.py Tests/UI/test_library_multiselect_media.py Tests/UI/test_library_multiselect_conversations.py Tests/UI/test_library_multiselect_notes.py Tests/UI/test_library_adaptive_reader_closeout.py Tests/Chat/test_chat_conversation_service.py Tests/Prompt_Management/test_prompt_preservation.py Tests/Skills/test_skills_library_flow.py Tests/Skills/test_skill_trust_service.py Tests/Skills/test_local_skills_bundle_io.py Tests/Skills/test_read_skill_file.py Tests/Skills/test_skill_file_trust_material.py
```

Expected: collection succeeds with no unknown manifest node. Record the actual count.

- [ ] **Step 2: Classify every failure before editing**

Use exactly: contract regression, harness defect, environmental issue, or out of scope. A contract regression may enter this PR only if its expected result is already one of the 17 IDs and the fix is localized with no new schema/ADR/authority/capability/redesign.

- [ ] **Step 3: For an admitted regression, write one focused RED in the existing owner**

Choose the smallest owner:

| Contract | Existing product owner(s) |
| --- | --- |
| SH-* | `tldw_chatbook/Library/library_adaptive_reader_state.py`, `tldw_chatbook/Widgets/Library/library_adaptive_reader_shell.py`, or existing route code in `tldw_chatbook/UI/Screens/library_screen.py` |
| ME-* | existing Media reader state/widget/screen owner |
| CO-* | `library_conversation_reader_state.py`, `library_conversation_reader.py`, or existing screen orchestration |
| NO-* | existing Notes session/state/work-pane/screen owner |
| PR-* | existing Prompt state/work-pane/screen owner |
| SK-* | existing Skills state/work-pane/canvas/screen owner |

Run only the new failing node and prove it fails for the expected assertion, not an environmental reason.

- [ ] **Step 4: Implement the minimal repair and prove GREEN**

Do not add a generic abstraction. Re-run the focused node, the owning destination module, the shared cross-reader modules, and runner self-tests. If CSS changes, edit only the source module, run `build_css.py`, and commit all regenerated sheets.

- [ ] **Step 5: Commit each admitted repair separately**

Stage only the focused test and existing owner files selected in Steps 3–4, then commit with
`fix(library): restore CONTRACT-ID description`, substituting the admitted catalogue ID and its
existing behavior. Do not stage unrelated worktree changes.

If no contract regression exists, make no product commit.

### Task 7: Freeze and verify the exact subject revision

**Files:**
- No file changes during verification.

- [ ] **Step 1: Run all targeted static and derived-artifact checks**

Run these exact commands; they are the required named checks, not placeholders:

```bash
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
../../.venv/bin/python scripts/check_profile_owned_path_inventory.py
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python scripts/check_backlog_task_ids.py
../../.venv/bin/python -m ruff check Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py Tests/Live/test_library_adaptive_reader_closeout.py Tests/UI/test_library_adaptive_reader_closeout.py
../../.venv/bin/python -m ruff format --check Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py Tests/Live/test_library_adaptive_reader_closeout.py Tests/UI/test_library_adaptive_reader_closeout.py
../../.venv/bin/python -m py_compile Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py Tests/Live/test_library_adaptive_reader_closeout.py Tests/UI/test_library_adaptive_reader_closeout.py
git diff --check
git diff --check origin/dev...HEAD
```

If Task 6 modified product files, add those exact files to Ruff/format/py_compile and run `../../.venv/bin/python tldw_chatbook/css/build_css.py` before `check_bundle_sync.py` only when a CSS source changed.

Expected: every command exits 0. Do not run `pytest -q` for the whole repository.

- [ ] **Step 2: Commit the complete subject sources**

Only if there are remaining harness/test/scenario changes:

```bash
git add Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py Docs/superpowers/reviews/evidence/task-23019/task23019_closeout_child.py Docs/superpowers/reviews/evidence/task-23019/task23019_scenarios.py Tests/Live/test_library_adaptive_reader_closeout.py Tests/UI/test_library_adaptive_reader_closeout.py
git commit -m 'test(library): finalize adaptive reader closeout subject'
```

- [ ] **Step 3: Prove a clean subject revision**

```bash
git status --porcelain
git rev-parse HEAD
git rev-parse 'HEAD^{tree}'
```

Expected: status output is empty. Record the commit and tree as the subject revision; do not edit product, harness, tests, or scenario declarations after this point.

- [ ] **Step 4: Run the complete automated and live matrix through the parent**

```bash
TASK23019_SUBJECT_REVISION="$(git rev-parse HEAD)" ../../.venv/bin/python Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py --subject-revision "$(git rev-parse HEAD)" --promote
```

Expected: all curated automated nodes pass; all 20 destination/size cells, five capability journeys,
five resize-purity cases, the fresh-screen preference reload, and the route cycle pass; no prohibited
attempt or tracked-owner leak is recorded; raw scratch is removed before evidence promotion; the
command exits 0 and adds only managed output paths inside the already-existing
`Docs/superpowers/reviews/evidence/task-23019/` directory.

- [ ] **Step 5: Inspect representative live captures**

Inspect at minimum:

```text
media-160x50
conversations-120x35
notes-100x30
prompts-80x24
skills-80x24
single-app-route-cycle
```

Confirm compositor text matches structured facts, long Items titles expand when Library is collapsed, Work never disappears, restore grips are reachable, focus/footer copy is truthful, and no host path or secret appears.

### Task 8: Promote evidence-only closeout documentation

**Files:**
- Create: `Docs/superpowers/reviews/evidence/task-23019/README.md`
- Create: `Docs/superpowers/reviews/evidence/task-23019/manifest.json`
- Create: `Docs/superpowers/reviews/evidence/task-23019/summary.json`
- Create: `Docs/superpowers/reviews/evidence/task-23019/hashes.json`
- Create: bounded `Docs/superpowers/reviews/evidence/task-23019/facts/*.json`
- Create: representative `Docs/superpowers/reviews/evidence/task-23019/captures/*.{txt,svg}`
- Modify: `Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md`
- Modify: `backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md`

- [ ] **Step 1: Validate the promoted bundle against the subject**

```bash
../../.venv/bin/python Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py --verify-evidence Docs/superpowers/reviews/evidence/task-23019
git status --short
```

Expected: evidence validation passes. The runner and status output prove that changes from the
recorded subject add only managed output paths inside the already-existing
`Docs/superpowers/reviews/evidence/task-23019/` directory, plus the capability inventory and
TASK-23019; the three executable source files remain byte-identical to the subject revision.

- [ ] **Step 2: Update the capability ledger**

Append one programme-closeout section naming:

- subject commit and tree;
- all 17 catalogue IDs and their fresh automated/live status;
- all four terminal sizes and five destinations;
- route-cycle isolation result;
- any admitted repairs and focused RED/GREEN evidence; and
- the evidence bundle path.

Do not rewrite the earlier per-destination history.

- [ ] **Step 3: Prepare TASK-23019 completion notes**

Check every acceptance criterion only after its evidence exists. Add concise `## Implementation Notes` covering approach, files, repairs or “none,” targeted commands, live results, subject revision, evidence commit relationship, ADR check, and whether a generalized lesson was actually produced. Usually no new lesson is necessary; do not invent one.

Keep the task `In Progress` until the final checks, evidence commit, final-HEAD proof, and explicit
self-review below have all passed.

- [ ] **Step 4: Re-run final-head documentation and derived checks**

```bash
../../.venv/bin/python Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py --verify-evidence Docs/superpowers/reviews/evidence/task-23019
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
../../.venv/bin/python scripts/check_profile_owned_path_inventory.py
../../.venv/bin/python scripts/check_persistent_diagnostic_inventory.py
../../.venv/bin/python scripts/check_backlog_task_ids.py
git diff --check
```

Expected: every command exits 0. The runner also proves product/harness/test/scenario source hashes at final HEAD match the recorded subject revision.

- [ ] **Step 5: Commit the evidence-only closeout candidate**

```bash
git add Docs/superpowers/reviews/evidence/task-23019 Docs/superpowers/reviews/2026-08-24-library-adaptive-reader-capability-inventory.md 'backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md'
git commit -m 'docs(library): close adaptive reader programme'
```

- [ ] **Step 6: Self-review and prove the committed closeout candidate**

Review `git diff origin/dev...HEAD` against every TASK-23019 acceptance criterion and all 17
catalogue mappings. Explicitly check for unrelated changes, path/secret leakage, stale subject
hashes, missing evidence, unowned cleanup, incomplete failure handling, and test assertions that do
not prove their stated contract. Fix any issue before continuing; a source fix invalidates the
subject and returns execution to Task 7.

```bash
../../.venv/bin/python Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py --verify-evidence Docs/superpowers/reviews/evidence/task-23019
git diff --check origin/dev...HEAD
git status --short
```

Expected: self-review finds no unresolved issue, evidence verifies, diff check is clean, status is
empty, and the committed candidate differs from the subject revision only by the retained evidence,
capability ledger, and TASK-23019 documentation.

- [ ] **Step 7: Mark TASK-23019 Done only after Definition of Done is satisfied**

Run only after Steps 1–6 pass:

```bash
backlog task edit 23019 -s Done --notes 'Closed the five-destination Library adaptive-reader programme with one bounded hermetic automated/live matrix tied to the recorded subject revision. Verified ADR-086 without changing storage, service authority, security, or application structure. See Docs/superpowers/reviews/evidence/task-23019/ and the capability inventory.'
git add 'backlog/tasks/task-23019 - Close-Library-adaptive-reader-programme-with-cross-destination-UAT.md'
git commit -m 'docs(backlog): complete adaptive reader closeout task'
```

- [ ] **Step 8: Prove the actual final HEAD**

```bash
../../.venv/bin/python Docs/superpowers/reviews/evidence/task-23019/task23019_closeout.py --verify-evidence Docs/superpowers/reviews/evidence/task-23019
../../.venv/bin/python scripts/check_backlog_task_ids.py
git diff --check origin/dev...HEAD
git status --short
```

Expected: evidence and task-ID checks pass, the diff is clean, status is empty, TASK-23019 is Done,
and actual final HEAD differs from the recorded subject only by the allowed evidence, capability
ledger, and task documentation paths.

## Execution notes

- A harness defect is fixed in the harness and rerun; it is not reported as a product regression.
- An environmental issue remains a failed/blocked result with its exact category; it is never converted to PASS.
- A finding outside the finite catalogue becomes a separate Backlog task and does not expand TASK-23019.
- Any source change after Task 7 invalidates the subject revision. Commit the change, discard the old uncommitted evidence bundle if it is owned by TASK-23019, and rerun Task 7 completely.
- The full repository suite remains out of scope unless the user explicitly opts in later.
