# Verification commands and provenance

All commands ran in
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev`,
using `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` and Ruff
from that same venv. `PYTHONPATH=$PWD`. No packages installed. Paths below are
relative to that worktree; `S` abbreviates
`.superpowers/sdd/2026-09-16-workflows-first-run` in this record only.

## Joined RED / GREEN

The profile preparation command imports no application modules:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  Tests/Workflows/test_file_to_note_integration.py \
  .superpowers/sdd/2026-09-16-workflows-first-run/task-6-bootstrap
```

With `TLDW_TEST_CONFIG_ROOT=$PWD/S/task-6-bootstrap` (expand S), exact pytest
arguments were:

```text
-m pytest -o addopts= -q --tb=short --show-capture=no
--basetemp=S/task-6-red-profile
Tests/Workflows/test_file_to_note_integration.py
```

Temporary task-owned test-only mutation before editing the review:
`monkeypatch.setattr(session, "update_review", lambda *args: False)`.
This removed the real edit-to-session wire without changing the expected content
or stubbing the HTTP/Notes boundaries. RED failed the exact saved content
assertion. The mutation was removed, and the identical arguments with
`--basetemp=S/task-6-green-profile` gave GREEN. Outputs are `integration-red.txt`
and `integration-green.txt`. The owned loopback bind requires sandbox network
escalation on this host; the initial EPERM is not behavioral RED.

## Merged manifest

Prepared `S/task-6-regression-bootstrap` using the same preparation command.
`TLDW_TEST_CONFIG_ROOT=$PWD/S/task-6-regression-bootstrap`, `PYTHONPATH=$PWD`:

```text
-m pytest -o addopts= -q --tb=short --show-capture=no
--basetemp=S/task-6-regressions-01
Tests/Workflows/
Tests/LLM_Calls/test_llamacpp_bounded.py
Tests/Chat/test_provider_endpoint_contract.py
Tests/Chat/test_local_server_discovery.py
Tests/Chat/test_sensitive_llm_logging.py
Tests/Agents/test_builtin_tool_gate.py
Tests/MCP/test_permission_store.py
Tests/MCP/test_permission_resolution.py
Tests/Notes/test_notes_scope_service.py
Tests/Notes/test_notes_library_unit.py
Tests/DB/test_chachanotes_diagnostic_path_privacy.py
Tests/UI/test_workflows_run.py
Tests/UI/test_workflows_editor.py
Tests/UI/test_workflows_paging.py
Tests/UI/test_app_quit_guard.py
Tests/UI/test_design_token_governance.py
Tests/UI/test_css_bundle_sync_guard.py
Tests/UI/test_workflows_stylesheet_loading.py
Tests/UI/test_workflows_projection_performance.py
```

Separate bootstrap-sensitive invocation with the same pytest flags:
`--basetemp=S/task-6-lifecycle Tests/ProductionApp/test_workflows_session_lifecycle.py`.
The ProductionApp conftest chooses its own private collection root, so this was
never mixed with live-profile collection.

Focused failure diagnosis (same flags, fresh basetemp for each):

| Output | Basetemp suffix | Selectors |
| --- | --- | --- |
| regression-failure-repro.txt | task-6-failure-repro | `Tests/Chat/test_sensitive_llm_logging.py::test_sensitive_direct_llama_logs_no_request_response_or_endpoint_secrets` then `Tests/UI/test_workflows_run.py::test_pressed_flow_edits_note_navigates_and_paints[size1]` |
| logging-order-repro.txt | task-6-logging-order | `Tests/Workflows/test_authoring.py::test_real_app_create_edit_navigate_quit_and_restart` then the sensitive-direct-llama selector above |
| ui-combined-repro.txt | task-6-ui-combined | `Tests/Workflows/test_file_to_note_integration.py::test_saved_file_controls_real_http_edited_local_note` then `Tests/UI/test_workflows_run.py::test_pressed_flow_edits_note_navigates_and_paints` (all 3 widths) |

## Actual live endpoint

Prepare a NEW task-owned profile with the preparation command, then invoke:

```text
python Tests/Workflows/test_file_to_note_integration.py
  S/task-6-live-b walk 160x48 Docs/Developer/Workflows/artifacts/2026-09-16-first-run
python Tests/Workflows/test_file_to_note_integration.py
  S/task-6-live-b restart 160x48 Docs/Developer/Workflows/artifacts/2026-09-16-first-run
```

Repeat walk/restart sequentially for `110x36` and `60x20`. Use the root venv Python
above. The launcher scrubs ambient credential/proxy/cache/root selectors, records
removed names only, then starts a NEW `python -m pytest` interpreter for exactly
`Tests/Workflows/test_file_to_note_integration.py::test_live_full_app`, with
`--basetemp=<profile>/probes/<size>-<phase>` and the same common pytest flags.
The live test is skipped in ordinary regression collection. No live POST occurs
from a plain preparation, ordinary integration test, or restart.

The first profile `S/task-6-live-a` and its failed broad diagnostic assertion are
preserved separately in `attempt-1/`. Never reuse these evidence output names
when doing a new walk; preserve this qualification packet.

## Static checks

```text
ruff check Tests/Workflows/test_file_to_note_integration.py
ruff format --check Tests/Workflows/test_file_to_note_integration.py
python S/check_static_delta.py --base cf61cb68505fc62991a0488c964a78cb7ab31cbb
  --ruff /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff
  <25 explicit paths listed as file fields in static-delta.json>
```

`static-current-ruff.txt` is the complete `ruff check --output-format concise`
output on those same 25 paths. `static-attribution.md` explains the three moved
legacy typing-import diagnostics that conservatively fail automatic span mapping.
No suppressions or broad formatting were used. Final `git diff --check` and exact
staged path review are required before committing.

## Approved failure corrections and covering runs

Same Python, environment and common pytest flags as above; every basetemp below
is under S and fresh for its invocation. Network escalation permits owned
loopback listeners, not a substitute live backend.

| Output | Basetemp suffix | Exact selection |
| --- | --- | --- |
| logging-order-green.txt | task-6-logging-green | `Tests/Workflows/test_authoring.py::test_real_app_create_edit_navigate_quit_and_restart Tests/Chat/test_sensitive_llm_logging.py` |
| library-held-permanent-red.txt | task-6-held-permanent-red | `Tests/Workflows/test_file_to_note_integration.py::test_saved_file_controls_real_http_edited_local_note[held-nested-mount]` before canvas fix |
| library-held-green.txt | task-6-held-green | `Tests/Workflows/test_file_to_note_integration.py` after canvas fix |
| library-focused-green.txt | task-6-library-focused | Expanded Library paths below; historical intended filename, actually 170 passed / 1 unrelated failure |
| library-focused-notes.txt | task-6-library-focused-notes | Notes-specific selection below, 165 passed |
| skills-stub-baseline-focused.txt | task-6-skills-baseline | `Tests/UI/test_library_canvas_scoped_sync.py::test_import_status_lines_patch_the_mounted_static_without_recompose` (unchanged baseline symptom, 1 failed) |
| merged-regressions-final.txt | task-6-regressions-final | Entire original merged manifest above, unfiltered |

Expanded selection:

```text
Tests/UI/test_library_canvas_scoped_sync.py
Tests/UI/test_library_canvas_sync_defects.py
Tests/UI/test_library_notes_riders_backlinks.py
Tests/UI/test_library_notes_work_session.py
Tests/UI/test_workflows_run.py::test_pressed_flow_edits_note_navigates_and_paints
Tests/Notes/test_notes_scope_service.py
Tests/Notes/test_notes_library_unit.py
```

Notes-specific follow-up retains all paths except the first file, which is
replaced by its three relevant selectors:

```text
Tests/UI/test_library_canvas_scoped_sync.py::test_loading_surfaces_keep_unicode_copy_and_notes_sync_hook
Tests/UI/test_library_canvas_scoped_sync.py::test_notes_per_click_updates_keep_screen_and_canvas_identity
Tests/UI/test_library_canvas_scoped_sync.py::test_notes_select_toggle_latency_probe
```

This selection transparently excludes the unrelated Skills-import stub failure;
no assertions were changed. Exact base/current source hashes and fresh focused
failure are preserved. The authorized Workflow merged manifest contains no
`test_library_canvas_scoped_sync.py` and is run without any deselection.

The held test uses an event barrier around the real nested mount, not sleeps or
stress repetition. It supplies preview then context state while the toolbar is
not mounted; after release, the queued callback observes context already active
and focuses it. There is no manual second apply to satisfy the post-mount check.

Final static comparison uses the same command with the 28 explicit `file` paths
in `static-final-delta.json`; the three additions are the new Task6 integration,
`Tests/Chat/test_sensitive_llm_logging.py`, and
`tldw_chatbook/Widgets/Library/library_notes_canvas.py`.

Fresh post-fix live command used the same launcher with
`S/task-6-live-postfix walk 110x36 artifacts/2026-09-16-first-run/postfix`
(artifact directory expanded from `Docs/Developer/Workflows/`). It made 3 actual
POSTs and completed all controls but failed the final whole-authority-table
equality check. Packet is preserved unchanged; see `postfix-config-diagnosis.md`.

Passing post-fix pair uses NEW `S/task-6-live-qualified`, walk then restart at
`110x36`, artifact destination
`Docs/Developer/Workflows/artifacts/2026-09-16-first-run/postfix-qualified`.
Authority defaults were predeclared instead of relaxing table equality; selected
model authority and effective post-quit paths were verified as described in UAT.

## Editor stale-callback RED / GREEN (approved Ruling 13)

Same pytest environment/flags as above:

| Output | Basetemp suffix | Selection |
| --- | --- | --- |
| editor-hit-focused.txt | task-6-editor-hit-repro | `Tests/UI/test_workflows_editor.py::test_incomplete_field_is_repairable_across_fresh_screens[step:summarize-/steps/2/config/max_tokens--256]` |
| editor-hit-combined.txt | task-6-editor-combined | `Tests/UI/test_workflows_run.py Tests/UI/test_workflows_editor.py` (113 passed before fix; not a waiver) |
| editor-hit-traced.txt | task-6-editor-trace | Same exact failing selector plus diagnostic plugin below |
| editor-hit-held-layout.txt | task-6-editor-held-layout | Same selector/plugin, `TASK6_EDITOR_HOLD_LAYOUT=1` |
| editor-permanent-red.txt | task-6-editor-permanent-red | `Tests/UI/test_workflows_editor.py::test_deferred_view_restore_preserves_newer_field_focus_and_invalid_paint` before fix |
| editor-permanent-green.txt | task-6-editor-permanent-green | Permanent test above plus `Tests/UI/test_workflows_editor.py::test_incomplete_field_is_repairable_across_fresh_screens` (all three cases) |
| editor-final-covering.txt | task-6-editor-final-covering | Exact final covering selection below |

The temporary diagnostic plugin was loaded with
`-p Tests.Workflows._task6_editor_probe`; application imports ran only in its
fixture after existing pytest isolation. It is archived as
`editor-diagnostic-plugin.py.txt`, not retained as a production/test plugin.
The permanent regression is independent of that probe and its environment flag.

```text
Tests/UI/test_workflows_editor.py
Tests/UI/test_workflows_run.py
Tests/UI/test_workflows_paging.py
Tests/UI/test_workflows_projection_performance.py
Tests/Workflows/test_file_to_note_integration.py
```

No third 1,557-test merged sweep: controller authorized this affected-boundary
coverage after the final editor fix. The previously merged 1,555 passes remain
historical evidence; the one failure has separate deterministic RED/GREEN proof.
Static `static-commit-delta.json` covers30 explicit paths, with complete current
diagnostics in `static-commit-current-ruff.txt`. Clean three-file checks are in
`task6-commit-{lint,format}.txt`.
