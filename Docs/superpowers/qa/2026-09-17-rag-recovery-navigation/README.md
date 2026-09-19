# Search/RAG recovery navigation — TASK-32720

2026-09-17 UTC, `feat/component-pattern-library`, based on `4f9749f846`.

Three defects were repaired within the existing navigation and design contracts:

- An unselected provider now offers the same **Open Settings ▸ Providers** action
  as a selected provider with a missing credential. The two cases retain their
  distinct structured recovery records; the visible reason uses the established
  Settings remedy.
- Reopening a restored Search/RAG draft synchronizes the rail query even when the
  panel's mount event already matches state. It does not reset settled results.
- Resuming a retained Library screen refreshes Run and conditional recovery from
  current provider readiness, even when no source counts changed. The existing
  serialized refresh preserves the mounted evidence results and history.

ADR required: no. Existing ADR-003 (Settings/Library boundary), ADR-031 (keys),
ADR-150 and ADR-161 (design language/components) apply. No new storage, service,
route, stylesheet, visual token or dependency was introduced.

## Automated evidence

[Missing-provider red](missing-provider-red.txt) records the absent keyboard
recovery target; [query-mirror red](query-mirror-red.txt) records the blank rail
field beside the restored query. [Four retained-screen red cases](retained-gate-red.txt)
reproduce stale Run readiness in both themes and sizes. The initial probes also
needed two harness corrections: use the actual `ingest-import-media` row ID and
current state-owner fields, and reopen Search/RAG after a fresh screen enters
through the Library landing page. These setup failures were not product defects.

The [initial wider selection](initial-targeted-tests.txt) returned 317 passes and
one stale assertion: a credential test still demanded technical environment/TOML
copy on screen, although TASK-32236 already moved it into the recovery record/log.
The updated assertion checks the Settings button and visible reason, absence of
technical UI copy, and preservation of the provider-specific recovery record.

The [final targeted selection](targeted-tests.txt) passes **324 tests in 293.58s**.
It covers display-state contracts,
21 new navigation cases, empty-scope recovery, history, gate races, resize/query
return, real credential readiness, rail submission, mirror traffic, retained
visits and the existing Search/RAG maturity cases. It is not a full suite.

[Static checks](static-checks.json) record successful new-file lint/format checks,
format checks of changed existing ranges, and no new lint diagnostics against
HEAD for the existing large files. [Independent review](review.json) found a
missing current-container identity assertion in the retained results test; it
was added. The new check verifies both history and results stay mounted.

## Native evidence

The [runner](native_check.py) boots real TldwCli with a private validated profile,
actual terminal driver, local SQLite databases and an acquired instance lock.
All [twelve round trips](result.json) pass: Import plus both unselected-provider
and missing-credential Settings recovery in dark/light at 170×48 and 80×24.
Recovery links are reached with Tab from the query and activated with Enter.
Actual Settings must select Providers & Models and mount its card. Ctrl+3 returns
to Library; the runner opens Search/RAG if the landing page is shown.

Provider readiness changes are ephemeral config fixtures, **not saved Settings
edits or credential validation against a server**. The unchanged-return case
stays blocked; the second case switches to the keyless Ollama readiness gate and
returns ready. No server is called. Retrieval, answer and ingest sentinels record
zero submissions; the source record remains unchanged and no ingest jobs exist.
The automated retained-screen tests additionally cover ready-to-blocked return.

Two native setup attempts tried to focus the hidden rail from compact Import:
[first](initial-native-harness-failure.json),
[second](second-native-harness-failure.json). The runner now activates Import's
existing `library-rail-open` handle as well as the adaptive reader grips. The
[next attempt](native-retained-gate-red.json) then exposed the genuine stale
provider gate on reuse, which fresh-screen tests had missed. That defect was
reproduced with four automated cases before the resume repair.

All eight final screenshots were rendered and visually inspected in one batch.
They show visible, non-obscuring keyboard focus and readable recovery text,
including wrapping at compact width. No visual changes were needed.
Only trailing SVG source-line whitespace was normalized; [hashes](capture-hashes.json)
record original and stored bytes.

| Theme / size | Import recovery | Provider recovery |
| --- | --- | --- |
| Dark / 170×48 | [Capture](textual-dark-170-import-link.svg) | [Capture](textual-dark-170-provider-link.svg) |
| Dark / 80×24 | [Capture](textual-dark-80-import-link.svg) | [Capture](textual-dark-80-provider-link.svg) |
| Light / 170×48 | [Capture](textual-light-170-import-link.svg) | [Capture](textual-light-170-provider-link.svg) |
| Light / 80×24 | [Capture](textual-light-80-import-link.svg) | [Capture](textual-light-80-provider-link.svg) |

[Lifecycle evidence](lifecycle.json) records normal app stopping, app-run return,
exit 0, absence of the exact PID before terminal cleanup, eleven healthy private
databases, zero conversation messages, unchanged default config/UI-state/policy
fingerprints, no error/critical/traceback log entries, and empty faulthandler logs.
[Allocation checks](id-check.json) record the task's sole owner.

The guide and audit explain the recovery/return behavior. The testing lesson
records why fresh-screen restoration cannot stand in for actual retained-screen
navigation. No full suite, push or merge was performed. Next bounded review:
Providers & Models keyboard editing, save/cancel and return feedback.

## Reproduce the targeted selection

```sh
.venv/bin/python -m pytest -q \
  Tests/Library/test_library_rag_state.py \
  Tests/UI/test_library_rag_recovery_navigation.py \
  Tests/UI/test_library_rag_scope_recovery.py \
  Tests/UI/test_library_rag_history_keyboard.py \
  Tests/UI/test_library_rag_query_gate_race.py \
  Tests/UI/test_library_rag_query_return.py \
  Tests/UI/test_library_shell.py::test_library_shell_search_rag_mode_blocks_run_without_a_ready_provider \
  Tests/UI/test_library_shell.py::test_library_shell_search_rag_mode_blocks_run_when_endpoint_named_but_credential_missing \
  Tests/UI/test_library_shell.py::test_library_shell_rail_search_submit_runs_search_canvas_query \
  Tests/UI/test_library_shell.py::test_library_shell_rail_search_empty_submit_selects_without_service_call \
  Tests/UI/test_library_shell.py::test_library_shell_stale_mirror_events_do_not_replenish_changed_traffic \
  Tests/UI/test_library_shell.py::test_library_shell_repeat_visit_composes_exactly_once_when_data_is_unchanged \
  Tests/UI/test_library_crit9_grammar.py::test_a_missing_provider_key_paints_one_line_and_no_owner_block \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  --tb=short --show-capture=no
```
