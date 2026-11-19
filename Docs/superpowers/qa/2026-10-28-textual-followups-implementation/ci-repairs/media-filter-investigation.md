# PR3290 Media filter preparation: native investigation

Date: 2026-10-28. Scope: bounded read-only investigation, no product/test/task/Git edits. Temporary probe code and evidence live under `.test-tmp/textual-37266-37268/ci-media-filter/`. Root owns any subsequent task, specification and implementation allocation.

## Finding

The exact CI symptom is reproducible through native spaced printable keys. The filter's debounce publishes loading/results by recomposing the Items canvas. Each replacement filter Input is prefilled from the requested/applied query and receives restored focus. Textual's default `select_on_focus=True` selects the entire prefill. A subsequent printable character reaches that **fresh, attached Input** and replaces the selected prefix. Repeating this across the six `roduct` characters leaves `t`, with both fixture rows, exactly as CI reported.

This establishes a real Media typing defect, and explains why the Reader identity test's existing two-stage `P` / `roduct` setup is timing-sensitive. It must not be described as merely a stale retained-widget test bug. A single native Paste is a verified robust way to prepare the filter for this particular Reader identity regression; it does not repair or discharge the separate typing obligation.

Observed versus inferred:

- **Observed in CI:** UI Fast Lane (12), run 40815454377, log `pr-3290/logs-3e414ff8/123255322945.log`, failed at the original line 192: query `roduct` never settled, applied query `t`, rows 2. The case took 22.77s; the job had 1 failed / 66 passed in 611.57s. No timeout.
- **Observed natively:** the original fast gesture passes; adding only 1.2s gaps between its existing `roduct` printable keys reproduces the verbatim assertion, with replacement Input identities and selected-prefill replacement captured below.
- **Inferred about the uninstrumented CI occurrence:** its scheduler allowed one or more debounce/recompose/focus cycles inside that burst. CI has no event trace, so its exact character-by-character interleaving cannot be claimed as directly captured. The native witness matches its terminal query/row state and exposes a sufficient production mechanism.
- A stale retained Input is not needed for the native witness: the next character is processed by the currently attached replacement. Partial-query restoration occurs as requested/loading and applied projections create fresh fields; the directly demonstrated loss is the fresh field's select-all followed by native character replacement.

## Source ownership and flow

Read the repository AGENTS instructions, `Docs/Architecture/library-shell.md`, relevant testing/live-verification lessons, TASK-32350, TASK-34000.18 and ADR-086. The lesson about releasing held responses inside `run_test` applies to diagnostic holds; this probe introduced none. The checkout-provenance lesson is covered by loaded module paths and hashes in each trace.

1. `_type` in `Tests/UI/test_library_media_editor_dirty_escape.py:131` sends one `pilot.press` per character. It provides no atomic burst boundary.
2. The existing Reader case intentionally settles `P`, clicks the fresh field, presses End, then sends six more independent keys. Its comment already acknowledges loss across the first recompose, but the second burst can cross more recomposes.
3. `LibraryMediaCanvas.handle_library_media_filter_changed` at `Widgets/Library/library_media_canvas.py:446` forwards the real Input event to the controller.
4. `LibraryMediaController.handle_library_media_filter_changed` at `UI/Library_Modules/library_media_controller.py:2338` captures `event.value` in a 0.25s timer. `_request_library_media_filter` requests authoritative browse. Submitted Input uses the same request seam immediately.
5. `_sync_library_media_browse_state` at `UI/Screens/library_screen.py:17146` projects state and queues focus restoration through `_sync_library_canvas`. `LibraryMediaCanvas.sync_state` at `:593` calls `refresh(recompose=True)`.
6. The canvas compose block around `library_media_canvas.py:1086` creates `Input(value=self.canvas.query)` with default selection behavior. Controller `_build_library_media_state` at `:1907` normally uses the applied result and substitutes the inflight query during loading at `:1956`.
7. Native Textual `_input.py` `_on_focus` selects `Selection(0,len(value))`; `_on_key` replaces a nonempty selection with the printable character. Its `_on_paste` inserts/replaces the entire Paste payload in one event.

A read-only `git diff --stat 7c2b8f5aab 3e414ff824 --` over the three involved test files, controller, screen and canvas returned no differences. The loaded application modules in the traces resolve to this worktree, not the main editable checkout. Native Textual resolves to the specified project venv.

## Native evidence

All calls used the approved `run-targeted.py` launcher, specified project Python with `-X utf8`, native 700 external profile/TEMP, existing bootstrap-profile marker, unchanged 180s guard and unchanged 15s condition helper. One App batch ran at a time. No feature-test sweep or CI rerun occurred. The probes do not assign Input values, controller state or query scopes, and do not replace getters or service returns. Observation wrappers call the original native methods; the paced arm changes only inter-key spacing. The full-Paste arm substitutes only filter preparation in a temporary compiled copy of the original function.

| Run directory | Gesture | Result | Call / total |
| --- | --- | --- | --- |
| `exact-native-01` | Original unchanged `P`, settle, End, `roduct` | 1 passed, 0 errors/skips | 41.90s / 84.47s |
| `paced-native-01` | Same original case; 1.2s native pauses between `roduct` keys | 1 intended failure, 0 errors/skips; verbatim CI query `t`, rows 2 | 42.86s / 70.41s |
| `fullpaste-native-01` | One native App `Paste("Product")` after the real filter click | 1 passed, 0 errors/skips | 34.36s / 60.90s |

Each run retained its raw `run.log`, `command.json`, `result.xml` and `verification.json`. There were no setup/teardown errors, no timeout and no diagnostic hold to release. All three native processes exited normally through their App contexts.

Key paced trace observations (monotonic seconds; raw `paced.trace.json`):

| Time | Native observation |
| --- | --- |
| 631769.078 | Field `0x2ed3677cd10`, value `P`, collapsed caret 1, handles `r` and becomes `Pr`. |
| 631769.343 | Controller requests `Pr`. |
| 631769.796 | Replacement `0x2ed35575850` receives focus with `Pr`; selection becomes `(0,2)` while request `Pr` is loading. |
| 631770.000 | Settled replacement `0x2ed36d609b0` receives focus with `Pr`; selection becomes `(0,2)`, applied=requested `Pr`. |
| 631770.375 | The same attached fresh field handles `o`, replacing `Pr` with `o`. |
| 631771.640 onward | Further fresh fields handle `d`, `u`, `c` with their previous one-character values selected. |
| 631775.796 | Fresh field `0x2ed354354f0`, value `c`, selection `(0,1)`, handles `t` and becomes `t`. |
| 631776.609 | Final replacement receives focus with `t`; applied=requested `t`; original condition later raises with rows 2. |

The original passing trace shows all six suffix characters entering one field in about 0.5s, each before the next debounce expiry; it becomes `Product` before recomposition. Thus the original test can pass without disproving the defect.

The Paste trace records exactly one `_on_paste` with `Paste(text='Product')` on an empty filter, changing it to `Product` atomically at 631881.796, followed by authoritative controller requests for `Product`. It passes the original off-page row, trash/back, Reader identity, highlighted-row absence, editing/dirty-title, service write-target/title and no-toast assertions.

`summarize.py` structurally compares the AST statements following `_browse_settled` in the original and temporary Paste function: `fullpaste_remaining_reader_assertions_ast_identical=true`. No Reader assertion was removed or softened. The temporary compiled function is retained as `fullpaste-probe-function.py`.

## Minimal proposed setup repair

Replace the entire two-stage P/roduct preparation and its timing comment in this Reader identity case with one native Paste after `_focus_filter_box`, retaining the authoritative browse and row checks:

```python
await _focus_filter_box(screen, pilot)
pilot.app.post_message(Paste("Product"))
await _browse_settled(screen, pilot, query="Product", rows=1)
assert [str(row.media_id) for row in _rows(screen)] == [ROW_0]
```

Import `Paste` from `textual.events`. Continue through the existing trash/back and every Reader/write/toast assertion unchanged. `_query_settled` is only used by the old preparation in this module and can be removed if it becomes unused. Do not modify shared `_type` or `_browse_settled`, increase waits, retry a failed target assertion, or change fixture/service behavior. This is verified native filter preparation for a test about Reader action identity, whose original task is TASK-34000.18.

## Product obligation and scope implications

The native paced-key witness should become a separate bounded Media regression/repair task allocated by root, not silently disappear when this Reader test starts pasting. The owner is the Media canvas/controller browse-publication and focus path, not the Reader action-target accessor. Required behavior: native typing/editing through loading and accepted-result publications keeps the current draft and caret/selection, while the scope line continues to describe the applied results under TASK-32350. The Reader dirty pin and off-page-selection behavior must remain unchanged.

The smallest product direction to evaluate is preserving the live filter control/draft/selection through result publication. A bare `select_on_focus=False` changes a demonstrated symptom but does not establish preservation of newer text across all loading/settled rebuilds; it is not a verified repair from this investigation. Any rebuild-based alternative needs explicit preservation of the latest live draft and selection and protection against obsolete publication. No product implementation was attempted here.

For the test-only preparation plan:

```text
ADR required: no
ADR path: N/A
Reason: native regression setup repair, preserving existing Library ownership and behavior.
```

For a contained product typing bugfix preserving the existing boundaries, no new ADR should be needed; ADR-086 leaves destination fields/drafts with Media, and TASK-32350 already defines draft-versus-applied presentation. Root must reassess ADR need if it chooses a larger structural or service-contract change. Root owns the new task, acceptance criteria, plan and later dispatch. The existing TASK37266–37268 should not be recharacterized as the cause: the investigated inputs are identical between their green and failing heads.

## Receipt integrity

`evidence-receipt.json` retains SHA256 for all six investigated current source files, all three XML files and all three raw traces. Each raw trace also records actual loaded Input/canvas/controller module paths and SHA256. XML SHA256:

- Exact: `5f2cbfe431b704be8d0f852f1dfb38a17808cfd4575ea21fecaf885acc99f0bb`
- Paced: `5cf935686ce4f2820f3754d2dc66e8b0d6d8b4b99c08357fe964b229afff2ba3`
- Paste: `8575a3dbac5979aef33ed645d390d7600748a95289a10ff00494c97a94397de8`

Limits: three bounded native cases establish this mechanism and the proposed setup gesture, not a race frequency estimate, Linux-specific scheduling trace, complete product typing repair, full-suite result or new hosted-CI success.
