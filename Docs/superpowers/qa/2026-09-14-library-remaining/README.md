# Library footer, entry, and Prompts repairs

TASK-32462; baseline `9126f1b43d`, branch `feat/component-pattern-library`.
The Workspace portion was repaired earlier in `af5a26a2af`. This follow-up
repairs the remaining three files and the product defects they exposed.

## Product repairs

- **Notes focus:** same-route canvas replacement restored focus before the
  resident canvas rebuilt. Rebuilding then removed the focused row and restored
  the filter. Restore now runs through the existing post-recompose callback,
  using the Notes identity captured before detachment. A newer user focus move
  still vetoes restoration. The added delayed-route-swap case failed before the
  fix; both it and the original selected-row regression pass afterward.
- **Prompt deletion:** closing the confirmation resumes Library while the
  mutation worker is still active. That resume issued a pre-write page read and
  count refresh, racing the worker's own refresh and survivor focus. The active
  mutation now owns settlement. Four survivor/empty-page cases, final-page
  clamping, and failed-refresh read-only state pass without loosening their
  exactly-once assertions. Ordinary leave/return refresh still runs.
- **First Prompt save:** the broad count snapshot also synchronized the work
  pane, rebuilding the saved editor fields. Snapshot reconciliation now updates
  the Items pane and rail; detail/save handlers retain editor ownership. The
  in-place save-action update also adopts the new saved identity, so clean-state
  actions do not keep treating the Prompt as unsaved. The real SQLite journey
  retains the original text widget, checks persisted content/version and the
  updated count, and verifies Save hides while Use in Console appears.

## Test repairs, distinct from product changes

| Test group | What was wrong and what is asserted now |
| --- | --- |
| Footer focus | The pure fake lacked the narrow-stage gate and mount state. It now describes an unmounted, wide Media context; all seven original shortcut assertions remain. |
| Pending conversation selection | The test tried to focus old rows that are intentionally disabled during a locator request. It now issues a second admitted navigation, establishes enabled row focus, and then releases the older request. Selection, owner identity, focus and SUPERSEDED remain required. |
| Dirty Prompt navigation | Two tests expected silence despite the shipped unsaved-changes warning. They now require that exact warning while retaining basket/selection and navigation-veto checks. |
| Prompt teardown | Calling an unmounted screen's unmount handler failed before reaching workspace shutdown. A production-styled mounted screen is actually popped; a late result is refused while shutdown is held, and the last accepted result remains unchanged. |
| Five CSS parity tests | Rules moved to `_library_panels.tcss` and the lazy Library screen stylesheet; values became design tokens. Checks now inspect the source and complete app stylesheet union and match standalone selectors rather than earlier scoped overrides. No stylesheet changed. |
| Two import/Undo journeys | Changing `app_instance` to the worker-owning harness dropped the other Library services, replacing the list with a legitimate unavailable state. The harness now carries those services. Real SQLite persistence, mutual exclusion, cancellation drain and subsequent Undo remain required. |
| Caret, conflict and delete readiness | A service-entry signal preceded the loading filter's rebuild, and conflict polling queried a temporarily absent action directly. Tests wait for the replacement filter to regain focus and for visible conflict actions. The stale request is released and observed returning before the final caret check; saved/source content assertions remain. A confirmed delete waits for its settled receipt, not a mutation flag that can still be idle before the queued confirmation is handled. |
| First-save UI readiness | Count/action queries raced targeted DOM rebuilding. Bounded waits now precede identity and action assertions; the original text-widget identity requirement exposed the production rebuild described above. |

## Verification

- Footer + entry baseline: **6 failed, 92 passed**. Four failures were the
  footer fake; the two entry failures were diagnosed separately above.
- Intermediate full Prompts run after the product fixes: **340 passed, 2
  failed**. The failures were the caret readiness race (cursor 10 instead of 4)
  and querying the conflict action during its rebuild. Both passed alone;
  explicit readiness conditions repair the race rather than relying on reruns.
- A subsequent run cleared those two cases and exposed a delete test accepting
  the pre-admission idle flag. That run stopped at **211 passed, 1 failed**;
  it is not a completed-file result. All six affected readiness/delete cases
  then passed with explicit postconditions.
- Final full Prompts file: **342 passed**.
- Seven-file neighboring run: **159 passed**, covering the complete entry,
  footer, Workspace, route residency, route storm, canvas-sync defects and
  Prompt browse-controller files.
- Separate first-save and real-app reusable-screen journey: **2 passed**.
  This overlaps the final Prompts file; counts are not all unique tests.
- Ruff formatting passes for every changed function. Complete-file diagnostic
  counts remain exactly at baseline (0/1/24 for the changed tests, 10/16/205 for
  the changed production modules), with no new diagnostic signatures. See
  `static.json`. Diff whitespace checks pass. No full repository suite ran.

```sh
.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -q --tb=short
.venv/bin/python -m pytest Tests/UI/test_library_entry_compose_once.py Tests/UI/test_library_footer_focus.py Tests/UI/test_post_release_workspaces_library_depth.py Tests/UI/test_library_phase_c_switch_residency.py Tests/UI/test_library_phase_c_switch_storm.py Tests/UI/test_library_canvas_sync_defects.py Tests/UI/test_library_prompt_browse_controller.py -q --tb=short
.venv/bin/python -m pytest Tests/UI/test_library_screen_reuse.py::test_library_reuse_and_suspend_timer_quiescence -q --tb=short
```

## Native evidence and limits

The real terminal driver (`LinuxDriver`, also used on macOS) reported 170×48 and used a copied audit profile, with all ten
configured database paths, `USER_DB_BASE_DIR`, XDG directories and `paths.data_dir`
inside private scratch. The final run asserted exclusive profile ownership.
Three same-route Notes refreshes retained `library-notes-tree-note-2` focus.
A first Prompt save retained its text widget and exposed the saved-state actions;
its rail count increased. A read-only SQLite check verified the exact saved
body, version 1 and active state. The screenshots were rendered and visually inspected.
The recorded native exit is **0**. No external model call or real user-data edit
was made. Route/refresh seams and widget actions were driven by an autopilot;
this is native rendering with real local services, not a fully manual keyboard
walkthrough. The Prompt screenshot qualifies the saved editor and count, not
completion of the separate Items page load.

The first probe waited on `screen.workers.wait_for_complete()`, which also
includes unrelated app jobs. It stalled and was terminated; a retry had warned
that this process still held the profile. Final verification requires exclusive
ownership and the measured terminal size, with output captured through tmux
without redirecting the app away from its terminal. Raw probe scripts, logs and
databases remain in ignored scratch.

The native app logs an unrelated Console sidebar startup error
(`_sidebar_state_save_timer` missing), optional audio warnings and unhandled
project-skills worker warnings. See `native-log-review.txt`; no clean-startup
claim is made. Pytest warnings concern cleanup of old temporary Kokoro folders.
The reusable-screen selection additionally reports a pre-existing
`datetime.utcnow()` deprecation in Media database cleanup.

## Scope and task status

ADR required: no. Existing ADR-031, ADR-086 and ADR-161 govern the unchanged
footer, adaptive ownership and component contracts. These are direct bug fixes
and verification repairs, without a new storage, service or interaction boundary.

TASK-32462 stays In Progress because its first four acceptance criteria require
passing on `dev`. Local verification does not constitute integration. No push,
merge or full-suite qualification is included. The broader component audit still
has remaining Prompt interactions, Library Skills/Collections/ingestion details and other app destinations to
review.
