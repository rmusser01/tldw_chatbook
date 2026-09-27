# Console live tool lifecycle verification

Date: 2026-09-27. Task: TASK-33095.
Decision: [ADR-195](../../../backlog/decisions/195-console-live-tool-call-presentation.md).

## Delivered behavior

Primary calls get stable Assistant-owned rows before permission review. Actual
pending approval rounds control the waiting label; runtime execution controls the
running timer and terminal outcome. Results appear individually, with at most
three wrapped preview rows and one Arguments/Result disclosure. Full-output and
diff actions remain available. New display facts remain session-only.

Review regressions cover reused provider IDs, production screen refresh after a
preamble, late diff attachment without remounting, discovery-tool capture
exclusions, model-shell takeover, and observer failures during approval/teardown.
Raw commands initiated by the user retain their standalone presentation.

## PR integration on current dev

The PR branch starts at `a7b6d5864bde6072b01eeb7a54a89e9bc977e1cb`.
The original checkout was substantially older; only this feature's files were
transplanted. Current dev's runtime arguments, tests, and component patterns were
preserved. New styles live in `features/_console_panels.tcss`, using current
sizing tokens. ADR-184 was already occupied on dev, so this decision is ADR-195.
The original dirty checkout was left unchanged (feature-file SHA-256s checked).

The final feature-focused run passed **185 tests**, covering every added or changed regression, including
real mouse expansion and Enter collapse in the mounted ChatScreen. Its report is
`/tmp/console-dev-33095-final.xml`. Additional dev-based runs exercised the
runtime, review hooks, approval controller, grouping, raw shell, and governance.
The generated stylesheet sync check and Backlog ID guard pass. The new module and
its tests pass Ruff lint/format; changed Python lines have no Ruff findings.
A separate read-only review found no new integration issues.

Broader dev checks are **not all green**. The first broad run was stopped after
29 repeated config-profile failures and 8 passes. An unmodified-dev module overlay
reproduced a representative `raw_source_selection_changed` failure before any
tool work. Changed config-reading regressions now use the repository's existing
`bootstrap_profile` marker, preserving the real recovery guard. Five legacy
Assistant-turn cases and 18 legacy approval cases failed again with unmodified
dev modules (23/23 reproduced). The service callback module also reproduces its
five failures on the unmodified-dev overlay. Two component governance failures
point exclusively to unchanged dev files: raw dimensions in `_agentic_terminal`,
`_settings_splash_theme`, and `_workflows`, and a Python width in
`library_skill_work_pane.py`; those files are byte-identical to the base. None is
claimed as a pass or silently fixed in this feature PR. The full suite was not run.

Dev-native walkthrough: `/tmp/console-dev-live-33095-run4` records actual loaded
ChatScreen/controller/bridge paths, queued and awaiting-approval states, the real
Deny all/Submit interaction, a running timer, succeeded/denied previews, mouse
expansion showing Arguments and the final result line, Enter collapse, and
100-column wrapping. The scripted run returned `done`; the terminal test exited
0 with no Textual exception, and its dedicated tmux server was closed. The probe
uses scripted provider/results and isolates unrelated sidebar subagent-count
reads; it does not qualify sidebar persistence or a real external provider.
Earlier dev-native attempts hit config selection or sidebar recovery admission
and are not counted as successful walkthroughs. The temporary probe was removed
from the worktree after retaining its source and captures under `/tmp`.

## PR review follow-up

Reproduced the shell findings before fixing them: four interrupted-takeover cases
and the mounted missing-metadata case failed (`/tmp/console-2861-review-red.txt`).
Run teardown now settles adopted shell rows, freezes elapsed time, preserves
partial output, and reports unknown exit/cleanup facts when no result exists.
An already received executor result wins over the enclosing run's cancellation.
All matching correlations are detached before best-effort display updates, so
late events and a failed update cannot strand another row's ownership.

The review exposed an additional ordering issue: an adoption display error could
skip registering the run for teardown. A real `run_reply` regression reproduced
both error and cancellation variants (two failures, two controls passed), then
passed after registration moved before projection. The final read-only review
also reran the original no-stream-event reproduction and found no remaining issue.

Expanded shell details share the canonical execution-metadata formatter and show
one output body. The mounted regression retains the same detail widget and focused
header through live output and terminal metadata updates. Argument truncation and
model validation now share one limit; public preview helpers document their inputs.
Warnings include a fixed operation description and exception class. Raw exception
text and tracebacks are intentionally excluded under ADR-029; sentinel regressions
verify tool arguments/output cannot leak through these diagnostics.

Verification: **197 feature-focused tests passed** in
`/tmp/console-2861-final.xml`; the final ordering fix then passed **15 targeted
shell/teardown tests** in `/tmp/console-2861-adoption-green.xml`. The three raw CLI
import-boundary tests passed in `/tmp/console-2861-extra.xml`. Counts overlap.
Edited Python ranges pass Ruff lint/format and `git diff --check`; unrelated
whole-file lint debt remains. This follow-up uses mounted UI and run integration
tests; the native walkthrough above qualifies the original feature behavior.
No full suite was run. ADR-195 applies; no new ADR is required.

CI's derived-artifact job identified the omitted diagnostic-inventory refresh.
The statement-level audit against the PR base found exactly three added warning
calls: two in the bridge and one in the approval controller. Each logs only a
fixed operation description plus `type(exc).__name__`; no user content, paths,
URLs, exception bodies, or new sink destinations are introduced. Regenerated
`Docs/security/production-diagnostic-inventory.json`; only those two owner rows
changed. The subsequent `--diff` check reports no drift (605 owners and 14 sink
files), recorded in `/tmp/console-2861-diagnostic-verified.txt`. The PR/UI Fast Lane
and the other initial-head CI checks passed; the new commit requires a fresh CI run.

## Original-checkout automated evidence

- Main targeted run: **630 passed**, two established baseline failures excluded.
  Files: runtime, runtime review hook, service step delivery, trace approvals,
  Console bridge, activity presentation/projection, raw-shell progress, Assistant
  turn widgets, token governance and generated CSS synchronization.
- Final boundary run: **30 passed** (338 unrelated cases deselected), covering
  actual controller approval waits, broken display observers, mounted screen
  refresh, shell takeover and wrapped previews.
- Complete turn-grouping module: **24 passed**.
- Final preview/token/CSS follow-up after adding the resize height cap:
  **19 passed** (41 unrelated cases deselected).
- Additional targeted regressions: display cleanup cannot fail a run, and
  completing a tool cannot pull a reader away from scrolled history: **1 passed
  each**. Counts above overlap; they are not a unique-test total.
- New projection module and its tests pass Ruff lint and format checks. Changed
  Python lines have no Ruff findings; scoped `git diff --check` passes. Existing
  whole-file lint debt was not reformatted or repaired.

The broader native-transcript check produced 32 failures in existing styling,
action menus, focus traversal and stylesheet assertions. Those same cases,
`test_fleet_teardown_pop_is_identity_checked_not_blind`, and
`test_generic_tool_credentials_are_scrubbed_at_durable_agent_step_boundary`
all failed again with the affected Python modules loaded from unmodified HEAD
copies in `/tmp`. All **34** are baseline failures, not claimed as passes. The
full repository suite was not run.

## Disposable terminal walkthrough

Mounted the real ChatScreen in its existing Console harness under pytest's
isolated HOME/config/data fixtures, using the production stylesheet set. Ran it
in a dedicated tmux server at 140×42, then 100×32. The provider and file results
were scripted; no external model or real file tool was invoked.

Observed two same-name calls in one Assistant turn: one queued and one awaiting
approval. Clicked the rendered **Deny all**, then **Submit**. The admitted call
showed **Running · 1.0s**, then **Succeeded · 6.0s**; the other showed **Denied by
you**. The final assistant response arrived and the run returned `done`.

Clicked the successful row to reveal Arguments (`README.md`) and the result's
final line. Enter collapsed it. After narrowing the terminal, the settled
previews showed omission hints within three rows. A transient resize frame
motivated the token-backed maximum height; its mounted/token/CSS follow-up passed.

An earlier probe seeded an already-complete assistant message and consequently
rejected the final response chunk. The corrected probe creates a streaming
placeholder and asserts the run's final outcome. Another early harness attempt
was missing the runtime-owned bridge assignment. Neither is counted as a
successful full-run verification.

The final isolated walkthrough passed and its dedicated tmux server was closed.
Local diagnostic captures remain under `/tmp/console-live-33095`; automated run
logs are `/tmp/console-lifecycle-final.txt`, `/tmp/console-boundary-final.txt`, and
`/tmp/console-preview-cap-final.txt`. These are disposable evidence paths.

Partial output streaming for ordinary tools remains v2. Historical arguments or
timing are not fabricated, and this work does not broaden durable capture.
