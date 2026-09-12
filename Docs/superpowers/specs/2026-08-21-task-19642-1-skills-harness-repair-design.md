# TASK-19642.1 Skills Harness Repair Design

Date: 2026-08-21

Status: Approved design; production ordering amendment approved and current

Task: TASK-19642.1

## Goal

Restore the 29 assigned Skills Library/import integration tests without
weakening their real service, trust-store, or runtime-policy coverage and
without reverting accepted Library or Skill-editor behavior.

## Evidence and root causes

The exact two-file baseline is 29 failed and 5 passed. The initial failures and
the two latent assertions exposed after repairing the rail fall into four
test-contract drifts. Independent closeout and quality verification then
exposed two production races:

1. Both Skills files import the shared `TldwCli` test factory directly. That
   factory truthfully admits a newly created test profile to the Library
   `unknown`/`starter` lifecycle. These tests require the full legacy Library
   rail, but `_wait_for_library_shell` only establishes that the shell and rail
   mounted; it does not promise that onboarding evidence has graduated or
   expanded the rail. Direct failure inspection observed
   `library_new_profile_admission == True`, lifecycle `unknown`, and no Skills
   row. The Library test module already provides a wrapper that explicitly
   creates the existing-profile/full-rail posture for tests with this contract.
2. The shared Skills fixture wires Notes, Media, and Conversations but leaves
   real Prompt, Study, and Quiz scope services on the app. Their optional
   decorative count calls can report unavailable backends. Production
   intentionally catches those errors and degrades the badges to uncounted, so
   the exceptions are log noise rather than the missing-row cause; the Skills
   fixture should make its non-owner seams explicitly inert.
3. TASK-19025 intentionally simplified the Skill editor. A clean saved Skill
   now exposes Back and More actions, Save appears only after a draft changes,
   Delete is revealed under More actions, and the first create-save reports that
   trust review is required. Several older end-to-end tests still press hidden
   or inactive controls and assert the superseded generic `Saved.` copy.
4. The first complete two-file gate after the planned changes exposed two
   additional stale assertions that had been hidden behind the earlier
   full-rail failure. A newly trusted clean Skill now collapses healthy trust
   actions behind `View details`, and allowed tools now live in the advanced
   editor as a `SelectionList` rather than a basic-mode `Input`. Both failures
   reproduce in isolation and their direct production owner tests pass.
5. A later clean-HEAD two-file run intermittently timed out waiting for the
   orphaned-manifest trust header. The list, import action, and blocked Skill
   row were already visible, while `_sync_library_canvas("skills")` logged a
   swallowed failure. Five subsequent instrumented ordered runs did not
   reproduce the exception, but history and lifecycle tracing identified a
   matching race window: rail selection starts the off-thread trust-posture
   worker before the Skills canvas is mounted. If the worker projects after
   compose captured the
   old posture but before the replacement canvas mounts, strict targeted sync
   has no owner and drops `FAILED`, leaving the newly mounted canvas stale.
6. The first no-service repair cleared the cached posture and repainted the
   mounted list, but it did not supersede an earlier trust-posture worker. A
   deterministic gated overlap removed the trust service while that worker's
   captured callable was in flight and observed projections `""`, then stale
   `"ready"`; the old worker still matched the current route and generation,
   so it restored the header after the no-service refresh cleared it.

The first four findings belong in the integration harness/tests. The final two
are production ownership defects at the existing mounted-owner/worker-group
boundary and need narrow runtime fixes; neither is safe to hide with a longer
test timeout or an integration-test refresh.

## Design

### Full-Library test posture

Both Skills integration files will build apps through the existing local
wrapper in `Tests/UI/test_library_shell.py`, which delegates to the shared app
factory and then sets `library_new_profile_admission = False`. This expresses
the tests' existing-user/full-rail prerequisite before any `LibraryScreen` is
constructed. It does not modify the shared factory or onboarding tests.

The first import test will retain `configured_default="library"`; the wrapper
already forwards that argument.

### Inert non-Skills owners

`_wire_empty_non_skill_services` will continue to install empty production-
shaped Notes, Media, and Conversation fakes. It will also replace Prompt,
Study, and Quiz scope services with objects that expose no optional count seam.
This uses the production-supported "service unwired" path: the Library renders
those decorative rows without counts and starts no failing count call.

The Skills service and, where applicable, the real trust service and real
`ServicePolicyEnforcer` remain untouched. No trust bypass or fake policy
enforcer is introduced.

The inert Prompt/Study/Quiz posture needs no new fake type: assigning plain
objects to the three optional service slots exercises the already-supported
missing-seam branch with the smallest possible fixture change.

### Observable interaction helpers

The Skills editor/import helpers will reuse the existing bounded
`_wait_for_selector`/state helpers to wait for the next externally observable
state before acting:

- the full Library rail must contain the Skills row;
- selecting Browse Skills must produce the requested skill row or Import
  control;
- selecting a skill row must mount the editor controls and settle its detail;
- dirty/save/delete transitions must be observed through their screen state or
  mounted action controls, rather than a fixed pair of event-loop pauses.

Waits remain bounded and fail with the missing selector/state named in the
message. They do not sleep for arbitrary wall-clock durations. This is not a
file-wide pause cleanup or a new polling abstraction: stable pauses outside the
29 failing scenarios remain untouched.

### Current editor lifecycle

Older scenarios will be updated to drive the accepted lifecycle:

- the trusted-skill save test makes a real edit before saving, preserving its
  purpose of proving that a save re-queues trust review;
- clean deletion scenarios reveal More actions and wait for the existing
  Delete control before pressing it;
- create-save scenarios expect
  `Saved. Review trust before using this Skill with the agent.`;
- the trust-bootstrap scenario opens healthy trust details before asserting
  the normal actions, and the blank-create scenario switches to Advanced
  before verifying its empty allowed-tools selection;
- assertions continue to verify the durable service result, trust posture,
  list membership/count, and real runtime-policy path.

The 29 assigned nodes are the regression coverage for this repair. No new test
module, fixture framework, or production-only seam is added unless an
implementation-time RED run exposes a contract not represented by those nodes.

### Mounted-owner ordering repair

`_select_library_rail_row_after_source_admission` will start the Skills
trust-posture refresh only after its targeted route replacement or whole-screen
recompose has completed and `#library-skills-canvas` owns the destination.
The refresh remains limited to Browse Skills list mode.

`_load_library_skills_trust_posture` keeps its current route/generation guards,
focus handoff, and `allow_screen_fallback=False` targeted-sync contract. No
retry loop, arbitrary delay, swallowed-error suppression, or broad screen
recompose is added. Service exceptions continue to degrade the posture to the
existing empty state.

A deterministic owner test will select Browse Skills through the real rail
route while spying on the refresh boundary. It must observe that the Skills
canvas is mounted when refresh begins. The existing orphaned-manifest flow will
continue to prove that a real `needs_resetup` posture renders the header and
action.

When no callable trust service is available, the refresh will always clear the
posture. Before clearing, it will cancel the existing
`library_skills_trust_posture` worker group, using the same Textual group that
an ordinary exclusive replacement already supersedes. The async worker awaits
its captured callable through `asyncio.to_thread`, so cancelling its task
prevents that coroutine from resuming into posture publication even though the
underlying thread may finish. The no-service refresh will repaint through the
same strict targeted sync with `allow_screen_fallback=False` only while the
Browse Skills list owns the mounted canvas and either cached posture or a
mounted trust header is stale. An already-empty list with no header still
cancels stale work and clears state, but skips the unnecessary canvas
recompose. This prevents a posture from the prior visit or an earlier in-flight
read remaining visible without recomposing an open editor or already-clear
list from stale screen state, or changing any route, generation, focus, trust,
or runtime-policy contract.

## Alternatives rejected

- Re-expose full Library navigation to new empty profiles: contradicts accepted
  ADR-076 and changes production to satisfy a stale fixture.
- Restore Save/Delete as unconditional primary actions: contradicts TASK-19025's
  accepted lifecycle and its passing owner tests.
- Make the shared app factory default to an existing profile: broadens the
  change to unrelated tests and would make onboarding coverage easier to bypass
  accidentally.
- Ignore Prompt/Study/Quiz exceptions because production catches them: leaves hidden
  background noise that can obscure future owner failures and violates this
  task's deterministic-fixture acceptance criterion.
- Retry a failed posture projection: adds another reconciliation state machine
  even though the rail-entry caller can satisfy the existing owner precondition.
- Add a separate posture request token: the existing Textual worker group
  already provides the required async-task supersession, and the gated overlap
  verifies that it blocks stale publication.
- Enable `_sync_library_canvas`'s whole-screen fallback: violates the accepted
  compose-once/focus-ownership contract for automatic entry workers and masks
  the ordering defect instead of removing it.

## Verification

Only tests related to the modified harness/functionality will run:

1. RED evidence from the current exact two-file gate and representative
   isolated failures is retained in TASK-19642.1 notes.
2. Run the 29 assigned nodes through the two files in their inventory order.
3. Run these directly related Skill editor owner nodes, which pin the accepted
   lifecycle the integration tests are being aligned to:
   - `test_skill_editor_clean_saved_mode_renders_navigation_actions_only`
   - `test_skill_editor_lifecycle_exposes_only_valid_primary_actions`
   - `test_handle_library_skill_delete_enters_confirm_state`
   - `test_create_save_success_consumes_scroll_receipt_after_recompose`
   - `test_mark_dirty_clears_stale_saved_status`
   - `test_skill_editor_healthy_trust_is_compact_until_details_are_requested`
   - `test_skill_editor_advanced_tool_picker_is_bounded_unique_and_lossless`
   - `test_library_skill_mode_switch_is_targeted_and_remembered`
4. Add and run deterministic owners for the mounted trust-posture lifecycle:
   - `test_skills_rail_starts_trust_posture_after_canvas_mount` proves refresh
     begins only after `#library-skills-canvas` mounts;
   - `test_skills_rail_without_trust_service_clears_mounted_header` proves the
     mounted list removes a stale trust header when the service disappears;
   - `test_missing_trust_service_supersedes_in_flight_posture_worker` proves a
     no-service refresh cancels an earlier gated posture worker before it can
     republish stale state;
   - `test_missing_trust_service_already_clear_list_skips_repaint` proves a
     repeated no-service refresh still cancels the posture group and clears
     state without syncing an already-empty list that has no header;
   - `test_missing_trust_service_snapshot_preserves_open_skill_draft` proves a
     background snapshot clears cached posture without syncing the editor or
     losing its live draft, widget identity, or focus.
   Re-run the directly related compose-once, stale-route, stale-generation,
   callback-composition, and editor-lifecycle owners. This focused
   ordering/reconciliation gate contains 11 cases. The 12 base editor-owner
   cases plus the two no-service lifecycle regressions form a 14-case
   editor/trust lifecycle gate.
5. Run the exact two-file gate three consecutive times from clean HEAD, then
   repeat the same focused gates after rebasing on current `origin/dev`.
6. Run Ruff on all modified Python files and a revision-range
   `git diff --check`. The current
   baseline has two small Ruff findings in `test_skills_library_flow.py`
   (one unused local import and one semicolon-separated pause); because that
   file is already in scope, remove those two findings. All five modified
   Python files have pre-existing whole-file Ruff-format drift, so do not
   bulk-reformat or claim a green whole-file formatter check; keep every changed
   hunk formatted consistently and record the baseline explicitly.
7. Perform inverse evidence by temporarily restoring one stale contract at a
   time (fresh-profile factory, direct clean Delete, or old create-save copy),
   confirm its named focused test fails, and restore the repair. Temporarily
   moving the posture refresh back before canvas mounting must also fail the
   mounted-owner test. Restoring the unconditional no-service canvas sync must
   fail the editor snapshot regression by observing a sync/recompose against
   the live draft. Restoring an unconditional no-service list repaint must fail
   the already-clear list regression with one sync. Removing the no-service
   worker-group cancellation must fail the gated overlap with projection
   `["ready"]`; restore it and require the old worker to remain cancelled with
   no projection.

No repository-wide pytest claim will be made.

## ADR check

ADR required: no new ADR

ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

Reason: ADR-076 and TASK-19025 already own the Library and Skill-editor
contracts, and the entry-reconciliation subsystem already requires automatic
workers to project only into their mounted owner without a whole-screen
fallback. Moving one existing refresh trigger across that mount boundary is a
routine sequencing fix within those accepted contracts. It changes no storage,
security, service interface, or user-facing architecture.
