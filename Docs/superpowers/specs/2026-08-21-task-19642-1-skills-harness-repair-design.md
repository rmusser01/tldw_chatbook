# TASK-19642.1 Skills Harness Repair Design

Date: 2026-08-21

Status: Approved direction; pending written-spec review

Task: TASK-19642.1

## Goal

Restore the 29 assigned Skills Library/import integration tests without
weakening their real service, trust-store, or runtime-policy coverage and
without reverting accepted Library or Skill-editor behavior.

## Evidence and root causes

The exact two-file baseline is 29 failed and 5 passed. The failures fall into
three test-contract drifts:

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

The focused production owner checks for these lifecycle states pass. The
repair therefore belongs in the integration harness/tests, not production.

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
- assertions continue to verify the durable service result, trust posture,
  list membership/count, and real runtime-policy path.

The 29 assigned nodes are the regression coverage for this repair. No new test
module, fixture framework, or production-only seam is added unless an
implementation-time RED run exposes a contract not represented by those nodes.

## Alternatives rejected

- Re-expose full Library navigation to new empty profiles: contradicts accepted
  ADR-076 and changes production to satisfy a stale fixture.
- Restore Save/Delete as unconditional primary actions: contradicts TASK-19025's
  accepted lifecycle and its passing owner tests.
- Make the shared app factory default to an existing profile: broadens the
  change to unrelated tests and would make onboarding coverage easier to bypass
  accidentally.
- Ignore Study/Quiz exceptions because production catches them: leaves hidden
  background noise that can obscure future owner failures and violates this
  task's deterministic-fixture acceptance criterion.

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
4. Run Ruff on the two modified test files and `git diff --check`. The current
   baseline has two small Ruff findings in `test_skills_library_flow.py`
   (one unused local import and one semicolon-separated pause); because that
   file is already in scope, remove those two findings. Both large files also
   have pre-existing whole-file Ruff-format drift, so do not bulk-reformat or
   claim a green whole-file formatter check; keep every changed hunk formatted
   consistently and record the baseline explicitly.
5. Perform inverse evidence by temporarily restoring one stale contract at a
   time (fresh-profile factory, direct clean Delete, or old create-save copy),
   confirm its named focused test fails, and restore the repair.

No repository-wide pytest claim will be made.

## ADR check

ADR required: no new ADR

ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

Reason: this is a test-only realignment with the lifecycle and Skill-editor
contracts already accepted by ADR-076 and TASK-19025. It changes no storage,
security, service, runtime, or user-facing architecture.
