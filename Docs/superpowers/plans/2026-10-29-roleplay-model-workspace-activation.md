# Roleplay model selection activation implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement this
> single task inline, then request an independent whole-branch review.

**Goal:** Keep character/persona conversations selected after model-picker return.

**Architecture:** Align the workspace registry at roleplay activation using the
existing workspace helper. Wire a required named callback into the character
controller; reuse the session controller's existing callback for handoffs.

**Tech Stack:** Python 3.12+, Textual 8, SQLite, pytest.

**Spec:** `Docs/superpowers/specs/2026-10-29-roleplay-model-workspace-activation-design.md`

**Task:** TASK-37620.

ADR required: no
ADR path: `backlog/decisions/033-application-session-state-ownership.md` and
`backlog/decisions/095-conversation-owned-console-generation-settings.md`
Reason: repair existing ownership; no new storage, scope or reconciliation policy.

## Global constraints

- Preserve session scope, transcript and assistant identity.
- Preserve pristine-tab reuse and deliberate cross-screen workspace changes.
- No provider traffic or new dependencies; targeted verification only.
- Constructor dependencies remain explicit named callables.

## Review focus

- New character picker conversations align global scope before UI publication.
- Worked character/persona handoffs align their new global conversations.
- Pristine character/persona handoffs retain their named workspace and tab.
- Model-picker Apply/Cancel cannot select another conversation.
- Genuine cross-screen workspace activation still reconciles on resume.

## Task 1: Repair roleplay activation

**Files:** `UI/Console_Modules/{character,session,wiring}.py` under
`tldw_chatbook/`; affected character-controller test builders; focused roleplay
model/workspace tests; `Docs/Architecture/console.md`; Console wiring lessons.

**Interfaces:** Consume the existing `(session_id: str) -> None` workspace
alignment helper. Add the required `set_active_workspace_for_session` constructor
callback to `ConsoleCharacterController`; introduce no new activation owner.

- [x] Write regressions that assert preserved chat/registry ownership after the
  real roleplay activation and model-picker return. Watch them fail on `dev`.
- [x] Call the existing alignment seam for picker creation and character/persona
  handoffs. Update every direct constructor test builder.
- [x] Run the new regressions plus affected controller/avatar/session, workspace
  reconcile, model apply and size-governance tests; run Ruff and formatter checks.
- [x] Document the activation invariant and measured incident. Update TASK-37620
  acceptance criteria and implementation notes, self-review, and commit.
- [ ] Obtain independent review, create a PR against `dev`, address all feedback,
  and rebase with lease protection if needed. Require current-head Qodo review
  and resolved threads before arming the enabled merge queue; verify merge.

## Evidence and rulings

- Initial baseline: workspace reconcile 9 passed; the four unmarked no-mount
  character-controller cases stop in the UI fixture with
  `raw_source_selection_changed`. Use the existing bootstrap-profile test
  lifetime convention when updating those constructor fixtures; do not relax
  production storage admission.
- The preceding investigation reproduced the jump after real character creation
  in an asserted active named workspace. A control using the existing alignment
  helper preserved the originating chat through model Apply.
- RED: the installed mouse-driven Apply cases for picker, character handoff and
  persona handoff each failed at `store.active_session_id == origin.id` before
  production changes. The origin had the chosen model while another chat became
  active. The case owns runtime/DB teardown through the existing fixture.
- Ruling: preserve existing `store.switch_session` calls because they notify
  runtime/Canvas observers and advance activation epochs. Updating the stale
  session/workspace module documentation accurately describes the repaired seam
  and reduces module size without raising the ratchet or compressing code.
- GREEN: all eight new regressions passed together after the repair, including
  six real mouse-driven Apply/Cancel paths and both pristine handoffs. The
  focused run passed 40 cases and exposed one remaining persona visual fixture
  missing `active_session_epoch`; after replacing that fixture with a real store,
  the complete persona visual file plus isolated avatar sync passed (12 cases).
- The sibling model-apply, character/persona session and ordinary activation
  selection passed 44 cases. Workspace reconciliation, character controller and
  avatar copy-budget cases passed in the focused run. Both session-size guards
  passed at 6,990 lines. Independent whole-diff review approved the final change.
- Ruff passes for changed Python files with only pre-existing E721 exact-type
  checks excluded; both substantially edited test files pass the formatter.
  The UI census (1,152 entries), Backlog ID/file guards and `git diff --check`
  pass. Targeted tests only; no full-suite claim.
- Diagnostic limits: two bare-screen architecture tests fail on unchanged `dev`
  as well, with stale `_library_activity` stubs missing `_session`. A broader
  avatar run encountered an order/load-sensitive portrait assertion followed by
  native-resource admission errors; that portrait case passes in isolation on
  both base and branch. The repaired roleplay scenarios use owned lifetimes and
  private-profile subprocesses and pass together.
