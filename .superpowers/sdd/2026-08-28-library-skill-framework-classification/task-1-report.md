# TASK-22867 implementation report

## Outcome

Library skill import now classifies root skills, repositories containing multiple
independent skills, valid framework repositories, malformed/unsupported packages,
and remote fetch/access failures before importing. Multiple-skill repositories pause
inside TASK-613's one app-owned operation for an explicit candidate choice; the
selected import uses the exact retained bytes and SHA-256 once, never refetches the
reviewed branch, and still lands trust-pending.

The task remains **In Progress** and all acceptance criteria remain unchecked for
independent review, as required. No new ADR was needed: ADR-009's local trust boundary,
ADR-069's copy-import posture, and TASK-613's coordinator remain authoritative.

## RED/GREEN evidence

- Classifier RED: the new classifier module did not exist, so collection failed.
  GREEN: 7 tests passed for root parity, stable/deduplicated candidates, framework vs
  direct-archive behavior, corrupt/empty input, unsafe paths, and symlink rejection.
- Retained archive RED: the remote inspect/import APIs did not exist (3 failures).
  GREEN: the focused retained-package selection passed 10 tests, with one
  `httpx.MockTransport` request, exact candidate import, and
  `trust_approved=False`.
- Candidate coordinator RED: selected-candidate admission did not exist; a subsequent
  stale-callback probe showed an old modal could target a later package. GREEN:
  synchronous candidate claim, generation fencing, cancellation-resistant settlement,
  exact local subdirectory selection, and cleanup passed 4 focused tests.
- UI RED: the choice modal module did not exist. GREEN: mounted normal (120×36) and
  compact (72×22) selection plus Cancel passed 3 tests; framework and failed/Retry
  production-shaped states passed 2 size cases.
- End-to-end GREEN: a mounted real Library screen inspected a two-skill local
  repository, presented the real modal, imported only `zeta`, and showed it in the
  real trust-pending service context while `alpha` remained absent.
- Privacy refinement RED: the safe snapshot exposed URL userinfo, and the retained
  package repr exposed a URL-derived name. GREEN: UI state strips credentials, query,
  and fragment; Retry keeps the raw URL private; package repr exposes neither source
  name nor archive bytes.
- A broader run exposed one test-driver timing race: a programmatic OptionList
  highlight had not settled before the button press. Waiting one Textual event-loop
  turn made the production-shaped choice deterministic. A separate UI compatibility
  run found two old fakes modeling retired screen-owned receipts; both now assert the
  real TASK-613 coordinator snapshot.

## Implementation

- Added one bounded, side-effect-free classifier shared by local directory and zip
  central-directory inspection. It accepts exact non-symlink `SKILL.md`, strips one
  archive wrapper, applies stable path ordering and the 20-candidate display cap, and
  performs no import, extraction, execution, activation, or trust mutation.
- Split remote download/inspection from import. The existing URL classification,
  runtime-policy admission, DNS/public-address validation, manual redirect checks,
  credential-origin rules, deadline, compressed-size cap, bounded re-rooting, and
  import seam remain in place.
- Extended the existing app coordinator with private retained-package ownership,
  candidate/cancel/retry claims, and safe snapshot fields. There is still one
  coordinator and one app worker group; both initial inspection and selected import
  use the incumbent cancellation-resistant terminal owner.
- Added one modal with a bounded single-select OptionList and explicit **Import skill**
  / **Cancel** actions. The Library row now distinguishes **Import skill…** from media
  ingestion and renders inspecting, choice, framework recovery, malformed, failed /
  **Retry**, success, and trust-review states at normal and compact sizes.
- Framework presentation uses the exact generic copy and only the four approved
  recovery actions. No framework, vendor, threat-hunting workflow, command, or
  briefing handoff is special-cased.

## Security and lifecycle audit

- Every successful local directory, local zip, and remote selected import reaches the
  existing scope service with exactly `trust_approved=False`.
- Remote selection imports only an allowed inspected candidate from bytes whose
  SHA-256 still matches. Candidate Cancel and every selected terminal path release the
  retained archive.
- Framework and malformed outcomes import nothing. Classification never executes or
  imports repository code.
- Display snapshots and custom reprs contain no URL userinfo/query/fragment,
  response body, raw exception text, archive bytes, or URL-derived suggested name.
  Remote failure presentation is fixed generic copy.
- The delayed inspection, delayed candidate import, forced second submissions,
  repeated outer cancellation, navigation, routed replacement, Cancel, and stale modal
  callback cases preserve the one in-flight contract.
- A scoped source scan found no ATHF, vendor, threat-hunting, or repository-specific
  naming in production, tests, or user guidance.

## Verification

- Exact task-plan target: **121 passed**, one inherited
  `RequestsDependencyWarning`, in 65.72s.
- Additional focused Skills canvas compatibility: **9 passed, 136 deselected**, one
  inherited warning.
- Ruff over every touched Python production/test file: `All checks passed!`.
- Python compilation and `git diff --check`: passed.
- Impeccable detector was run exactly once over the touched Library UI files and
  returned no findings (empty output, exit 0).
- No full suite, live network, live user skill store, CSS build, push, or merge was
  used.

## Files

- `tldw_chatbook/Skills_Interop/skill_package_inspection.py`
- `tldw_chatbook/Skills_Interop/skill_remote_fetch.py`
- `tldw_chatbook/UI/Library_Modules/library_skill_import_controller.py`
- `tldw_chatbook/UI/Library_Modules/skill_import_choice_modal.py`
- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/Widgets/Library/library_skills_canvas.py`
- `Tests/Skills/test_skill_package_inspection.py`
- `Tests/Skills/test_skill_remote_fetch.py`
- `Tests/Skills/test_skill_import_choice_modal.py`
- `Tests/Skills/test_skills_import.py`
- `Tests/UI/test_library_skills_canvas.py`
- `Docs/User_Guide/library/skills.md`
- `backlog/docs/lessons-testing-evidence.md`
- `backlog/tasks/task-22867 - Classify-framework-repositories-during-Library-skill-import.md`
