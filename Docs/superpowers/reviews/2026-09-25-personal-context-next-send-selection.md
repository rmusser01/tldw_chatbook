# Personal Context Next Send selection execution review

Date: 2026-09-25
Task: TASK-25907.3
Status: Complete after two reviewer-confirmed freshness fixes.

Plan: [Next Send selection implementation](../plans/2026-09-25-personal-context-next-send-selection.md)
Design: [Memory evolution, section C](../specs/2026-09-25-personal-context-memory-evolution-design.md#c-explain-selection-in-next-send)
ADRs: [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md), [ADR-088](../../../backlog/decisions/088-console-lightweight-next-send-history-projection.md), [ADR-150](../../../backlog/decisions/150-design-token-system-and-design-language.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Delivered behavior

The Next Send inspector explains which agent-eligible Personal Context records
were selected, replaced by a workspace override, or omitted by a whole-record
byte or token budget. It shows priority groups and content-free unavailable,
disabled, locked, empty and insufficient-budget states. The service derives the
explanation while building the actual preview block in one traversal; it does
not run a second selection to explain a possibly different request.

The explanation is an inspector-only, repr-suppressed sidecar. Ordinary context
snapshots, the prepared model request, chat persistence, exports, raw JSON,
logs and Sync do not receive it. It includes only already-authorized eligible
candidates. Private, inaccessible, expired and conflicted records, and
unscoped quarantine inventory, yield no new identifier or count hint.

Publication checks the captured session, workspace, draft, attachments,
staged sources, turn configuration, resolved provider/model, request budget,
profile identity/revisions and eligibility time. Both screen inputs and the
app-owned profile service are checked again after asynchronous validation.
Refresh clears old details before work starts; generation, visibility and a
one-shot expiry fence prevent late results from repopulating them. The panel is
collapsed by default, renders imported identifiers literally and boundedly,
and remains keyboard-scrollable at 60 columns.

## Verification evidence

The final affected-area run passed **173 tests in 88.70 seconds**. It selected
`Tests/Personal_Context/test_context_service.py`, `test_service.py`,
`Tests/Agents/test_personal_context_prompt.py`,
`Tests/Chat/test_console_personal_context_snapshot.py`,
`Tests/UI/test_console_personal_context_selection_factory.py`,
`test_console_conversation_inspector.py`, `test_design_token_governance.py`,
and `test_css_bundle_sync_guard.py`. Tests used synthetic profile data and a
fresh temporary root. This was a targeted run under the repository restriction,
not a full suite.

The service tests compare explained and ordinary snapshots for identical
inputs and clock, including Unicode byte/token ceilings, zero/header-only
budgets, a later small record after an oversized one, workspace overrides,
typed authority states, profile replacement, revision changes and expiry.
Chat tests exercise the actual preview and prepared-request entry points and
check that diagnostics do not enter provider payloads or generic serialization.
Mounted inspector tests cover Refresh, suspension/dismissal, stale workers,
expiry without a revision change, literal labels, exports and narrow keyboard
access. The two final gated regressions failed before their fixes and passed
afterward; the 173-test run includes them.

Production stylesheet hosts were inspected at 120 and 60 columns using
synthetic data. The initial narrow pane clipped explanation rows; replacing
the fixed-height inner payload pane with a scrollable body and a local wrap
rule restored access. Generated CSS passed the bundle sync guard. Focused
Python files passed formatting, and a Ruff check against changed lines found
no new finding. Whole-file Ruff remains noisy in long-standing bridge,
controller, screen and inspector code; no whole-file lint-clean claim is made.
`git diff --check` passed.

A broadened `Tests/UI/test_console_modal_dismissal.py` run had six failures in
untouched modal inventory, Settings and ConsoleModelPopover contracts. Its
inspector-specific dismissal contract passed. Those unrelated failures are
not counted in the 173-test affected-area result.

## Independent review and fixes

One fresh read-only reviewer inspected the full implementation diff against
the plan, design, privacy and ownership requirements. It found no Critical
issue and two Important freshness races:

1. Screen inputs were checked only before an awaited controller validation.
   A draft or owner change during that await could publish old diagnostics.
2. The controller checked the app-owned service before off-thread validation.
   Replacing the service while the worker ran could publish under a retired
   owner.

Both were reproduced with gated RED tests. The screen now checks its captured
inputs after the await; the controller reacquires and compares the service
after worker completion, immediately before returning success. Focused GREEN
and the final integrated run passed. No second profile selection was added.
The reviewer found no Minor issue. Its verdict was ready after these fixes.

The reviewer declined to judge existing `unsupported_records_present`
model-block disclosure and `device_only` provider-disclosure semantics. Both
remain documented under ADR-182 and the roadmap; this slice does not copy the
unscoped flag into diagnostics or claim to repair either existing contract.

## Native execution decisions

1. Run targeted tests only, as required by the repository and user. Cost if
   wrong: regressions outside the affected area remain unmeasured.
2. Make the inner payload body scrollable and wrap long diagnostic rows after
   the approved panel clipped at 60 columns. Cost if wrong: the inspector has
   one more scroll surface; mounted keyboard and production-CSS checks cover it.
3. Move the inspector's final owner and expiry validation after awaited
   project-instruction state. Cost if wrong: another await could reopen a
   publication window; the final boundary and gated regression cover it.
4. Reject duplicate diagnostic callbacks instead of choosing the last one.
   Cost if wrong: an unexpected planner shape displays no explanation even
   when a snapshot exists; this fails closed and leaves the snapshot usable.
5. Exclude six unrelated broadened modal failures from the affected test
   receipt. Cost if wrong: a baseline regression elsewhere could remain;
   this task does not certify those surfaces.
6. Recheck screen inputs and app service at the final async publication
   boundary after the reviewer's probes. Cost if wrong: the explanation may
   disappear when ownership is uncertain, while the ordinary snapshot remains.
7. Preserve the reviewer's provider-disclosure exclusions as explicit
   ADR-182 follow-up rather than asserting a broader privacy repair. Cost if
   wrong: existing disclosure behavior still requires TASK-25907.7.
8. Complete tracker and document closeout locally after the one fresh
   implementation review. Cost if wrong: the documentation has no separate
   reviewer; scoped link, task-family and whitespace checks provide evidence.

## Acceptance and closeout

| Criterion | Evidence |
| --- | --- |
| 1: Same-pass selected/override/budget reasons | Service parity and mounted explanation tests |
| 2: Existing budgets, order and root-run semantics | Context packing and prepared-request regression tests |
| 3: Eligible-only privacy and quarantine separation | Synthetic hidden-record and typed-state tests; payload review |
| 4: Disposable, separate diagnostic path | Repr, serializer, export, refresh and late-worker tests |
| 5: Captured owner, request, budget and expiry | Controller/factory fences and gated post-await regressions |
| 6: Honest unavailable states | Typed authority and empty/unknown failure cases |
| 7: Integration and narrow layout | 173 affected tests, production-CSS inspection and reviewer |

Ten task-family files and all 59 child acceptance texts were checked against
the planning base; dependencies point backward. Local Markdown links and
whitespace were checked at closeout. No new ADR was needed because ADR-182,
ADR-088 and ADR-150 already define the relevant boundaries. No general lesson
was added: the production-stylesheet geometry issue is already covered by
[the existing testing-evidence lesson](../../../backlog/docs/lessons-testing-evidence.md#a-geometry-harness-must-mount-the-production-hierarchy-and-stylesheet).

Implementation commits: `9b4c93a16e` (service), `72632de419` (preview
transport), `bca54dc58a` (inspector), `6592129b59` (review fixes). The branch
remains local; no push, PR or merge was requested.
