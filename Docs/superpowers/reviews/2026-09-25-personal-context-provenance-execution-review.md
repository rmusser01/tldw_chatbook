# Personal Context provenance execution review

Date: 2026-09-25
Task: TASK-25907.2
Status: Complete. All seven task criteria verified; independent review found no Critical or Important issue. One Minor recovery issue is deferred below.

Plan: [Provenance implementation](../plans/2026-09-25-personal-context-provenance.md)
Design: [Memory evolution, section B](../specs/2026-09-25-personal-context-memory-evolution-design.md#b-inspect-existing-provenance)
ADRs: [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md), [ADR-150](../../../backlog/decisions/150-design-token-system-and-design-language.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Delivered behavior

My Profile and pending proposal review have an expandable Recorded provenance
section. It describes authenticated current metadata: recorded source, actor,
reason, timestamps, state and retained identifiers. Source references and hashes
are separate, unverified metadata. Missing editing/inference history stays
unknown. Approval does not establish authorship, unchanged acceptance, source
support or durable confidence.

Deleted records have a separate metadata-only list. Selecting a tombstone does
not recover its old title, payload, Undo or historical bodies and cannot enter
record mutation actions. User-owned Settings may inspect private records; no
agent tool, context field, export, source resolver or canonical schema was added.

Inspection uses immutable service projections and captured profile identity,
purge generation and selected-object tokens. Pending proposals have no revision
field, so an internal canonical fingerprint detects a changed envelope; it is
neither displayed nor persisted. The UI checks selection/request generations,
owner replacement, screen visibility and read age before publishing. It clears
known local invalidations immediately and checks external changes once per
second while expanded on the current screen. A two-second lease from read start
clears metadata if renewal stalls. This is bounded refresh, not instantaneous
cross-process notification or a permanent maintenance job.

Proposal input and acceptance workers retain their existing behavior. Only
inspection is disposable. Metadata is literal, strips terminal/directional
controls, clips values explicitly and displays at most eight references and
eight hashes with their retained totals.

## Verification evidence

All runs used synthetic data, temporary directories, encrypted SQLite and the
existing offline test guards. No real profile, provider or full test suite was
used. Commands ran through `.venv/bin/python -m pytest -q` with a fresh dedicated
`--basetemp`; the exact selected files are below.

| Selection | Result |
| --- | --- |
| `Tests/Personal_Context/test_settings_provenance.py`, `test_service.py`, `test_proposal_service.py` | 103 passed in 203.52s |
| `Tests/UI/test_personal_context_provenance.py`, `test_settings_personal_context.py`, `test_personal_context_proposal_review.py`, `test_personal_context_review_modal.py`, `test_design_token_governance.py`, `test_css_bundle_sync_guard.py` | 110 passed in 139.33s |
| `Tests/Agents/test_profile_tool_provider.py`, `Tests/Chat/test_console_personal_context_snapshot.py` | 33 passed in 86.07s |
| Full application CSS host follow-up: Settings/proposal files with `-k 'production_settings_inspects or provenance_renewal'` | 5 passed in 18.14s; both widths and all three proposal actions |

The follow-up repeats five cases from the UI selection after strengthening their
hosts and geometry checks; it is not five additional unique tests. New source
modules and changed UI tests pass Ruff. Owned Python formatting and whitespace
checks pass. Existing lint findings remain in legacy files: two BLE001 thread
exception collectors in `test_service.py`, 118 in `settings_screen.py`, 11 in
`personal_context_panel.py` and four in `personal_context_review_modal.py`.
Comparison against the starting code found no newly introduced lint finding.
The agent/context run retained an existing invalid-escape SyntaxWarning at
`Tools/patch_tool_impls.py:32`.

The initial service test run failed on the missing API. A second-read foreign
profile regression failed before the profile check was added. The widget run
failed on the missing widget; production-host cases failed before integration.
An unmount regression exposed loss of the widget's App lookup in a late thread;
capturing App at dispatch fixed it. Broad UI verification exposed two startup
timing regressions, fixed by preserving the panel's existing startup timing,
and three test races with a save's follow-up read. Those tests now wait for the
new selected version or policy while retaining their canonical mutation checks.

The ordinary destination/proposal harnesses omit the application stylesheet.
Their early screenshots are excluded from visual evidence. The final follow-up
uses `_SettingsCssHarness` and `_ProvenanceHost` with the complete bundle. It
checks keyboard expansion, on-screen and contained headings, stale reload,
tombstone mutation controls and preserved proposal inputs/actions.

Verified synthetic renders:

- [My Profile, 100 columns](evidence/personal-context-memory/provenance-record-100.svg)
- [Deleted metadata, 60 columns](evidence/personal-context-memory/provenance-deleted-60.svg)
- [Proposal review, 60 columns](evidence/personal-context-memory/provenance-proposal-60.svg)

These are mounted production-surface renders, not a claimed full-app live launch.
Raster inspection used the existing Cairo library; some terminal glyphs have
fallback rendering. Original SVGs are retained. Existing design tokens were not
changed, and broad application theme/contrast certification is not claimed.

## Native execution decisions

1. Run the approved targeted selections instead of skill examples' full suite.
   The explicit user/repository restriction wins. Cost if wrong: regressions
   outside the affected area remain unmeasured.
2. Preserve the existing broad exception collectors and other legacy lint
   findings instead of changing unrelated behavior. Cost if wrong: whole-file
   lint cannot be described as clean; new/changed diagnostics are checked and
   the existing findings are disclosed above.
3. Forward non-bubbling Textual screen lifecycle events from canonical
   SettingsScreen and the proposal modal. A child cannot observe those events
   immediately by itself. Cost if wrong: the extra UI owner and its lifecycle
   interaction need correction; production-host suspension/resume is tested.
4. Record successful targeted receipts directly instead of rerunning tests only
   to satisfy the ledger helper. Cost if wrong: completion depends on the saved
   command output and exit receipt. Behavior changes or failures trigger new
   affected runs, as documented above.
5. The reviewer set aside broader provider disclosure and cross-owner
   forgetting. Keep those contracts in TASK-25907.6/.7 under ADR-182: this slice
   provides user-owned Settings inspection, not new disclosure or erasure
   guarantees. Cost if wrong: existing provider/filtering and derivative-removal
   limitations still affect users; the roadmap retains those known limits.
6. The reviewer inspected proposal integration but set aside broader existing
   body rendering and acceptance contracts. Preserve them because the changed
   wiring and all three mounted actions passed regression checks and no new
   issue was identified. Cost if wrong: an untested baseline issue could remain;
   this is not a certification of all proposal behavior.
7. Complete documentation closeout as the implementer because the independent
   verdict covers the implementation commits. Check task criteria, links,
   evidence and scope locally. Cost if wrong: documentation mistakes have only
   the author's validation, not a second independent review.

## Independent review

One fresh read-only reviewer (gpt-6-astra) reviewed
`42ed50ab7a..05c94995c0` with the plan, spec, all five Review Focus items and
the decision ledger. It inspected the implementation and test receipts and
ran a focused mounted probe. It found no Critical or Important issue. The
implementation author also checked the reported lifecycle behavior against
the code. No second review or unrelated fix pass was run.

**Deferred minor: the changed-state recovery control disappears on collapse
or screen suspension.** In
[`PersonalContextProvenanceDetails.suspend()`](../../../tldw_chatbook/Widgets/Settings_Widgets/personal_context_provenance.py),
the content-free changed message and Reload details action are replaced while
the stale-selection latch stays set. Re-expanding cannot renew or recover that
captured section. Recreate the containing view/selection to recover. The
reviewer's mounted probe reproduced the missing action after collapse; the
same suspension method is used for screen coverage. This remains Minor by
user effect: provenance becomes temporarily inaccessible, while stale
metadata stays cleared and canonical edits/approvals are unaffected.

The follow-up should preserve the content-free changed state and restore its
message/action after expansion or resume, with mounted regression cases for
both transitions and the existing stale-result fence. It is recorded, not
implemented, under native execution's minor-deferral policy.

The review's three declined-to-judge areas are resolved in decisions 5–7
above. No independent whole-application privacy, proposal-contract or final
documentation certification is claimed.

## Acceptance and closeout

| Criterion | Evidence |
| --- | --- |
| 1: Recorded metadata on both surfaces | Service projection cases and mounted My Profile/proposal inspection |
| 2: No inferred quotation, authority or confidence | Independent reference/hash states, unknown-history and approval/edit cases |
| 3: Honest manual/migration/approval/promotion/deleted states | Stored-field classification and metadata-only tombstone tests |
| 4: Settings authority and lifecycle fences | Private-record positive control, unchanged agent denial, identity/purge/version/lock/late-worker cases |
| 5: Existing fields and services, no new storage/provider path | Read-only encrypted-repository tests, unchanged canonical fixtures, code review |
| 6: Bounded literal rendering | Markup, terminal/directional controls, long references and Unicode cases |
| 7: Targeted integration and narrow layouts | 246 distinct selected tests, five repeated full-CSS checks, retained SVGs and independent review |

Scoped closeout passed: the ten existing task-family files, 59 unchanged child
criteria (13 checked), backward-only dependencies, 46 local links across 15
documents and whitespace. The task-ID guard ran with
`.venv/bin/python scripts/check_backlog_task_ids.py --tasks-dir <temporary-copy-of-the-ten-task-files>`;
it found no filename/frontmatter collision or Windows-incompatible task path.
The seven provenance criteria are complete; the other seven unstarted child
tasks remain To Do. TASK-25907.3 (Next Send selection explanations) is next.
No general lesson is newly introduced: the stylesheet-host trap follows the
existing [production hierarchy and stylesheet lesson](../../../backlog/docs/lessons-testing-evidence.md#a-geometry-harness-must-mount-the-production-hierarchy-and-stylesheet).

Final static commands (exit 0):

```sh
.venv/bin/python -m ruff check --no-cache tldw_chatbook/Personal_Context/settings_provenance.py tldw_chatbook/Personal_Context/service.py tldw_chatbook/Widgets/Settings_Widgets/personal_context_provenance.py Tests/Personal_Context/test_settings_provenance.py Tests/UI/test_personal_context_provenance.py Tests/UI/test_settings_personal_context.py Tests/UI/test_personal_context_proposal_review.py
.venv/bin/python -m ruff format --check tldw_chatbook/Personal_Context/settings_provenance.py tldw_chatbook/Personal_Context/service.py Tests/Personal_Context/test_settings_provenance.py Tests/Personal_Context/test_service.py tldw_chatbook/Widgets/Settings_Widgets/personal_context_provenance.py tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py Tests/UI/test_personal_context_provenance.py Tests/UI/test_settings_personal_context.py Tests/UI/test_personal_context_proposal_review.py
git diff --check
git diff --check 42ed50ab7a..HEAD
```

Ruff checked seven files without findings; all ten formatting targets were
already formatted. Legacy differential lint limitations are disclosed above.

## Commit boundaries

- `27efca2711` — immutable service projection and service tests.
- `05c94995c0` — provenance UI, host lifecycle integration and mounted tests.

The branch remains local. No merge, push or PR was requested.
