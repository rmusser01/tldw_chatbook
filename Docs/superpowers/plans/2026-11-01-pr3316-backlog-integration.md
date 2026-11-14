# PR3316 backlog integration plan

> For agentic workers: use superpowers:executing-plans inline and superpowers:requesting-code-review for independent final review.

Goal: integrate the existing preflight and superseded-run cleanup implementation while preserving every current-dev lane, guard and verdict.
Architecture: retain the checkout-free cleanup job and preflight graph from TASK-38413; reconcile them onto dev after the older PR3315 merges.
Tech Stack: GitHub Actions, Python/pytest/PyYAML, Node inline-script tests.
Spec: Docs/superpowers/specs/2026-10-29-ci-preflight-and-superseded-runs.md.

ADR required: no additional ADR beyond the source amendments.
ADR paths: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md; backlog/decisions/218-in-repo-merge-queue.md.
Reason: source already records the CI ordering and bounded cancellation policy.

## Constraints and review focus
- Source head 1a536c706c42efbcc63340d603d66061fb19accc is OPEN/non-draft/unarmed. Refresh immediately before action.
- Preserve current-dev 32 UI shards, max-parallel17, all seven application lane definitions, installs, selectors, timeouts and UAT steps; only add their preflight dependency.
- Keep all13 moved artifact guards byte-equivalent in parsed YAML and retain report-all-failures conditions.
- Missing/changed PR or run identity must never authorize cancellation; preserve current-head evidence and any controller/verdict work.
- User waived hosted CI and size limits; report hosted scheduling/cancellation/throughput as unqualified, never as measured.
- No full suite, manual required-workflow dispatch, release or main promotion. Drafts and PR3267/3297 excluded.

## Execution
- [x] Refresh dev after PR3315 merge, record its exact SHA, then rebase source while retaining both lesson sets and newer workflow/tests/ADRs.
- [x] Compare parsed application jobs against exact base after removing only new needs. Compare moved guard steps; verify all32 shards and UAT selectors survive.
- [x] Run the focused queue/workflow/preflight/cancellation contracts from the initial review, excluding only unchanged exhaustive shard redistribution. Run actual13 artifact guards and the two selector checks with pinned Mermaid reproduction using LF Git bytes.
- [x] Diagnose failures against exact base; preserve guards and evidence. Run scoped AST/lint/diff checks.
- [ ] Record QA and task evidence with hosted limits, commit, obtain independent immutable-head review, then exact-lease publish and authorized merge after refreshed eligibility/threads.
- [ ] Confirm actual merge and identical branch-protection restoration before next backlog item.
## Integration ledger

Ruling: prepare and test the independent CI delta on frozen PR3315 head 5d7cad1c8598495e398cf25cd976352c8689b23a while its UI run completes. Publication remains ordered after PR3315; rebase onto its confirmed merge and repeat any validation affected by later repairs before final review. This overlaps local preparation only, not merges.

### Baseline preflight repairs (2026-11-01)

The frozen CI run and exact PR3315-base comparison both fail the fixture-derived edit/resend node selector and stale inherited-target list. Replace the two selected edit/resend cases with the entire six-case function (coverage expands), and retain the existing terminal-input target in the preservation assertion. All other application-job bytes remain unchanged apart from preflight needs.

The actual markup guard also found two unchanged Kanban exception notifications without literal rendering. The screen, checker and census are identical to the provisional base. Add a mounted error-path regression with notifications enabled, observe it fail, then pass markup=False at those two existing sinks. Keep the census untouched. ADR required: no new ADR; routine enforcement of existing literal-notification convention under ADR237/ADR150. Run the affected Kanban cases and unchanged markup guard.

The next selector pass exposed the same fixture-ID limitation on compact-now warm-agent. Select the entire three-case function too; coverage expands from one to three, with no checker change.

Confirmed base: de5dec02b104a061793504c12426987510652e57. Rebase range-diff preserves all four commits exactly. Current-base parsed preservation and both selector checks pass; no additional production changes were made.
