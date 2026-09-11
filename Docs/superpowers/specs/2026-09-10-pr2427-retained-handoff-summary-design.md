# PR2427 retained Handoff summary repair

Status: user approved the in-place design on2026-09-10; written-spec review pending.
Task: TASK-31932, latest-dev integration.

## Evidence and scope

At31d32d9d2c, the complete companion reader/workspace cohort finishes78 passed,
2 failed (`/private/tmp/pr2427-latest-reader-contracts.log`). Both failures are
in `Tests/UI/test_post_release_workspaces_library_depth.py`: the retained Handoff
Static says sources are unavailable after the source snapshot arrives. The
first test already proves the action button has the current cross-workspace
tooltip/class, so another readiness delay cannot repair the omitted label write.

Source arrival invalidates the existing workspace-depth cache correctly.
`LibraryRail.sync_state` updates retained source/count/DB rows and the action
button, but not the Handoff summary. Do not change eligibility policy, source
queries, workspace ownership, accepted copy, focus, scroll, or widget lifetime.

## Design

Add one optional keyword-only owner-formatted summary string to the existing
`LibraryRail.sync_state` presentation update. The default is None. The Screen
passes `_workspace_handoff_summary_label(self._library_workspace_depth_state())`
at all three callers: strict entry reconciliation, snapshot reconciliation,
and lifecycle reconciliation. The existing policy owner remains the authority;
the rail does not count records or derive eligibility, reasons, or remedies.

On the existing in-place path, update the mounted `#library-workspaces-handoff`
Static using `library_dim_label_text("Handoff", summary)`, the same rendering
used by first composition. Preserve the exact widget instance. None means leave
the label untouched, and an absent optional Static is a harmless no-op for
standalone rails. An empty supplied string remains a supplied value. Keep the
existing full-recompose path and body factory authoritative on actual shape
changes; do not call the body factory to extract text or force a recompose.

No change to the active-workspace label is bundled into this demonstrated
source-arrival bug. Do not add a new controller, state cache, generalized row
patch framework, logging, timer, persistence, or cleanup behavior.

## Alternatives

- Rebuild the Details panel: rejected for this repair because it replaces live
  widgets and adds unnecessary focus/scroll/lifecycle risk.
- Have the rail read workspace services or rerender its body to recover text:
  rejected because formatting/eligibility already has an explicit owner and
  constructing widgets is not a presentation-state API.

## Verification

Before implementation, add a retained-label regression to the existing
`Tests/Widgets/Library/test_library_rail_workspace_action_sync.py` and record
RED. Exercise empty, eligible/blocked, and empty-again owner state; assert exact
summary plus current action tooltip/class, unchanged Static/Button identity,
unchanged focus/scroll, and no extra body-factory call. Cover omitted summary
and absent optional Static without remount. Add a mounted source-arrival
regression using the existing entry snapshot reconciliation harness, preserving
rail identity and focus. Verify all three Screen callsites supply the value.

Run the complete workspace-action sync, post-release workspace-depth, entry
compose-once and affected rail tests with isolated authority paths. Re-run the
two original failures unchanged, scoped lint and existing architecture caps.
Use observation-only native teardown evidence for the mounted cohort. Do not
claim the unrelated13 size-limit failures are solved or increase any ceiling.

ADR required: no new ADR.
ADR path: backlog/decisions/141-library-work-retirement-and-in-place-sync-followups.md
and Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md.
Reason: correct an omitted presentation write through the existing retained-
owner refresh boundary; no new owner, lifetime, policy or service authority.

## Workflow gates

- [x] Explore exact failures, existing owners and all refresh callers.
- [x] Compare in-place update with Details replacement and explain trade-offs.
- [x] Obtain user approval of the in-place design.
- [x] Write this bounded spec.
- [ ] Independent spec review.
- [ ] User review of written spec.
- [ ] Write the implementation plan before code changes.
