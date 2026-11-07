# Console approval resync implementation plan

> Agentic workers use the existing approved design and task34564-34569 interfaces; execute the independent conflict domains in parallel, then one integrated fresh review. No new feature design or permission model.

Goal: rebase PR3036 onto current dev, address all current source/review/qualification blockers, and prepare authorized auto-merge under the repository queue rules.
Architecture: preserve the published0b9ae198 snapshot at codex/pr3036-published-20261106. Reuse the clean existing PR worktree on codex/pr3036-rebase-20261106. Compose the approved capture/controls/Details/feedback into current runtime and served relay owners. Host arbitration remains authoritative for either decision origin.
Tech stack: existing Python3.12+, Textual8, stdlib; no new dependencies.
Spec: Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md (approved).
ADR required: existing ADR update/renumber only.
ADR paths: approval interaction ADR235 retained as older; younger passkey decision renumbered ADR354 after524-ref scan, served relay ADR236 and current human-decision coordination ADR220.
Reason: integrate already accepted ownership contracts; no new permission, storage or runtime boundary.

Global constraints: preserve Until Chatbook exits lifetimes, original matching arguments, supported choices, fail-closed checks, private-profile guards, FIFO/deadlines, neutral gestures, token governance, every prior CI census member/limit. Targeted checks only; no full sweep authorized.

Review focus: phone vs desktop resolution races; stale/recreated cards; complete redacted arguments; grouped grant facts; modern grant/checkpoint security checks.

- [x] Resolve provider/provenance conflicts in Agents/{approval_provenance,local_tool_provider,mcp_tool_provider,virtual_cli_provider}.py and the two provider tests. Preserve both fresh dev gates and observation-only capture.
- [x] Resolve runtime/controller/interrupt/wiring/screen conflicts without replacing newer runtime/relay ownership. Keep optional observational callbacks disposable and after actual settlement.
- [x] Resolve card and geometry/compact/budget tests plus approval token additions, preserving modern generic card/served behavior and the approved control/Details lifecycle. Do not hand-edit generated CSS.
- [x] Retain newer workflow/census semantics and all prior executable members; union the approved approval targets. Rebuild generated styles and diagnostic inventory from canonical sources. Resolve user docs/lessons additively.
- [x] Complete rebase; inspect every auto-merged source and range-diff. Retain older approval ADR 235, renumber younger passkey ADR to 354, and update live references, preserving existing evidence provenance.
- [x] Trace served relay mapping and phone-originated resolution through actual current owners. Add minimal regressions for real winning/losing origin, staged desktop state and no duplicate authorization; fix only demonstrated integration defects.
- [ ] Run protected targeted provider/host/feedback/card/relay/CI contracts, source governance and generated-artifact checks. Exercise available live native/browser paths and record actual timing/dispatch qualification; do not turn absent evidence into pass.
- [ ] Fresh independent complete-branch review; address material findings, confirm current-head hosted Qodo review and resolved threads. Push with explicit expected-old-head force-with-lease because user requested rebase. Confirm current queue mode and only then mark ready/arm auto-merge when requirements and evidence allow it. Preserve all worktrees.

Owner steering2026-11-06: native timing and actual Windows dispatch are mandatory before auto-merge; retain draft and do not reclassify missing physical evidence as functional acceptance.

Resync result: provider/card/runtime and served relay integration are verified. Actual Windows dispatch is qualified on final product sources. Fresh source review is clear. Required native/browser presented-frame timings remain unqualified with zero product samples; profiling methods are now explicitly authorized; synthetic capabilities are established, and product qualification is pending. Tasks remain In Progress and the PR remains draft with auto-merge off. See the QA record at `Docs/superpowers/qa/2026-11-06-console-approval-resync/README.md`.

Hosted CI follow-up: fixed demonstrated eager-helper and UAT-label regressions without raising limits; actual pre-layout Details restoration now has a failing-before/passing-after regression. Fresh source review is clear. See `Docs/superpowers/qa/2026-11-06-console-approval-resync/ci-followup.md`. Source is frozen for a local commit and guarded complete-workload measurements; no ready/auto-merge action until qualification passes.
