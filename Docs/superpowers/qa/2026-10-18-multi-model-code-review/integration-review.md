# TASK-35582 scoped delivery integration review

What this change does: It carries the approved review feature onto current dev while preserving current dev's runtime and workspace changes. It also renumbers the feature decision to ADR-281 and keeps a smaller, explicitly indexed evidence packet.

**Findings: none in the requested integration scope.**

**Verdict: ready for the remaining bounded delivery work.** This is not an AC24 completion or merge-readiness verdict. The candidate is still uncommitted at review time, and AC24 correctly remains open.

## Reviewed boundary

- Candidate: `/Users/macbook-dev/.codex/worktrees/task-35582-delivery/tldw_chatbook`, branch `codex/task-35582-delivery`.
- Base/current-dev HEAD at inspection: `e3d07afa65927b9d477fe9f13af619550da9e005`.
- Three-way references: task base `b07fc0425288fb51d17cde5f745051720a7c9db6`, approved task `3bce87f15c25e0f2ffb3b4048d11ca6ee1842caa`, and current dev above.
- Read `integration-report.md`, task acceptance/status and compressed implementation notes, retained evidence README/manifest, conflict resolutions, and relevant three-way changes in the nine overlapping Python owners. Static AST comparisons narrowed attention to functions changed on both branches; none of dev's changed functions reverted exactly to the old base.
- Expected operating context: the existing single-user Console with its existing finite worker/fleet ownership. No new load or broader feature-contract audit was undertaken.

## Integration checks

- **Bridge conflict:** `_consume` retains both `received_item` and `call_continuation_target` as nonlocals. Dev's cancellation poll remains wired into the actual stream iteration, every received chunk sets `received_item`, and stall streaks reset only after a received item. Child continuation identity is rebuilt from the effective routed destination. Dev's model-loop default-executor drain is also retained.
- **Controller conflict:** the captured owning-loop path executes the authority callback directly; calls from other threads retain Textual's native-thread identity guard and dispatcher. The authority callback itself still performs the existing current-selection check. The merge does not remove an authority gate.
- **Wake overlap:** the existing automatic-primary drain remains after delivery settlement and before slot release. Review-boundary eligibility is composed into that owner; the integration does not replace dev's worker-drain lifetime.
- **Other overlapping owners:** LocalToolProvider retains project-index capability/read-floor handling; the new source authorization is restricted to existing native read operations and rechecks binding/deny state after approval. Store cancellation cleanup and durable activity-row parameters, detached turn-context fields, session change-review root/stamp handling, ChatScreen's dev UI work, workspace index mutation notifications, and config credential/write-lock behavior remain present. Review changes in the other owners are additive or the already approved task changes; no concrete caller break caused by integration was found.
- **AgentRuns:** current dev's owner is schema 24. The candidate's only removed line from that owner is the schema constant changing 24 to 25; the boundary table/link and version row are additions. Existing goal fields and migrations are preserved. No schema collision or lost dev field was found.
- **Documentation conflicts:** the file-authority document retains optional folder-index authority and adds captured-review authority. Both lessons documents preserve dev incidents and append the task incidents; the inspected integration diffs do not replace dev history.
- **Generated CSS:** the resolved self/scoped files add the review modal, and the file-notes split adds the new shared control-width token. The report records generation by the normal builder. I did not rerun it.
- No unresolved Git conflict entries exist.

## ADR and evidence retention

- The candidate preserves current dev's `240-mcp-destination-retirement-tools-hub-in-settings.md` byte-for-byte.
- The feature decision is uniquely `281-primary-agent-multi-model-code-review.md`. No old feature ADR-240 path or ADR-240 reference remains in changed canonical Markdown/Python outside the retained historical QA files. Newly added Markdown links in the changed canonical documents resolve locally, including retained imported design documents.
- All 15 retained evidence files match the manifest's SHA-256 and byte count. The two required fixtures match the approved task revision byte-for-byte.
- The history bundle exists, is 6,095,386 bytes, and matches manifest SHA-256 `a34611fa85d92e1344ff5ab6f0457cb2e5c97ef0ef8097497569fbde43110fec`. The manifest identifies its prerequisite and the preserved source branch/commit. This check verifies bytes, not an independent restore exercise.
- The evidence README and task accurately keep the latest live verifier incomplete, total historical spending at 255 requests/18 children, and new allowance/spending at zero. Historical ADR numbering is explained rather than silently rewritten inside verbatim reviews.

## Verification and limits

The integration report records 97 focused passing nodes and two current-dev Stop/stall controls, with delivery-source import assertions. These are reused verification results, not tests rerun by this reviewer. Their scope does not establish the remaining live Console/provider qualification.

No source edits, tests, provider/network requests, credentials, push/PR actions, subagents, or new spending were used. This review wrote only this report. Root still owns final staging/commit, the private qualification gate, explicit budget authorization, and the narrow remaining AC24 work.

Not checked: whole-feature requalification, full suite, live providers, bundle restoration, or the eventual final commit. Risk: AC24 remains open and no additional live allowance is authorized.
