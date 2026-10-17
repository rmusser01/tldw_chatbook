# Managed plugins PR3148 dev rebase

**Goal:** Rebase PR3148 onto recorded current dev, reconcile both implementations without losing approved plugin/hook behavior or current host fixes, run relevant targeted checks, obtain the explicitly requested independent code review and publish the rebased PR branch.

**Context:** Original reviewed/published head7b4d5c501ba8500267d8f9f84f524742c33ec882. Pinned dev1fac8f83291f5b48c3daa7b9095253a23c05d44d. Common base06cc148a91b7a7d4b4b45ad45431e737b0b13081. PR3148 is draft with auto-merge off. The merge preview reports165 conflicted paths across existing host owners and overlapping plugin/hook implementations. These are not permission to wholesale prefer either tree.

## Global Constraints

- Use only /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/managed-plugins on codex/managed-plugins. The primary checkout and unrelated worktrees are read-only references.
- Preserve the original published head and all ignored evidence before rewriting history. Root owns Git, Backlog, ADR/rulings and publication; one implementer owns product/test conflict resolution and targeted verification. No concurrent implementers or worker helper/reviewer agents.
- Preserve existing current-dev features and the approved plugin/hook contracts. Adapt changed call sites, ownership, persistence and navigation seams rather than restoring removed infrastructure or taking entire conflicted host files from one side. Read current architecture/ADRs and design-token/component documents before affected UI work.
- One installation and selected revision exist per user-data directory. Global activation starts disabled. Catalog discovery does not install or enable entries.
- Required constraints never disappear because parsing, configuration, hooks or persistence fail. Native/foreign instructions remain attributed untrusted context.
- Package activation grants no filesystem binding, tool permission, network credential or trusted project status. Apply current authority before injection/launch/dispatch/result acceptance; disable and cleanup preserve exact scoped ownership and actual terminal proof.
- Keep current dependency pins and existing crypto/credential/SQLite/httpx services. No new package/permission runtime, automatic package build/install, broad platform support or unreviewed public contract. Existing explicit package commands remain distinct from absolute host executable lookup.
- Run targeted checks tied to actual changed owner/caller/fixture boundaries on the rebased source. Preserve commands, raw output, before/after source copies and real failure outcomes; private pre-import HOME/XDG/profile, null keyring and external-network refusal. No real profile/keychain, broad host execution or dependency installation. Request a concrete scoped context change from root only if actual evidence proves one is necessary.
- Full-suite runs require explicit user opt-in; the request authorizes relevant targeted checks only. Passing pre-rebase checks do not qualify the new base. Do not repeat successful unchanged checks without a new source/failure reason.
- Preserve source modules for generated CSS and rebuild with the existing builder. Do not hand-edit generated bundles or relax static/startup/module caps to make integration pass.
- Independent review occurs after resolution and covering checks, using the requesting-code-review template and a concrete current-dev-to-final-head package plus resolution/range evidence. Do not ask the reviewer to repeat passing tests. One reviewer gate satisfies this single integration task and its final review; no duplicate seat.
- User authorized updating PR3148. Publish with an exact force-with-lease against the observed remote head after review, retain draft status unless readiness is established, and never merge or arm auto-merge. Detect a moved remote branch instead of overwriting it.

ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: Existing accepted plugin/hook contracts govern direct integration. No new ADR is planned; propose any unavoidable contract decision explicitly before code and preserve newer applicable host ADRs.

## Task 1: Rebase PR3148 and qualify the integrated owners

**Backlog:** TASK32695 is reopened for one explicit integration criterion, preserving its14 completed historical criteria and notes.

### Requirements

1. StageA is read-only diagnosis. Compare original head, pinned dev and common ancestor, trace current callers and overlapping plugin/hook implementations, and provide a concrete resolution proposal with semantic conflict groups, exact initial edit/read-only paths, dependency/config/export/schema/UI consequences and a bounded negative/positive targeted-check map. Identify already-landed patches and changed architecture before any product edit. Include exact private runner/provenance plan and any genuinely unavailable checks. Do not claim165 merge-preview paths are necessarily the final rebase conflict set.
2. Root accepts the proposal and records the task/plan before StageB. Root saves a local pre-rebase branch/ref, starts a normal rebase onto pinned dev, and controls stage/continue/commit. Drop only patches proved already applied or genuinely empty after approved resolution; retain a per-stop resolution record. Worker edits product/tests only after accepted scope, using actual current host owners. Root resolves Backlog/Git/ADR metadata. Any new semantic/interface decision needs root ruling before code; routine path expansion is recorded with its cause.
3. Review the complete resulting diff against pinned dev, and compare the replay to the old published branch with range-diff/resolution evidence. Verify no upstream behavior or approved component/required-hook protection was lost, schema/private-owner inventories and routes remain coherent, and source/fixtures/config/export consumers match.
4. Run targeted behavioral and static checks for changed integration seams: actual Console capture/runtime/queue/child/compaction/disposal, tool operation and permission dispatch, MCP ownership/credentials/transport, plugin admission/variables/data/recovery/acquisition, UI/navigation/CSS/startup as warranted by the actual changes. Choose bounded files/nodes based on source, retain positive/refusal controls, and qualify unavailable platforms honestly. New current-base failures are investigated before broadening; no blanket skips or inherited-pass claims.
5. Hand off a source-bound report and immutable evidence index. Root commits any integration repairs, generates the review package pinnedDev..HEAD and dispatches one fresh independent reviewer using requesting-code-review. Fix real Critical/Important findings with the same implementer and scoped re-review; out-of-scope observations stay explicit.
6. Once resolution, relevant checks and review support it, root closes the integration criterion, updates the PR branch with exact force-with-lease, verifies the remote head/base/conflict status and updates the PR description with current validation/remaining limits. Preserve the worktree and evidence. No merge or automatic follow-up.

### Preflight consistency

| Pair / task | Producer and consumer | Disposition |
| --- | --- | --- |
| Task1 original branch vs current dev | Both define Plugins/hooks/MCP and changed Console owners | StageA must compare actual implementations and current architecture; no whole-tree winner. |
| Task1 code vs verification | Rebase changes caller/fixture context; old passes are historical | Targeted tests run on copied rebased source; gates and unqualified platforms stay explicit. |
| Task1 review/publication | Reviewer needs final source and resolution evidence; push rewrites PR head | One independent read-only gate, explicit lease pin, no auto-merge. |
| Task1 self consistency | Root Git/metadata vs worker product/testing | Sequential handoffs; no source writes or background test phase during root rebase/continue. |
