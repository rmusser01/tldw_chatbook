# ADR-155: Confirmed agent worktree recovery with explicit filesystem authority

Status: Accepted
Date: 2026-09-12
Tasks: TASK-31210, TASK-31211
Related: ADR-067, ADR-069, ADR-082, ADR-101, ADR-129, ADR-139, ADR-150

## Current decision: restore ordinary local Git operations

The user approved restoring worktree creation with repository authorization and identity checks, followed by Console confirmation and durable recovery. This section supersedes the execution mechanism and platform-qualification requirements below. The earlier blanket refusal was an interim implementation decision, not a missing external dependency.

Use the existing local Git lifecycle implementation. Thread the exact selected `RunAdmittedWorkspaceRoot` from the accepted Console turn through its bridge into AgentService. Do not derive it from the provider compatibility root, the currently viewed workspace, a stored locator, or an arbitrary first binding. No selected writable named binding means refusal. Revalidate the captured binding, root identity, and tool kill switch before creation, after creation, and before child path operations. Parent/child metadata identity and source state must also be checked before confirmed recovery. Revocation refuses further work; partially created checkouts are retained.

These checks detect observable changes at application boundaries. They do not provide atomic protection against an external process replacing a repository or its Git administrative metadata during a Git command. The copied-metadata experiment remains valid evidence of that limit. Ordinary local Git operation with this documented limit is the selected scope; a new sandbox, alternate Git implementation, metadata alias scheme, or proof against every concurrent root replacement is not a prerequisite. ADR-101's existing filesystem/read-only Git worker behavior is unchanged. Internal Git commands remain fixed application operations, with no model-selected executable, argv, environment, network operation, or hooks for generated mutations.

Keep automatic cleanup and failed-start deletion disabled. Creation does not depend on a merge card. Enable merge/discard only after exact human confirmation and durable positive writer-drain checks are implemented. Preserve the original-base/ownership records, transactional mutation claim, no uncertain replay, and separate recovery cancellation lifetime described below. Confirmed logical discard retains a detached baseline checkout and never force-removes a root by pathname. Platform support follows the actual operation: portable creation is not blocked by a missing descriptor-relative discard primitive.

Current spec: [Worktree restoration](../../Docs/superpowers/specs/2026-09-12-agent-worktree-restoration-design.md). First implementation slice: [Creation restoration](../../Docs/superpowers/plans/2026-09-12-agent-worktree-creation-restoration.md). Previous execution experiments and results remain historical evidence; they do not override this decision.

Implementation clarification: opening another AgentRunsDB handle never rewrites held or in-flight records. A held writer has no positive completion proof and remains unavailable; this does not require guessing whether a previous process crashed. Exact operation completion may release a claim back to unresolved only after positively proving no destination effect, for example a pre-apply conflict or an oversized patch after a disclosed child-only capture commit. Ambiguous Git effects or failed completion persistence remain in-flight/uncertain and are never replayed. Preview remains capped at8192characters and patch spooling at32MiB. Logical discard initially uses POSIX descriptor-relative no-follow cleanup and retains the baseline checkout; missing discard primitives refuse that action without disabling creation or other supported actions.

## Historical design before the scope correction

## Decision

Console exposes a per-call same-turn confirmation and a user-operated Recover agent work list. Previous-turn work is identified by local durable structural records joined to AgentRuns ownership; old fleet handles and model conversations are not revived. Every operation captures the current exact writable named Workspace binding and validates its fingerprint/root identity again after consent. Stored locators and repository branch names are evidence, not authority.

AgentRuns adds versioned local-only worktree records containing original base SHA, exact source branch/path/identity, destination repository/binding identity, and unresolved/in-flight/resolved/uncertain outcome. Creating an isolated child records identity before child execution. Missing ownership/base is not reconstructed heuristically. Temporary Console scratch remains nonpersistent/nonrecoverable under ADR-082.

Recorded unresolved and uncertain work survives GC even when clean. Unknown agent-prefixed branches are not implicitly owned. A mutation reserves its record by exact transactional compare-and-set; ambiguous worker or persistence completion leaves uncertain state and disables automatic retry. This ADR does not promise exactly-once Git effects across a crash.

Mutation occurs through a narrow extension to ADR-101's contained one-shot worker: closed internal create/preview/apply/merge/discard operations, strict app-originated identity records, no arbitrary Git argv/shell/executable/config. Git inherits pinned cwd and existing containment. Adding these specific internal mutating operations is authorized by this ADR, not by ADR-101 alone; public read-only git_* tools remain read-only. Two-root operations must retain both admitted source and destination identities, with supported-platform evidence or fail-closed refusal.

A visible exact-request Allow is required for landing/destruction; permission/binding gates still apply. Worktree confirmations retain ADR-139's separate host path and ADR-067 human-wait behavior. Recovery owns an independent retained operation/cancellation signal rather than borrowing the already-finished primary turn's event. Navigation parks presentation; closing the owning session cancels pre-admission work; ambiguous post-admission outcomes remain uncertain.

A terminal DB run is necessary but insufficient for destructive recovery: existing runtime can terminalize an abandoned live thread. Reuse the existing ExecutionOwner/OwnedOperation ledger: one-shot callbacks run outside its lock after root and every physical tool/model operation finish, persisting writer_state drained. A sticky cleanup-unproven flag instead persists uncertain; prior-process held records reopen as uncertain. No second ownership registry is introduced; unproven abandoned execution remains non-actionable. Logical fleet pruning cannot erase these fences.

## Discard and cleanup policy selected by root

Confirmed discard restores the pinned child to recorded base, removes untracked data descriptor-relatively without following symlinks, detaches HEAD and removes the exact agent branch by expected-old-SHA update-ref. The baseline checkout remains as discarded_cleanup_pending and is explicitly reported. Tool descriptions and task criteria reflect this retained checkout. No forced pathname root removal or automatic root-removal GC is introduced.

This avoids pretending `git worktree remove --force <locator>` remains bound to a previously pinned child. Storage residue is accepted. Unsupported platforms refuse multi-root mutations before execution; real POSIX qualification and explicit Windows refusal tests are required. This planning decision authorizes implementation, not any operation on actual user data.

## Context

The original service stores worktree handles/base only in a turn-local dictionary. New services cannot find previous work. Git worktree listing does not encode original creation base. GC preserves dirty work but can remove a clean unmerged checkout and leave only its branch. Same-turn merge/discard plumbing exists but its visible card is absent. Activating it also exposes use of compatibility provider.root instead of current admitted binding.

Root-pinned execution already exists for local fs/read-only Git. Reusing pathname lifecycle helpers in the app would bypass it, and generalizing the worker to arbitrary Git would introduce a larger authority than needed. Durable recovery also adds a lifecycle outside any agent turn, so its cancellation must not be owned by disposable views or stale primary signals.

## Alternatives considered

- Persist/reconstruct FleetHandles: rejected; mixes execution continuation with file recovery and still lacks durable original base/authority.
- Discover every `agent/*` branch and infer base with merge-base: rejected; branch naming grants no ownership and changed ancestry does not preserve original delta.
- In-memory handle preservation alone: rejected; bridge constructs new services and application restart loses state.
- Persist scratch and reopen it as recovery authority: rejected by ADR-082 temporary-chat contract.
- Treat terminal run status as drained execution: rejected by actual `_settle_fleet` abandonment path.
- General raw-Git executor or shell fallback: rejected; closed operations and existing containment suffice.
- Force-remove by pathname after checking identity: rejected; check/use can delete a replaced root.
- Automatically retry ambiguous apply: rejected; prior effect may have succeeded and duplicate application cannot be ruled out.

## Consequences and limits

Users gain explicit recovery for newly recorded prior-turn work in their current authorized repository. Every source/destination is reviewable and every mutation individually confirmed. Sensitive locators stay local and out of model-facing run metadata/export. Storage retention increases for uncertain, legacy and cleanup-pending work; this is intentional preservation, not hidden deletion.

Unsupported platform pinning fails closed. Root pinning does not create a general hostile repository sandbox or protect against all descendant metadata/config/filter manipulation. Actual Windows secondary-root retention and operation cleanup require platform evidence. Legacy unknown-base records and post-crash uncertain mutation do not receive speculative automatic repair.

## Acceptance evidence required

Real SQLite migration/reopen and two-turn recovery; actual pinned temp-repository operations; mounted exact-ID confirmation/remount; deterministic root and authority replacement races; live abandoned-owner refusal; crash-after-effect uncertainty; no replay; targeted CSS/import guards and unchanged read-only Git exposure. Final task notes link exact commands and limitations. No full-suite claim follows from these targeted checks.

## Refined implementation consequence

The existing capacity ledger is the positive physical proof. A DB terminal row plus an absent runtime snapshot is never the proof; a durable exact-owner drain callback is. Child workspace executor cleanup-unproven errors mark the owner before its physical tool lease is released, so completion cannot accidentally clear an unproven descendant. Recovery itself uses that same capacity ledger with its independent manual execution owner and cancellation event.

Selected logical discard is POSIX-only initially: pinned-child base restoration, descriptor-relative no-follow untracked removal, detach and exact-ref CAS, with baseline checkout retained. Standard linked-worktree administrative metadata is validated but not claimed immune to arbitrary descendant/config manipulation; nonstandard external gitdir/submodule layouts refuse. Windows and missing descriptor capabilities refuse rather than use pathname clean/remove. Root must retain these explicit limits in canonical ADR and UI results.

## Git configuration and bounded operation policy

Internal worktree workers retain only host-originated absolute HOME/XDG_CONFIG_HOME outside the admitted root for user Git configuration. They keep the existing safe executable search and inherit no raw Git injection, SSH/askpass credentials or model-selected environment. Child commits retain fixed agent identity; parent merges use configured user identity and refuse if missing. Generated commit/merge hooks are disabled; existing attribute/filter behavior remains under the non-hostile-metadata limitation. Preview is capped at8192characters; binary patches are streamed to an owned spool capped at32MiB and oversize refuses before destination mutation. A confirmed child capture commit may precede that refusal; preserve its source, original base and unresolved record, and disclose the outcome.

[Design](../../Docs/superpowers/specs/2026-09-12-agent-worktree-recovery-design.md) · [Plan](../../Docs/superpowers/plans/2026-09-12-agent-worktree-recovery.md)
