# Restore agent worktrees and complete confirmed recovery

Scope: TASK-31210 and TASK-31211. The user approved restoring creation, then completing confirmation and recovery. ADR-155's current decision governs this work and supersedes the earlier worktree execution-qualification design.

## Creation

Reuse ordinary local Git worktrees. The accepted Console turn already contains the exact project-instruction selection and its captured `RunAdmittedWorkspaceRoot`. Pass that exact selected writable named authority through `ConsoleAgentBridge.run_reply(worktree_repo_authority=...)` to `AgentService(worktree_repo_authority=...)`. Default None refuses isolation; a compatibility root, one unselected folder, private scratch, or currently viewed workspace never supplies authority.

Before Git, require the real LocalToolProvider plus matching live binding, writable access, unchanged root identity, and clear kill switch. Use the selected source root. Validate again after creation; retain the created record/checkout even if revalidation, provider admission, cancellation, or thread start fails. Child path tools use the existing per-run routing and pinned executor; their guard also checks the original source authority and child identity. Shell and virtual CLI remain excluded from isolated children. No-fleet/inline worktree requests refuse without shared fallback. Automatic retirement only removes provider routing; no checkout or branch deletion.

Use unique generated child destinations and exact validated run identifiers. Fixed local Git commands disable hooks for generated mutations. No command, executable, config map, environment, external destination, or network operation comes from the model. This patch does not replace Git or the public read-only Git worker. Application checks detect drift at operation boundaries; concurrent external root/metadata replacement during a Git command is an explicitly accepted limitation of ordinary local Git, not a guarantee supplied by before/after checks.

## Confirmation and durable records

Preserve the existing controller's exact-request Allow/Deny and parked/remounted interrupt lifecycle. Wire a real Console card showing operation, source, destination and bounded diffstat. Preview/live tool disclosure must match the actual surface. Mutation must revalidate authority and source content after Allow, refuse changes requiring a fresh preview, and preserve work on denial or conflict. Never silently merge, discard, retry, or fall back to shared execution.

AgentRuns schema 19→20 stores local-only original base commit, child/source identities, exact run/conversation/binding ownership, writer state and mutation state. Record creation before child execution. Recovery never guesses ownership from an agent branch prefix or reconstructs an unknown base. Legacy unrecorded checkouts remain untouched.

Use the existing ExecutionOwner for physical lifetime. Add an outside-lock one-shot drain callback and a sticky cleanup-unproven flag. Persist drained only when the root and every owned operation actually finish. A terminal DB row alone is insufficient. A held/uncertain record stays protected across restart. Transactional claims prevent duplicate recovery; ambiguous outcomes remain uncertain and are never automatically replayed.

## Recovery and discard

Add a Console Recover agent work entry listing recorded work for this conversation and its current exact writable selected repository. The recovery operation owns its cancellation signal independently of any finished turn. Navigation parks/remounts confirmation; session close cancels only its own operation. File/DB reads and Git operations run off the UI thread.

Confirmed discard restores the original base, clears child changes without following symlinks, detaches the exact branch and deletes its ref with an expected-old-value check. Retain the baseline checkout and disclose that in the receipt. Automatic and forced pathname root deletion remain disabled. An operation lacking a required platform primitive refuses specifically; it does not disable unrelated supported operations.

## Verification and delivery

Deliver and review creation first; then durable ownership and confirmed mutations; then the visible Console card and recovery list. Use real temporary Git repositories and SQLite under Tests/conftest isolation, plus mounted wide/narrow UI verification. Cover successful isolated writes, wrong/missing/read-only authority, revocation, changed roots, failure retention, positive drain, denial, conflict, changed preview, restart and duplicate/uncertain recovery. Keep evidence precise about command-boundary checks and actual tested platforms. Targeted checks only; no full-suite claim.
