# ADR 005: Console Workspace Server-Readiness Boundary

Status: Accepted
Date: 2026-06-08
Related Task: [backlog/tasks/task-86 - Console-workspace-switching-server-readiness-contracts.md](../tasks/task-86%20-%20Console-workspace-switching-server-readiness-contracts.md)
Supersedes: N/A

## Decision

Console workspace switching stays local-first and registry-backed in this tranche while rendering explicit server-readiness, handoff eligibility, runtime, and ACP task/run package states behind adapter boundaries; no background sync engine or server hydration is implemented or implied. When no active workspace exists, Chatbook creates a safe built-in `Default` workspace so users can chat and see saved conversations without manually configuring a workspace, but that workspace has no runtime bindings, cannot be granted filesystem/file-tool access, and must not expose stale or externally inserted runtime bindings through registry reads.

## Context

Console is becoming Chatbook's primary agentic control surface. Users need to understand the active workspace, which conversations and sources can be used in that workspace, and whether future server/ACP handoff paths are unavailable, ready, failed, or blocked.

The current workspace model already has a local registry, membership transfer policies, runtime bindings, and authority/sync enums. The adjacent `tldw_server2` workspace implementation is also local-store-oriented in the browser UI, exposes server runtime availability as a status, and models ACP workspace/task/run records separately from generic chat state. Server-side sync is not complete and must not be represented as working.

The product contract requires global Library and Notes visibility to remain intact. Workspace switching changes Console operating context and staging eligibility, not whether users can browse, search, open, or edit their global content.

Users must not be forced to configure workspaces before basic chat works. At the same time, implicit default state must not accidentally grant filesystem read/write or agent tool access, because a prompt-injected chat should not inherit local disk authority merely because the app needed an operating context.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Start implementing a sync or hydration engine for server-backed workspaces | The server sync path is not complete, conflict/replay policy is not proven, and the user explicitly rejected treating sync as available or high priority. |
| Treat remote-only workspaces as switchable local workspaces | This would imply local materialization and risk false affordances. Remote-only states must remain visible but unavailable until an adapter can hydrate them. |
| Hide non-active workspace Library/Notes content | This violates the approved visibility-versus-eligibility contract and would make workspace switching feel like data disappearance. |
| Put ACP task/run handoff behind Settings or generic sync copy | ACP owns runtime/task/run setup. Settings only owns defaults. Console can surface readiness and audit details, but ACP handoff remains a future adapter-backed target. |
| Add a second workspace source of truth for Console | A parallel Console workspace registry would diverge from Library/Notes/workspace membership and make cross-screen eligibility harder to reason about. |
| Keep the no-workspace state as a display-only `Local Default` label | This made the UI appear usable while the underlying context remained `None`, causing inconsistent rail persistence and active-context eligibility. A real safe `Default` workspace is easier to reason about and test. |
| Allow runtime bindings on the built-in `Default` workspace | This would make the implicit workspace a privilege escalation path. Filesystem, worktree, container, VM, remote runtime, and ACP bindings require an explicit user-created workspace. |

## Consequences

Console can show local-only, server-unavailable, remote-only, conflict, runtime-missing, and ACP handoff states without performing sync. Local workspace switching remains usable through the existing registry fallback.

Handoff eligibility must be explicit per source or conversation: copy, reference, metadata-only, or local-only. Ineligible items remain visible but blocked from active Console staging until copied or linked into the active workspace.

Future server-backed workspace work must wire through an adapter that can report availability and hydration readiness before enabling actions. Future ACP task/run package migration can reuse the visible readiness/audit states introduced here instead of inventing another status contract.

The built-in `Default` workspace is local-only and not-configured for sync. It can hold normal chat/conversation context and keep saved conversations visible, but it is not portable and cannot own runtime bindings. Registry reads sanitize or hide stale Default runtime bindings so accidental storage drift cannot enable filesystem/tool authority. Users who need filesystem reads/writes, git worktrees, sandboxes, containers, VMs, remote runtimes, or ACP sessions must create or select an explicit workspace.

## Links

- [Backlog task TASK-86](../tasks/task-86%20-%20Console-workspace-switching-server-readiness-contracts.md)
- [Workspace operating context PRD](../../Docs/superpowers/specs/2026-05-20-workspace-operating-context-handoff-prd-design.md)
- [Workspace operating context implementation plan](../../Docs/superpowers/plans/2026-05-20-workspace-operating-context-implementation.md)
- [Server parity roadmap](../../Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md)
