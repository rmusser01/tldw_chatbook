# Buddy PR review and merge

Task: TASK-32084
Spec: Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md
ADR required: no new ADR
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: corrections within existing ownership, runtime and interaction contracts.

## Global Constraints

- Preserve independent artwork/Persona identity, exact explicit target, one visible Buddy, navigation continuity, workspace no-microphone and guarded speech.
- Verify external review claims; do not blindly follow their suggested implementation. Preserve central security and transaction/thread ownership, bounded resource behavior, attribution and public compatibility.
- No full local test sweep, live provider/microphone/audio or production profile changes. Run affected regressions and existing CI checks. Never bypass branch protection or ignore a required failing check.
- Work only in this isolated worktree. Implementer owns assigned production/tests; root owns Backlog/docs/derived artifacts, commits, pushes and merge. No nested agents.

## Task 1

Verify all eleven Chatbook Qodo comments and correct confirmed validation, blocking I/O, database, speech-order and import-provenance issues with focused tests. Root owns derived artifact/allowlist CI repair and publication.

1. Read AGENTS, the listed task/spec/ADR and relevant testing lessons. Reproduce each suspected defect or establish source evidence for a non-applicable suggestion.
2. Make the smallest corrections and add behavior-focused tests. Preserve UI-thread publication after background reads and revalidate authority after awaits.
3. Run affected tests/static checks, freeze source, and record commands/results and every review disposition.
4. Independent review before root commits/pushes; resolve Important/Critical findings with scoped rechecks.

## Root integration

Confirm latest dev and rebase, retaining independent conflict additions. Run required derived checks after source is frozen, verify new PR head/checks/Qodo state, then merge only once the requested gates are met. The explicit user request authorizes rebase, lease-protected feature-branch updates and merge; no merge to a different base or administrative bypass.
