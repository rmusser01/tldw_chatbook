---
id: TASK-23088
title: Fork Console chat from a selected message
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 21:00'
updated_date: '2026-08-27 13:56'
labels:
  - console
  - chat
  - ui
  - persistence
references:
  - Docs/superpowers/specs/2026-08-26-console-chat-fork-design.md
  - Docs/superpowers/plans/2026-08-26-console-chat-fork.md
  - backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a user create and immediately open an independently owned Console chat copied through one selected stable message while leaving the source chat and all of its live and durable state unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eligible selected USER and ASSISTANT messages expose Fork immediately before Regenerate plus the `f` action, and a compact naming dialog clearly identifies the boundary, saved or temporary destination, exclusions, validation, progress, cancellation, and degraded-success recovery.
- [x] #2 Confirming copies exactly the canonical active lineage through the selected boundary and its fenced visible text or generated-image choices with fresh mutable ownership, while off-path, later, display-only, unsettled, unsaved-durable, and unsupported state is rejected or excluded as designed and the source remains byte-for-byte and live-state unchanged.
- [x] #3 Durable forks commit conversation ancestry, messages, supported sidecars, active leaf, policy, governed citation owner links, and sanitized project context atomically before publication; temporary forks remain detached and sanitized, and a non-ephemeral source without durable IDs produces a saved independent-root fork without persisting the source.
- [x] #4 Forks preserve declarative Workspace, model, role, Library, RAG, and project-instruction selections without copying scratch, approvals, permissions, resolved instruction bodies, continuations, recovery, derived context, usage, tool activity, or ephemeral video authority; citation and media degradation remains truthful.
- [x] #5 One preallocated conversation or session identity makes retries idempotent, precommit failure creates nothing, and postcommit publication or activation failure identifies and reopens the already-created fork without duplication.
- [x] #6 The USER and ASSISTANT action row uses the approved stable direct actions and labelled `More…` menu with captured message targeting, safe teardown, and deterministic focus fallback at 80x24 and wider production-shaped layouts.
- [x] #7 Targeted domain, real-SQLite persistence, authority, media, cancellation-race, action/menu, modal, reload, and live local TUI verification pass, and Console user documentation describes the boundary, temporary behavior, shortcut, exclusions, and video/citation caveats.
<!-- AC:END -->

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md`

Reason: ADR-092 already governs the durable copy, identity, authority, and publication boundaries; this task implements that accepted contract without a schema migration or new ADR.

1. Define the pure allowlisted fork projection, title rules, eligibility, and sanitized project-instruction contract.
2. Fence and revalidate the canonical active-lineage prefix plus selected generated-image state, then stage fresh independent ownership without source mutation.
3. Add one idempotent real-SQLite bundle for ancestry, messages, supported sidecars, policy, governed citations, project context, and active leaf.
4. Add the direct Fork action, captured-target More menu, media-card controls, compact six-state modal, and cancellable controller orchestration.
5. Verify atomic failure, cancellation races, reload, layout/focus, temporary promotion, source immutability, and the provider-free live journey; update user docs and task evidence.

Detailed TDD steps and commands: `Docs/superpowers/plans/2026-08-26-console-chat-fork.md`.

## Implementation Notes

Implemented the ADR-092 fork projection and durable bundle, Console orchestration,
stable direct action row and captured-target `More…` menu, six-state naming modal,
temporary-fork promotion rules, citation/media handling, and authority recreation. The
copy contract is an explicit allowlist with fresh mutable owners: it preserves supported
declarative settings and sidecars while excluding live execution and authority state;
no schema migration or new core dependency was needed.

Task 7 added `Tests/integration/test_console_chat_fork_flow.py` and updated the Console
chat-basics, branching-and-rewind, and design-specification documentation. The journey
uses a pytest temporary directory as a fresh `HOME`, `XDG_CONFIG_HOME`,
`XDG_DATA_HOME`, and `XDG_CACHE_HOME`; a file-backed SQLite database at
`isolated-profile/data/chatbook.sqlite`; a temporary named Workspace and bound project
with project instructions; the real citation repository; and the production
`ChatScreen`/Console hierarchy with its consolidated stylesheet. It seeds saved
branching, governed citations, attachments, a generated-image selection, generated
video, Workspace/project, Library, RAG, approvals, recovery, and scratch state without
calling a provider or reading developer data.

At `120x35`, the journey pointer-selects a middle USER message, opens Fork with `f`,
verifies a painted naming modal, replaces the selected default title, confirms with
Enter, then switches through source/fork tabs and uses the pointer Fork button at a
middle ASSISTANT generated-image boundary. At `80x24`, it renders and confirms a
temporary video-boundary fork, verifies no durable write, preserves textual `[S1]`
while omitting citation ownership, emits a path-free video tombstone, and promotes the
fork as an independent durable root. A new production app/store then reopens the same
database at `120x35` and switches through every restored fork. Assertions prove exact
inclusive prefixes and ancestry, active leaves, generated-image/attachment/citation
ownership, source durable and live immutability, fresh scratch identity and leases,
fresh project-binding resolution, and absence of approvals, grants, recovery, runs,
instruction bodies, raw paths, attachment bytes, and provider secrets from fork state,
render snapshots, notifications, and captured logs.

The exact targeted regression command was:

```bash
../../.venv/bin/python -B -m pytest \
  Tests/integration/test_console_chat_fork_flow.py \
  Tests/Chat/test_console_chat_fork.py \
  Tests/Chat/test_console_chat_fork_persistence.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_edit_message_modal.py \
  Tests/Chat/test_console_regenerate_branching.py \
  Tests/UI/test_console_fork_chat_modal.py \
  Tests/UI/test_console_native_transcript.py \
  Tests/UI/test_console_message_controller.py \
  Tests/UI/test_console_session_controller.py -q
```

Result: `663 passed, 2 warnings in 206.79s`; the warnings were the existing requests
dependency-version warning and Python's `audioop` deprecation through pydub. No full
suite was run. Changed-file Ruff check/format and `git diff --check` passed; Ruff made
three layout-only changes in the new fork region of
`tldw_chatbook/UI/Console_Modules/session.py`, with no behavior change. Self-review
against ADR-092 found no source mutation, shared mutable owner, authority transfer,
non-atomic durable publication, duplicate retry, or narrow-layout regression. The
five-digit task was marked Done by direct file edit because the Backlog CLI cannot
reliably address it.

ADR required: yes

ADR path: `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md`

Reason: ADR-092 is the governing copy, identity, authority, atomicity, and publication
contract implemented by this task; no new decision was introduced. No lessons entry
was added because the task did not surface a new reusable repository incident.

Immediately before closeout, the uniqueness sweep checked 787 local/remote refs and
233 worktrees. It found one committed-ref occurrence and two worktree occurrences of
`id: TASK-23088`, all at this canonical task path, with zero competing ID claimants and
zero same-title tasks at another path.
