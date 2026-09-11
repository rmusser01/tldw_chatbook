# Console archive recovery implementation plan

> For agentic workers: use subagent-driven-development for independent modules, then integrated review.

**Goal:** Complete archive, search, transcript review, restore and original-conversation continuation for novice and power users.
**Architecture:** Database-owned conversation archive state, bounded Library reader, dedicated original-ID Console resume handoff, shared workspace restore naming. Widgets expose explicit lifecycle actions with receipts.
**Tech stack:** Python >=3.12, Textual >=8,<9, SQLite; existing dependencies.
**Spec:** Docs/superpowers/specs/2026-09-10-console-archive-recovery-design.md

ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: durable lifecycle storage, service contracts and cross-screen resume identity.

## Global constraints

- Preserve the original dirty checkout. PR integration uses isolated branch `codex/console-archive-recovery` from `origin/dev` at `068535986ed005e85045cf6e08397c372377cb28`.
- Targeted tests only. Existing review checks provide baseline evidence; capture further baselines per modified module.
- No provider calls needed for ordinary archive/review. No new dependencies. Parameterized SQL and optimistic concurrency. Off-loop I/O; no new terminal-convention keybindings.
- User authorized a PR against dev; commit and push only the isolated feature branch.

## Tasks

- [x] TASK-32300: DB/service archive lifecycle. Own ChaChaNotes_DB.py, migration, Chat conversation services and DB/Chat tests. Red/green real SQLite, scope pagination and migration checks. Expose `archive_scope`, `set_conversations_archived` and per-ID archive state.
- [x] TASK-32274: Workspace recovery and close clarity. Own workspace registry, Console workspace switcher/lifecycle controller sections, Settings workspace sections and close copy in session.py; tests. Red/green collision, rendered state, restore-without-activate, busy guards and receipt recovery.
- [x] TASK-32275: Library conversation workflow. Own Library conversation state/canvas, dedicated Library controller module and minimal library_screen wiring; tests. Active/Archived/All, individual/bulk actions with Undo, bounded transcript review/find, Resume versus source, responsive layout and focus.
- [x] TASK-32276: Console integration and exact resume. Own app.py, pending_handoff_store.py, chat_screen.py, new Console archive controller/controls and session switcher full-search entry; tests/docs. Original-ID branch-preserving activation, dedupe, workspace disclosure, busy/draft guard and history filtering.
- [x] Integrated review and verification: exercise lifecycle end-to-end, remedy cross-module gaps, inspect compact/wide frames, run targeted lint/format, update user guide and task notes/AC/status.

## Progress and rulings

- User approval covers the reviewed design and all A1–A9. No repeated design gate required.
- Current checkout contains extensive concurrent changes; preserve it and compare against a private baseline rather than moving or committing those changes.
- Agents own distinct files/sections; parent owns cross-module contract and integration.

Final verification: Docs/superpowers/qa/console/2026-09-10-archive-recovery.md. Implementation and PR integration complete; targeted verification and historical-test limitations are recorded in the QA report.

## PR integration

ADR required: yes
ADR path: backlog/decisions/147-conversation-archive-and-exact-resume.md
Reason: same approved archive lifecycle; renumbered after checking current dev and open PRs.

Port only feature deltas to current dev, preserving its schema history, asynchronous Console hydration, switcher modes, and canonical Library Reader. Use migration 70→71. Reuse the current Reader for review/Find instead of adding a parallel preview. Re-run targeted backend, workspace, Library, exact-resume and send tests; then create the requested PR. Task IDs were initially reassigned to 32273–32276 because dev had claimed the original range. The lifecycle task subsequently moved to TASK-32300 after the landed reasoning-history task claimed 32273; the other archive tasks remain 32274–32276.
