# PR3341 managed-transaction-depth integration

**Goal:** Integrate the public managed-depth accessor and move remaining affected repository callers to it while preserving real SQLite transaction ownership.

**Architecture:** Expose existing thread-local managed depth on CharactersRAGDB. Native in_transaction remains an independent test; no kernel conversion or new transaction policy.

**Spec:** TASK36949.1, existing ADR228 and ADR126; detailed preparation `.superpowers/preparations/pr3341/repair-plan.md` and source review `C:/Users/GDesktop-1/.codex/worktrees/pr3298-resync/pr3341-initial-review.md`. Exact refreshed source and immutable integration base are bound by the selection ledger.

ADR required: no
ADR path: existing backlog/decisions ADR228 and ADR126 (resolve exact paths)
Reason: Public accessor and caller migration preserve established transaction/admission ownership.

## Global constraints

- Root alone refreshes/disarms/publishes the selected PR. No remote actions, drafts,3267/3297,release/maincheckout work. User substitutes independent requesting-code-review forQodo and waives hostedCI/numeric size; security/custody/source correctness and original caps remain.
- One implementer owns local source/index/tests/imports/collection/generators, no delegation. Rebase all selected source commits, narrowly reconcile current schema85/deadlines/admission/native retirement/stream provenance. No older whole-file replacement, kernel port or broad cleanup.
- Read applicable AGENTS/task/architecture/ADRs/lessons and current fixture conventions. Use existing read-only main .venv interpreter, private fixture ownership runner and unique per-process roots; no dependency changes or guard bypass. Preserve full tool results, session IDs and explicit exits. UTF8, stop on native command errors, normal hooks.
- Targeted changed-contract tests only, no full DB/Notes suite or historical author sweep. Separate actual PASS, inherited failures, skips and numeric WAIVED. Before/after source and selected fixture closure hashes; private raw logs/config stay private.

## Task 1: Integrate accessor and qualify affected ownership

1. Rebase refreshed source onto exact selected dev, retain all commits/current contributions and accumulated notes. Record conflicts. Add distinct Implementation Plan/ADR disposition through Backlog CLI before any repair, preserving historical notes.
2. Inspect current private-depth callers and migrate only affected external reads to managed_transaction_depth(). In Notes/note_folder_repository.py preserve independent native transaction state, SQLITE_BUSY_SNAPSHOT-only retry, whole-owned-operation bounds and borrowed transaction no-retry. Do not replace accessor internals or native-handle checks.
3. Run the new real-DB nesting/rollback/thread control, affected ActorPack/PersonaVisual guard cases and the five named Notes controls in the detailed preparation, including actual WAL snapshot restart. Add a narrowly causal public-accessor linkage control only if existing cases cannot distinguish the migrated caller; no implementation-mirroring tests or broad suite.
4. Check changed Python lint/format, remaining external private-depth reads and full BASE..HEAD whitespace. Regenerate/validate diagnostic inventory only if actual statements require it; review all resulting deltas. No pin/cap raise or unrelated generated churn. Attribute inherited failures to exact base if necessary, not old author reports.
5. Record supported task notes via CLI, commit with normal hooks, and supply clean finalHEAD, retained-commit/conflict record, selected commands/case identities/results, raw versus Git blob manifest and full binary patch digest. Explicitly drain/release all processes and ownership. Root obtains independent whole-branch review and fresh eligibility before publishing.
