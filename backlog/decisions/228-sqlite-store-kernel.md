# ADR-228: One SQLite store kernel for held connections and transactions

## Status

Proposed (TASK-33132; implementation in progress)

## Context

The DB layer grew by copy-paste convergence: 25 independent `transaction()` context managers and 10 verbatim copies of the held thread-local connection + liveness-ping idiom exist across the stores, because `DB/base_db.py` provides operation-owned-connection and quiescence machinery but none of what a store's own access path needs. The copies self-document their origin — `RAG_Indexing_DB.py` states it mirrors the `Workspace_DB` template; `AgentRuns_DB.py` says it follows the Workspace pattern; three stores carry `close()` aliases to feel like `BaseDB` without inheriting it. The count is growing, not shrinking (18 → 21 → 25 `transaction()` definitions across recent months), and each new copy risks re-learning lessons already encoded elsewhere:

- autocommit (`isolation_level = None`) is required on a held connection, or Python's implicit BEGIN makes the explicit one raise (task-15465/15466 audits),
- per-connection PRAGMAs (`synchronous`, `foreign_keys`) must be applied in the one place connections are created, not at call sites,
- transaction rollback must catch `BaseException`, not `Exception`, or an escaped `CancelledError`/`KeyboardInterrupt` leaves the transaction open and silently loses every later write on that thread (task-32801.5, proven on Prompts_DB),
- a liveness probe with careful invalidation handling must transparently replace a connection another component closed.

`Backup_Recovery/participants.py` (TASK-32628 lineage) already coordinates stores through the `_core_getter`/`_core_transaction` decorators and `_core_*` hooks — but it observes whatever machinery each store still hand-rolls; it does not provide that machinery.

## Decision

Extract one kernel — `DB/store_kernel.py`, a `SQLiteStoreKernel` mixin consumable by both `BaseDB` subclasses and standalone stores — implementing the hardened Workspace/RAG pattern as canonical:

1. **Held connection**: per-thread `threading.local` connection, opened through a single overridable `_configure_connection` (the one place per-connection PRAGMAs and `isolation_level = None` live), with the 30-second liveness ping and the invalidated-connection replacement dance.
2. **`transaction()`**: `BEGIN IMMEDIATE`; body yields the held connection; `except BaseException: rollback; raise`; clean exit commits.
3. **Nesting is fail-loud**: a nested `transaction()` on the same thread raises `sqlite3.OperationalError`, with the rollback still landing through the outer block. This is the deliberate, documented semantics of the reference ports — not an accident to fix.
4. **Participants integration is built in**: the kernel's getter/transaction entry points carry the `_core_getter`/`_core_transaction` decoration and call `_core_access`/`_register_core_connection`/`_core_cached_connection`, so every migrated store is observable by `Backup_Recovery/participants.py` by construction.
5. **`:memory:` policy is per-store**: the kernel stays uniformly thread-local (Workspace/RAG semantics); stores that need a shared in-memory connection (e.g. ClientNotificationsDB's inbox) override the accessor and say so in their docstring, as they do today.

**EventStateRepository is an ADR-228 exception**: its connections are keyed by thread ident in a plain dict rather than a ``threading.local`` because ``close()`` must reach connections it does not own (requiring ``check_same_thread=False`` on the file branch), and its ``:memory:`` branch deliberately shares ONE connection (an in-memory database lives inside its connection). Both properties conflict with the kernel's per-thread store; migrating it would regress worker-pool cleanup. ClientNotificationsDB demonstrates the other documented exception shape: it consumes the kernel for connection/transaction/close while keeping its ensure-schema lock, dual-owner open (``db.base`` file seam + ``notifications.client`` memory branch), and shared-connection ``:memory:`` policy as overrides — and gains the task-32801.5 BaseException rollback its old ``except Exception`` lacked.

ChaChaNotes' richer `TransactionContextManager` (depth-tracked nesting, quiescence integration) is **not** collapsed into the kernel: it remains the documented exception until another store demonstrates a need for nested transactions, at which point nesting support becomes a kernel feature rather than a second idiom. This refines TASK-33132's original AC #1 wording ("ChaChaNotes transaction semantics as canonical"): the fleet's de facto canonical semantics are the fail-loud variant above; ChaChaNotes keeps its own.

Migration is incremental, one store per PR, pilot first (`RAG_Indexing_DB` — the best-documented port, whose comments largely move into the kernel verbatim). Stores with genuinely different contracts (Chunking_Lab's durability protocol, Client_Media's chunk/blob layer) may consume the kernel without inheriting its policy.

### BaseDB-provenance stores (review round 1)

Most planned migrations subclass `BaseDB`, whose per-call `_get_connection()` opens through the **`db.base`** private-sqlite owner, and `_register_core_connection` validates participants identity against exactly that policy. The kernel's `_kernel_sqlite_owner` is the *connect* owner for stores that open their own connections (like the pilot's `db.rag_indexing`); it is not a new participants identity, and no per-store `db.<name>` policy entries should be minted for BaseDB-provenance stores. For those stores:

- **MRO is load-bearing**: declare `class Store(SQLiteStoreKernel, BaseDB)` — kernel first — so the kernel's autocommit `_get_connection()`, `transaction()`, and the participants-gated `close()` win over `BaseDB`'s per-call/non-autocommit `_get_connection` and its no-op `close()` (which the retirement closers map calls).
- Where a store needs BaseDB's per-call connection for specific operations, it retains that seam explicitly rather than blending the two.
- Per-store semantic deltas surface at migration time and are decided individually. Deferred-BEGIN contracts are expressed as `_transaction_immediate_default = False` (a class attribute) rather than method overrides: `transaction(*, immediate=None)` resolves None to the store's class default, and an explicit value always wins (the tri-state, decided at migration 3 per the workspace-PR review's recommendation — five stores carry deferred-BEGIN contracts, and per-store method overrides would recreate the boilerplate the kernel deletes). Workspace uses the class attribute; AgentRuns keeps the kernel's immediate default with no override.

Two micro-pins deferred to follow-up before the fleet migrates (kernel review, minor): the probe-error asymmetry (a failed probe on a still-live connection must propagate, not replace) and `_configure_connection` applying on revival. Both hold in the current code; they deserve dedicated tests in the next kernel PR.


## Consequences

- Each migration deletes a store's private copy of the held-connection/transaction machinery (~120–180 LOC per store) and makes participants wiring non-optional-by-omission.
- New stores get the incident-hardened semantics (BaseException rollback, autocommit, PRAGMA placement) without re-learning them; the lessons live in one auditable module.
- Reviewers of future store PRs can reject hand-rolled `def transaction()` as a regression against this ADR.
- Risks: the mixin must not perturb `BaseDB`'s operation-owned-connection/quiescence machinery (different concern, per-call vs held); migrating a store changes its failure surface only if its copy had drifted from canonical — each migration's test suite is the guard, and drifts are surfaced and decided individually.
- Rejected alternatives: (a) collapsing ChaChaNotes' depth-tracked manager for everyone — larger blast radius, no demonstrated need; (b) making participants.py the kernel — it is a coordinator, and overloading it couples backup admission with connection mechanics; (c) leaving the copies — the count grows monotonically and each new copy is a fresh chance to forget task-32801.5.
