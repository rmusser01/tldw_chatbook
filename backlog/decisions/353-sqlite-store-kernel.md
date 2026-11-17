# ADR-353: One SQLite store kernel for held connections and transactions

## Status

Proposed (TASK-33132; implementation in progress)

## Context

The DB layer grew by copy-paste convergence: 25 independent `transaction()` context managers and 10 verbatim copies of the held thread-local connection + liveness-ping idiom exist across the stores, because `DB/base_db.py` provides operation-owned-connection and quiescence machinery but none of what a store's own access path needs. The copies self-document their origin — `RAG_Indexing_DB.py` states it mirrors the `Workspace_DB` template; `AgentRuns_DB.py` says it follows the Workspace pattern; three stores carry `close()` aliases to feel like `BaseDB` without inheriting it. The count is growing, not shrinking (18 → 21 → 25 `transaction()` definitions across recent months), and each new copy risks re-learning lessons already encoded elsewhere:

- autocommit (`isolation_level = None`) is required on a held connection, or Python's implicit BEGIN makes the explicit one raise (task-15465/15466 audits),
- per-connection PRAGMAs (`synchronous`, `foreign_keys`) must be applied in the one place connections are created, not at call sites,
- transaction rollback must catch `BaseException`, not `Exception`, or an escaped `CancelledError`/`KeyboardInterrupt` leaves the transaction open and silently loses every later write on that thread (task-32801.5, proven on Prompts_DB),
- a liveness probe with careful invalidation handling must transparently replace a connection another component closed.

`Backup_Recovery/participants.py` (TASK-32628 lineage) already coordinates stores through the `core_getter`/`core_transaction` decorators and `core_*` hooks — but it observes whatever machinery each store still hand-rolls; it does not provide that machinery.

## Decision

Extract one kernel — `DB/store_kernel.py`, a `SQLiteStoreKernel` mixin consumable by both `BaseDB` subclasses and standalone stores — implementing the hardened Workspace/RAG pattern as canonical:

1. **Held connection**: per-thread `threading.local` connection, opened through a single overridable `_configure_connection` (the one place per-connection PRAGMAs and `isolation_level = None` live), with the 30-second liveness ping and the invalidated-connection replacement dance.
2. **`transaction()`**: `BEGIN IMMEDIATE`; body yields the held connection; `except BaseException: rollback; raise`; clean exit commits.
3. **Nesting is fail-loud**: a nested `transaction()` on the same thread raises `sqlite3.OperationalError`, with the rollback still landing through the outer block. This is the deliberate, documented semantics of the reference ports — not an accident to fix.
4. **Participants integration is built in**: the kernel's getter/transaction entry points carry the `core_getter`/`core_transaction` decoration and call `core_access`/`register_core_connection`/`core_cached_connection`, so every migrated store is observable by `Backup_Recovery/participants.py` by construction.
5. **`:memory:` policy is per-store**: the kernel stays uniformly thread-local (Workspace/RAG semantics); stores that need a shared in-memory connection (e.g. ClientNotificationsDB's inbox) override the accessor and say so in their docstring, as they do today.

**EventStateRepository is an ADR-353 exception**: its connections are keyed by thread ident in a plain dict rather than a ``threading.local`` because ``close()`` must reach connections it does not own (requiring ``check_same_thread=False`` on the file branch), and its ``:memory:`` branch deliberately shares ONE connection (an in-memory database lives inside its connection). Both properties conflict with the kernel's per-thread store; migrating it would regress worker-pool cleanup. ClientNotificationsDB demonstrates the other documented exception shape: it consumes the kernel for connection/transaction/close while keeping its ensure-schema lock, dual-owner open (``db.base`` file seam + ``notifications.client`` memory branch), and shared-connection ``:memory:`` policy as overrides — and gains the task-32801.5 BaseException rollback its old ``except Exception`` lacked.

ChaChaNotes' richer `TransactionContextManager` (depth-tracked nesting, quiescence integration) is **not** collapsed into the kernel: it remains the documented exception until another store demonstrates a need for nested transactions, at which point nesting support becomes a kernel feature rather than a second idiom. This refines TASK-33132's original AC #1 wording ("ChaChaNotes transaction semantics as canonical"): the fleet's de facto canonical semantics are the fail-loud variant above; ChaChaNotes keeps its own.

Migration is incremental, one store per PR, pilot first (`RAG_Indexing_DB` — the best-documented port, whose comments largely move into the kernel verbatim). Stores with genuinely different contracts (Chunking_Lab's durability protocol, Client_Media's chunk/blob layer) may consume the kernel without inheriting its policy.

### BaseDB-provenance stores (review round 1)

Most planned migrations subclass `BaseDB`, whose per-call `_get_connection()` opens through the **`db.base`** private-sqlite owner, and `register_core_connection` validates participants identity against exactly that policy. The kernel's `_kernel_sqlite_owner` is the *connect* owner for stores that open their own connections (like the pilot's `db.rag_indexing`); it is not a new participants identity, and no per-store `db.<name>` policy entries should be minted for BaseDB-provenance stores. For those stores:

- **MRO is load-bearing**: declare `class Store(SQLiteStoreKernel, BaseDB)` — kernel first — so the kernel's autocommit `_get_connection()`, `transaction()`, and the participants-gated `close()` win over `BaseDB`'s per-call/non-autocommit `_get_connection` and its no-op `close()` (which the retirement closers map calls).
- Where a store needs BaseDB's per-call connection for specific operations, it retains that seam explicitly rather than blending the two.
- Per-store semantic deltas surface at migration time and are decided individually. Deferred-BEGIN contracts are expressed as `_transaction_immediate_default = False` (a class attribute) rather than method overrides: `transaction(*, immediate=None)` resolves None to the store's class default, and an explicit value always wins (the tri-state, decided at migration 3 per the workspace-PR review's recommendation — five stores carry deferred-BEGIN contracts, and per-store method overrides would recreate the boilerplate the kernel deletes). Workspace uses the class attribute; AgentRuns keeps the kernel's immediate default with no override.

### Finite materialized Workspace reads (TASK-35582 AC23)

A value-returning Workspace registry read may wrap its complete synchronous work
in the existing `operation_owned_connection(self.db)` scope. `get_runtime_binding`
uses this same policy as `get_workspace` and `list_workspaces`; its scope includes
the legacy Default-binding repair and resulting transaction before retirement.
Only a newly acquired exact file-backed owner's handle is retired, on its creating
thread, after managed DB operations settle. Pre-existing caller handles and
transactions, in-memory/custom owners, and uncertain native closes retain their
existing ownership. Canceling an awaiter does not close its still-running producer.

This clarifies finite caller ownership under ADR-126; the kernel's held cache,
transaction and thread-affinity contracts do not change. TASK-35582's real Console
probe found seven Workspace handles surviving dead producer threads, all acquired
through this read. Generic agent worker cleanup's default-registry discovery did
not cover the actual injected registry, and a modal reader ran independently.
Fixing the shared finite read covers both without widening shutdown authority.

Two micro-pins deferred to follow-up before the fleet migrates (kernel review, minor): the probe-error asymmetry (a failed probe on a still-live connection must propagate, not replace) and `_configure_connection` applying on revival. Both hold in the current code; they deserve dedicated tests in the next kernel PR.


## Consequences

- Each migration deletes a store's private copy of the held-connection/transaction machinery (~120–180 LOC per store) and makes participants wiring non-optional-by-omission.
- New stores get the incident-hardened semantics (BaseException rollback, autocommit, PRAGMA placement) without re-learning them; the lessons live in one auditable module.
- Reviewers of future store PRs can reject hand-rolled `def transaction()` as a regression against this ADR.
- Risks: the mixin must not perturb `BaseDB`'s operation-owned-connection/quiescence machinery (different concern, per-call vs held); migrating a store changes its failure surface only if its copy had drifted from canonical — each migration's test suite is the guard, and drifts are surfaced and decided individually.
- Rejected alternatives: (a) collapsing ChaChaNotes' depth-tracked manager for everyone — larger blast radius, no demonstrated need; (b) making participants.py the kernel — it is a coordinator, and overloading it couples backup admission with connection mechanics; (c) leaving the copies — the count grows monotonically and each new copy is a fresh chance to forget task-32801.5.


## TASK-35582 AC23-B: retained DB custody during private root preparation

An exact core DB operation cannot grant filesystem authority. The fleet launch
retains its original counted AgentRuns operation/lease through child handoff or
refused-launch terminalization/refund. A no-argument Backup_Recovery helper admits
only the existing profile `agent_worktrees` base through ordinary native storage
acquisition, suspending core discovery inside that owner and restoring/revalidating
it in `finally`. Pause/source/private-path guards and `_Operation.check` are
unchanged. This follows existing retained-core raw-source scopes; generic path
widening, dropping the launch operation and caller-selected callbacks are rejected.

The full decision and required native pause/symlink/unrelated-path/refused-child/
uncertain-custody qualification are in
[ADR-126's TASK-35582 amendment](126-complete-local-backup-and-recovery.md#task-35582-ac23-b-independently-prepare-the-fixed-agent-worktree-root).
Creation translates exactly `RecoveryRequired`/`PrivatePathError` from this fixed
preparation into body-free `storage_unavailable`, allowing parent DB cleanup;
helper callers keep native exceptions and no private exception body is logged.

## Renumbering provenance

Renumbered from ADR-353 to ADR-353 on 2026-11-03 under TASK-36943.3 (owner decision 2026-10-25: the younger file of a shared number is renumbered). ADR-228 now refers to 228-config-encryption-derived-key-memoization.md.

## TASK-40608: existing-schema Workspace command connections

A contained project-index key reader may construct `WorkspaceDB` with explicit keyword-only `initialize_schema=False, must_exist=True`, leaving normal defaults unchanged. Forward the first option to the existing BaseDB constructor and the second through its existing connection keywords to `connect_private_sqlite`; retain `db.base` provenance, kernel registration/configuration and all private-path/native checks. The reader validates the expected schema through that admitted connection and refuses missing/outdated stores without file creation, DDL or migration. It uses the same original registry query/transaction and positively retires the exact native connection/core owner, including configuration-failure births. No raw SQLite shortcut or new per-store policy is authorized. The owning process and complete callback contract is specified by ADR126/230 and the [key-owner design](../../Docs/superpowers/specs/2026-11-14-task-40608-project-index-key-owner-design.md).

### TASK-40608 — Console execution-service ownership (2026-11-15)

Under the [Console execution-service contract](../../Docs/superpowers/specs/2026-11-15-task-40608-console-execution-service-design.md), same-filename parent and backend database instances remain separate registered owners. No cross-process core_operation borrowing or connection/pin transfer is introduced. Original transactions, native/core retirement, and failure custody remain; existing-schema/must-exist options apply only at explicitly selected attachment boundaries, while normal lifecycle initialization retains its migration ownership. The first in-process composition extraction changes no database ownership.

### TASK-40608 Task61: retained local display and settings phases (2026-11-15)

The selected execution-service extraction uses fresh physical display phases around every original parent-only character checkpoint. One logical FULL retains its sealed character → context → appearance section set and original observers across lease-free gaps. Each metadata-pair midpoint, scope-owner/error-classification check and post-groups check keeps its original order, branch, exception and three-attempt budget. Native connection/core/producer ownership must positively retire before a parent checkpoint wait; the next phase obtains fresh admission and source checks. No section publishes before all issued physical work settles; admission or retirement uncertainty renews none. The existing same-handle legacy composition and borrowed transactions remain unchanged. This explicitly qualifies the earlier one-callback batching permission; it is not a claim that the new phase boundary was already present.

The backend owns actual domain sources and the existing authoritative display state; the actual attached view owns attachment/visit/teardown and presentation checks. No fake Screen/App, copied authority, second cache, generic callback transport, eager warm, longer reconciliation interval or skipped original midpoint is permitted. Parent archive admission retains its real App inflight/generation/cache owner, bracketing the complete original native archived-state read with fixed checkpoints after physical retirement.

Selected session settings retain one exact pending operation through the original live pre-write → full metadata snapshot attempt → settings-revision/context post-write sequence. Short preparation-lock phases surround the original retained native writer. Same-session coherent capture waits asynchronously; Cancel, decisions and source fences remain dispatchable. Exact fork-transition and global voice admission ownership remains balanced, without thread-local admission across await. The existing session lifecycle orders full snapshots with component CAS drain. Writer exception/False preserves original applied live state and does not certify durable success. The source effect includes required post-publication and exact native settlement; legacy synchronous composition remains available on its existing route.

See `.superpowers/sdd/2026-11-13-task-40608-sub-five/task-61-display-owner-plan.md` for the finite owner/phase inventory and the canonical execution-service design for the release contract. Local causal qualification is required; no process route, production selection or sub-five result follows from this amendment.
Task61 bounded convergence clarification (root ruling 2026-11-15): readiness refresh's original post-native pristine-default convergence remains before readiness key recomputation/publication. Fixed parent prepare/finish checkpoints retain the actual session controller's pristine memo, convergence spans, explicit-default generation, projection guard and provider notification. The backend performs only the exact captured automatic settings replacement through the retained pending pre/native/post owner, revalidating original session/incarnation/settings revision/config/source. Preserve follows_saved_defaults, settings=None, persona preservation and all original short-circuits (ADR095 policy unchanged). Legacy composition remains synchronous. A convergence nested inside the current ReadProjection uses that command's actual source inline; it cannot enqueue behind itself. No native lease spans a checkpoint, no second memo/span map is created, and failed/stale phases cannot publish a success/span/notice.
Task61 selected parent-settings continuation clarification (root ruling 2026-11-15): actual early FULL core-state sync and autonomous readiness refresh can reach original default convergence without a user command. They must acquire an explicitly registered ParentSettingsCapture from the same live controls/port/runtime graph, with exact real parent stamp, source revision/session/incarnation and a fixed named settings purpose. They do not invent ControlIdentity or alter external sequence/require_control_source. Only the settings resolver/effect path accepts this private registered owner; provider/tool/MCP/control authority remains unchanged. Reuse existing same-session capture tails, finite effects, task custody and settings pre/native/post operation. Register before exposure, revalidate after waits, and retire/unregister only on actual settlement, including no-entry/scheduling failure. Command-owned nested convergence remains inline. Preserve early core-state sync before later main-display/archive/readiness preparation; no early readiness warm.

Selected synchronous prompt/provider actions must defer the complete owning action when settings are pending; async actions wait before coherent capture. Capture original arguments/origin/session/incarnation/attachment, preserve action order without action coalescing, and retain original stale/refusal/notice outcomes. Pure render work retains one owed replay under its original coalescing policy. No mixed-state reads, copied settings fallback, arbitrary callback dispatcher, native lease across wait, or success before completion. Legacy synchronous composition remains original. Exact entrance map is task61-settings-parent-entrances.md; normal production selector/process integration remains outside Task61.


Guarded prompt transaction compensation ruling (implementation pending): keep original composer checkpoint→draft→component write order in one exact parent transaction Task/effect. Only an exception escaping that original mutation transaction triggers its fixed compensation; original False, converted-to-False writer failure and display-only failure never roll back. After the first operation is physically confirmed retired, exact transaction-owned resulting state/revisions/source/session/incarnation/binding/definitions may authorize old-session backend draft/settings restoration under the same effect. Do not overwrite intervening changes. Composer/undo/UI restoration additionally requires original composer+visit; use retained original checkpoint data for old backend restoration, never replacement-view state. Retirement uncertainty or changed source/owner cannot authorize another write. Library claim acknowledgment follows full transaction/compensation; legacy sync bool stays original. No generic rollback authority or external command.
