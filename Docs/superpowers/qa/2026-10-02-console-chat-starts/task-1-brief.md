# Binding context


- “The new destination and mode arguments belong only to `new_chat`.”
- “Retain the existing 120-character title cap and 20,000-character per-field prompt/instructions caps.”
- `destination=same_workspace|casual`; `mode=draft|start`; defaults are `same_workspace` and `draft`.
- “Both modes keep the user's current chat, active workspace, composer, and focus intact.”
- Casual persistence is `scope_type=global`, `workspace_id=NULL`; the session uses `CONSOLE_GLOBAL_WORKSPACE_ID`.
- Destination defaults and explicit standing instructions determine the fresh assistant; source identity, bindings, staged inputs and grants do not transfer.
- Both grant caches use requesting session incarnation, tool, resolved destination identity and mode. Denial ceilings remain per run/tool.
- “Each accepted chat start consumes one automatic generation; draft-only creation consumes none.”
- “Start attempts and fleet wakes share the existing automatic-primary admission limit and manual capacity reserve.”
- “Capacity refusal leaves a draft; this feature adds no waiting/retry timer.”
- AgentRunsDB acceptance is the ownership cutoff. Both durable fences must succeed before dispatch or `started`.
- Human decisions or paused preparation before acceptance return `not_started`; interrupted or uncertain starts never replay automatically.
- Version-2 handoffs persist edits/clears until accepted consumption or explicit discard. Unversioned handoffs keep the legacy contract.
- Bodies stay in private conversation storage; budget tables, badges and generic diagnostics carry bounded provenance/reasons only.
- Targeted tests only. A full sweep requires user opt-in. UI changes follow `backlog/docs/design-language.md` and its existing tokens.
- Planning changes no application code. At execution, inspect the dirty checkout and attached worktrees, then use a suitable isolated checkout under the worktree skill; preserve unrelated changes.


## Task 1: Shared allowance and native start-attempt foundation

**Backlog:** [TASK-33804](../../../../backlog/tasks/task-33804%20-%20Console-shared-automatic-allowances-and-chat-start-attempts.md).

**Independently reviewable outcome:** The ledger can represent cross-chat
automatic descendants without widening conversation/run scope. No tool/UI route
launches them yet; existing roots and wake behavior continue to work.

**Files:**

- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (`AUTOMATIC_WORK_SCHEMA`, version migrations, `create_run` scope guards).
- Modify: `tldw_chatbook/DB/automatic_work.py` (all admission/accounting/recovery methods).
- Modify: `tldw_chatbook/Agents/automatic_work_budget.py`, `automatic_work_runtime.py`.
- Create: `tldw_chatbook/DB/migrations/agent_runs_v18_to_v19_chat_starts.sql` at the current schema version.
- Create: `Tests/DB/test_automatic_chat_starts.py`.
- Modify: `Tests/DB/test_automatic_work_budget.py`, `test_automatic_work_deadlines.py`, `test_automatic_work_migration.py`, `test_automatic_wake_attempts.py`.
- Modify: `Tests/Chat/test_automatic_work_lineage.py` for accepted-context and same-conversation ownership evidence.

### Interfaces produced

Add `AutomaticChatStartAttempt` to `Agents/automatic_work_budget.py`:

```python
@dataclass(frozen=True)
class AutomaticChatStartAttempt:
    id: str
    source_run_id: str
    source_chain_id: str
    chain_id: str
    conversation_id: str
    session_id: str
    session_incarnation: str
    owner_id: str
    draft_revision: int
    context_epoch: int
    request_fingerprint: str
    generation_reservation_id: str
    state: str
```

On `AutomaticWorkLedger`, add these exact interfaces:

| Method | Contract |
| --- | --- |
| `allowance_root(chain_id: str) -> str` | Return the direct canonical root ID; do not change scope or renew limits |
| `prepare_chat_start(*, attempt_id: str, source_run_id: str, target_conversation_id: str, target_session_id: str, target_session_incarnation: str, owner_id: str, draft_revision: int, context_epoch: int, request_fingerprint: str, limits: AutomaticWorkLimits | None = None) -> AutomaticChatStartAttempt` | Derive source chain from the persisted run; validate current owner; atomically create the target member, reserve generation 1, and insert the exact attempt |
| `read_chat_start_attempt(attempt_id: str, *, owner_id: str) -> AutomaticChatStartAttempt` | Validate exact persisted owner and return a body-free projection |
| `accept_chat_start(attempt_id: str, *, owner_id: str, limits: AutomaticWorkLimits | None = None) -> bool` | Atomic prepared→accepted cutoff and generation commit; only the first `True` can proceed to the conversation fence |
| `abort_chat_start(attempt_id: str, *, owner_id: str) -> bool` | Release only a still-prepared, proven uncommitted reservation |
| `complete_chat_start(attempt_id: str, *, owner_id: str) -> bool` | Settle accepted attempt once; no wake claim or delivery mark |

Append `attempt_kind: Literal["wake", "chat_start"] = "wake"` to
`AutomaticWorkContext`. Use the kind to select the corresponding ledger reader
in both `mark_accepted()` and `check()`; all existing wake constructors retain
their default. Do not set its acceptance latch until the conversation receipt has
also committed. Source run lineage stays in the attempt; target primaries have no
cross-conversation `parent_run_id`.

### Steps

- [ ] **1. Capture current versions and targeted baseline.** Read `_CURRENT_SCHEMA_VERSION` in both DB owners and the current task. Run the existing four DB budget/deadline/migration/wake-attempt files and `Tests/Chat/test_automatic_work_lineage.py`. Save existing failures as baseline evidence; fix only regressions introduced by this task.
- [ ] **2. Add a real-SQLite failing test for native attempts and the shared last generation.** Reuse `db` and `chain` from `Tests/DB/test_automatic_work_budget.py`; define this helper in the new test file, then run its test and confirm the missing interface fails:

```python
from Tests.DB.test_automatic_work_budget import _automatic_work_db as db, chain
from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
import pytest

def prepare(db, source_run, target, attempt):
    return db.automatic_work.prepare_chat_start(
        attempt_id=attempt,
        source_run_id=source_run,
        target_conversation_id=target,
        target_session_id=f"session-{target}",
        target_session_incarnation=f"incarnation-{target}",
        owner_id="owner",
        draft_revision=1,
        context_epoch=0,
        request_fingerprint="a" * 64,
    )

def test_two_targets_share_the_last_generation(db):
    root = chain(db, conversation="source", generations=1)
    source = db.create_run(
        conversation_id="source", agent_kind="primary", work_chain_id=root
    )
    first = prepare(db, source, "target-a", "attempt-a")
    assert db.automatic_work.allowance_root(first.chain_id) == root
    assert db.automatic_work.snapshot(first.chain_id).conversation_id == "target-a"
    assert db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    assert not db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    assert not db.automatic_work.abort_chat_start(first.id, owner_id="owner")
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        prepare(db, source, "target-b", "attempt-b")
    assert db.automatic_work.snapshot(root).used["generation"] == 1
```

Run: `.venv/bin/python -m pytest Tests/DB/test_automatic_chat_starts.py::test_two_targets_share_the_last_generation -q`.

- [ ] **3. Add the migration and membership guards.** Add nullable `allowance_root_chain_id REFERENCES automatic_work_chains(id)` and an index. Extend the immutable identity trigger; require a nonnull reference to name an existing null-reference root on INSERT and UPDATE. Include native attempt identity/state, a unique active target-conversation index, and the generation reservation FK. Forbid identity rewrites with a trigger, including source/target/owner/request fields. Mirror the SQL in the runtime migration; preserve all existing roots and run scope triggers. Check FK integrity after both runtime and standalone upgrades.
- [ ] **4. Resolve root state in every ledger path.** Keep `_chain()` local. Add `_allowance_chain(conn, chain_id)` that reads the direct root once and validates its null reference. Use it in `_snapshot`, `_check_admission`, `_start_automatic`, `_admission_transaction`, `pause`, `settle`, and `recover`. Keep `snapshot.chain_id` and `conversation_id` local. Aggregate reservations with this parameterized query:

```sql
SELECT reservation.*
FROM automatic_work_reservations AS reservation
JOIN automatic_work_chains AS member ON member.id = reservation.chain_id
WHERE member.id = ? OR member.allowance_root_chain_id = ?;
```

Supply `(root_id, root_id)` and read limits/status/deadline/clock anchors from the
root. A refusal's savepoint rollback must retain observed root time and root pause,
with the existing `runtime_owner_replaced` exception preventing stale-owner pause.
Settlement of a descendant's unknown usage or excess updates the root. Recovery
collects canonical roots from both wake/start attempts and unresolved reservations,
fences old owners once, and retains review status after late settlement.

- [ ] **5. Implement native attempt transitions inside the ledger's existing FULL-synchronous transaction.** Derive source chain from `agent_runs.work_chain_id`; refuse missing chain or terminal source and validate the current ledger runtime owner. Live source-session/run ownership is checked by the controller before calling this trusted interface; the existing run row has no per-run runtime-owner column. Use `root_submission_id=f"chat-start:{attempt_id}"` for the local target chain and point it directly to the canonical root. Compare every immutable field on duplicate preparation. Prepare target membership and generation reservation in the same transaction, with no nested public ledger transactions. Check active wake and start attempts in both preparation paths so the two tables cannot own the same target concurrently. Accept rechecks root budget/owner and starts the root clock once. Abort refunds only prepared state; complete has no survivor delivery side effects.
- [ ] **6. Add root-state and authority failure tests.** Add the following test to the same new file; run it red before the root settlement implementation and green afterward:

```python
def test_descendant_unknown_usage_blocks_siblings(db):
    root = chain(db, conversation="source", generations=3)
    source = db.create_run(
        conversation_id="source", agent_kind="primary", work_chain_id=root
    )
    first = prepare(db, source, "a", "first")
    sibling = prepare(db, source, "b", "sibling")
    db.automatic_work.reserve(
        first.chain_id, reservation_id="usage", owner_id="owner",
        kind="tokens", amount=20,
    )
    db.automatic_work.commit("usage", owner_id="owner")
    db.automatic_work.settle("usage", owner_id="owner", actual_amount=None)
    for member in (root, first.chain_id, sibling.chain_id):
        assert db.automatic_work.snapshot(member).status == "review_required"
    with pytest.raises(AutomaticWorkRefused, match="usage_unknown"):
        db.automatic_work.accept_chat_start(sibling.id, owner_id="owner")
```

Also extend existing barrier/deadline/migration tests to cover: two targets racing
for one slot; recursive starts point directly to the root; overage pauses siblings;
backward clock and original deadline across members/reopen; direct SQL reparent,
cycle and cross-conversation run insert refusal; startup marks prepared/accepted
foreign-owner starts for review; missing source chain mints nothing; existing wake
attempts and manual roots remain compatible. Use injected clocks and Events,
not timing-only assertions. Extend accepted-context tests for both attempt kinds,
wrong owner/kind, missing acceptance and current kill switch.

- [ ] **7. Run focused verification and review the foundation diff.** Run all changed DB tests plus `test_automatic_work_lineage.py`; check the migration query plan uses the membership/reservation indexes without statistics. Run scoped lint/format checks below. Update the task's Implementation Notes only after the implementation passes; link ADR-211 and record both migration paths. Commit only foundation files with `feat: preserve automatic allowance across agent-created chats`.



## Shared scoped checks (extracted-plan supplement)

Use the existing absolute env interpreter. Run `-m ruff check --select E9,F63,F7,F82` on the explicit modified Python files. New files must pass `-m ruff format --check`. For pre-existing files with formatter debt, record `scripts/terminal_qualification/format_ratchet.py snapshot --base 9ba96ebb626dd010f14d093b0e8d40f37c715d32 --output <owned-baseline-path> --path <each-modified-existing-Python-file>` and `verify --baseline <owned-baseline-path> --head HEAD` after committing. Run `git diff --check`. Avoid unrelated formatter sweeps.


## Shared reading and ownership reference


Read both the spec and ADR-211, each Backlog task before its changes, and:

- `backlog/docs/lessons-testing-evidence.md`, `lessons-live-verification.md`, `lessons-console-wiring.md`, and `lessons-backlog-hygiene.md`.
- ADR-150 agent creation, ADR-134/135 automatic budgets/recovery, ADR-079 workspace assistant defaults, ADR-069 project instructions and ADR-082 scratch.
- `backlog/docs/design-language.md` before card or screen changes.

The code below defines new contracts or shows the core changes; it is plan content,
not a claim that those interfaces exist today. Existing code locations are search
anchors because the checkout contains concurrent edits.

## File ownership map

| Owner | Files and changes |
| --- | --- |
| Automatic ledger | `DB/AgentRuns_DB.py`, `DB/automatic_work.py`: immutable allowance membership, root-wide accounting, native start-attempt state transitions and recovery |
| Automatic authority | `Agents/automatic_work_budget.py`, `Agents/automatic_work_runtime.py`: typed attempt projection and explicit accepted-attempt kind |
| Tool contract | `Agents/tool_catalog.py`, `Chat/console_agent_bridge.py`: schema, strict validation, per-invocation trusted run provenance and matching grant keys |
| Fresh defaults | `Chat/console_runtime.py`, `Chat/console_chat_controller.py`, `Chat/console_chat_store.py`, `Chat/console_generation_settings_metadata.py`: reuse creation resolver and snapshot codec; save and restore the same destination selection |
| Draft persistence | `Chat/console_chat_store.py`, `Chat/chat_persistence_service.py`: revision-fenced version-2 handoff edits, clears and accepted consumption |
| Durable submission | `DB/ChaChaNotes_DB.py`, `Chat/console_dispatch_checkpoint.py`, `Chat/console_dispatch_repository.py`, `Chat/message_metadata.py`: native machine origin and exact-attempt receipt |
| Start scheduling | New `Chat/console_chat_start.py`; existing `Chat/console_fleet_wake.py`, `Chat/console_runtime.py`, `Chat/console_chat_controller.py`: one immediate attempt, shared slots/owner, runtime-owned execution |
| Manual intent and view | `Chat/console_prompt_queue_coordinator.py`, `UI/Screens/chat_screen.py`, `Widgets/Chat_Widgets/chat_create_confirm_card.py`: manual arbitration before busy guards, observable outcomes, full approval preview |
| Migrations | AgentRunsDB 18→19 and ChaChaNotes 73→74 at the inspected checkout; verify the actual versions at execution and rename both artifacts if another migration has landed |

Only one new production module is planned. Do not extract a generic scheduler,
chat-management service or provider adapter. No stylesheet change is expected;
if an existing token-backed status class is insufficient, change source CSS and
rebuild through `css/build_css.py` under the design constitution.
