### Task 7: Integrate holders, atomic first persistence, and promotion rollback

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:206-340,516-760,927-1510,5421-5845`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Create: `Tests/Chat/test_console_chat_store_library_policy.py`
- Create: `Tests/Chat/test_console_chat_store_atomic_promotion.py`

**Interfaces:**
- Consumes: holder/coordinator, transaction contributions, repository primitives.
- Produces: `ConsoleChatSession.library_policy_holder`, store-owned coordinator registration, `stage_first_persistence()`, `publish_committed_identity()`, contribution-aware `promote_ephemeral_session()`, and unresolved-operation promotion guard.

- [ ] **Step 1: Write RED session lifecycle tests.** New local session captures current defaults; untouched defaults do not make a pristine tab dirty; an explicit empty-tab edit does; first local persistence inserts policy even when not edited; restored missing-row conversation stays write-free Never/Blocked until explicit save.
- [ ] **Step 2: Write RED staged-identity tests.** Inject conversation/policy write failures and assert `persisted_conversation_id`, title, scope holder, message IDs, attachments, and policy holder remain byte-for-byte pre-call. A retry creates one conversation and publishes ID/title only after commit. Workspace membership is an idempotent post-commit projection from durable `workspace_id`, never part of the Chat transaction.
- [ ] **Step 3: Write RED promotion tests.** Promotion refuses unresolved preparation/checkpoint analogue before any write; success persists policy/full lineage/contributions atomically in ChaChaNotes; each injected Chat write failure restores ephemeral identity, policy, messages, scope, contributions, and retryability. Workspace projection failure preserves committed identity and exposes retryable reconciliation without duplicates.
- [ ] **Step 4: Run RED.** Run the two new files plus `Tests/Chat/test_console_chat_store.py`; expected: lifecycle/atomicity failures.
- [ ] **Step 5: Refactor eager mutation out of first persistence.** Introduce immutable staging rather than assigning session fields inside `create_conversation`:

```python
@dataclass(frozen=True, slots=True)
class ConsoleStagedConversationIdentity:
    conversation_id: str
    title: str

def publish_committed_identity(
    self, session_id: str, identity: ConsoleStagedConversationIdentity
) -> None:
    session = self._session_or_raise(session_id)
    session.persisted_conversation_id = identity.conversation_id
    session.title = identity.title
```

The publishing method is called only after the transaction context exits successfully.
- [ ] **Step 6: Integrate holder/coordinator ownership.** Register/unregister holders on restore/create/close/rollback/state replacement, hydrate restored policy off-loop before execution, centralize current defaults in Store/runtime, publish committed saves to same-process siblings, and share one coordinator per app/store.
- [ ] **Step 7: Run GREEN and targeted foundation battery.** Run `Tests/Chat/test_console_library_policy.py`, `Tests/Chat/test_assistant_generation_state.py`, `Tests/DB/test_chachanotes_console_library_migration_seed_openers.py`, `Tests/DB/test_chachanotes_console_library_policy_migration.py`, `Tests/ChaChaNotesDB/test_migration_atomicity.py`, `Tests/ChaChaNotesDB/test_console_library_policy_repository.py`, `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`, `Tests/Chat/test_console_library_policy_coordinator.py`, `Tests/Chat/test_console_transaction_contribution.py`, `Tests/Chat/test_console_chat_store_library_policy.py`, and `Tests/Chat/test_console_chat_store_atomic_promotion.py` together, then scoped Ruff and `git diff --check`.
- [ ] **Step 8: Finish TASK-19900.1 hygiene.** Check every acceptance criterion, add concise Implementation Notes naming schema/sync/repository/lifecycle changes and targeted evidence, update a lessons file only if an actual reusable incident occurred, then run `backlog task edit 19900.1 -s Done` only when all DoD items are true.
- [ ] **Step 9: Commit.** Commit `feat(console): persist Library policy atomically`.

---

## Delivery 2 — Runtime authority and provider composition (`TASK-19900.2`)
