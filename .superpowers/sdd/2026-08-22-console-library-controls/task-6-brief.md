### Task 6: Implement policy/checkpoint repositories, coordinator, and generic transaction contributions

**Files:**
- Create: `tldw_chatbook/Chat/console_library_policy_repository.py`
- Create: `tldw_chatbook/Chat/console_library_policy_coordinator.py`
- Create: `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- Create: `tldw_chatbook/Chat/console_dispatch_repository.py`
- Create: `tldw_chatbook/Chat/console_transaction_contribution.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Create: `Tests/ChaChaNotesDB/test_console_library_policy_repository.py`
- Create: `Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py`
- Create: `Tests/Chat/test_console_library_policy_coordinator.py`
- Create: `Tests/Chat/test_console_transaction_contribution.py`

**Interfaces:**
- Consumes: v45 schema, strict models, existing DB transaction/message version/hash/sync primitives, ADR-063 codec.
- Produces the exact methods below:

```python
class ConsoleTransactionWriter(Protocol):
    def next_trajectory_sequence(self) -> int:
        """Allocate one seq for the accepted conversation in this transaction."""

    def execute(self, statement: str, parameters: tuple[object, ...], /) -> None:
        """Execute one parameterized INSERT through the caller transaction."""

    def executemany(
        self,
        statement: str,
        parameter_rows: Iterable[tuple[object, ...]],
        /,
    ) -> None:
        """Execute parameterized INSERT rows through the caller transaction."""

class ConsoleTransactionContribution(Protocol):
    def write(
        self,
        *,
        writer: ConsoleTransactionWriter,
        conversation_id: str,
        message_ids: Mapping[str, str],
    ) -> None:
        """Write through the caller-owned capability without committing."""

class ConsoleLibraryPolicyRepository:
    def read(self, conversation_id: str) -> ConsoleLibraryPolicyReadResult:
        """Read one policy or an explicit fail-closed outcome."""

    def insert(self, conversation_id: str, candidate: ConsoleLibraryPolicyCandidate) -> ConsoleLibraryPolicyWriteResult:
        """Conditionally insert revision one without overwriting a race winner."""

    def compare_and_swap(self, conversation_id: str, expected_revision: int, candidate: ConsoleLibraryPolicyCandidate) -> ConsoleLibraryPolicyWriteResult:
        """Commit exactly one expected revision or report conflict."""

class ConsoleDispatchRepository:
    def insert_with_messages(self, cursor: sqlite3.Cursor, acceptance: ConsoleDurableTurnAcceptance) -> ConsoleDispatchCheckpoint:
        """Insert one USER, assistant owner, and accepted checkpoint."""

    def read_for_session(self, conversation_id: str) -> ConsoleDispatchReadResult:
        """Read and validate at most one active-path recovery owner."""

    def cas_state(self, transition: ConsoleDispatchTransition) -> ConsoleDispatchWriteResult:
        """Apply an expected-revision accepted/dispatch-started transition."""

    def settle_with_assistant(self, settlement: ConsoleAssistantSettlement) -> ConsoleDispatchWriteResult:
        """Commit terminal assistant state and delete its checkpoint atomically."""

    def handoff_to_provider_continuation(self, handoff: ConsoleContinuationHandoff) -> ConsoleDispatchWriteResult:
        """Commit ADR-063 ownership and remove dispatch ownership atomically."""

class ConsoleLibraryPolicyCoordinator:
    def register_holder(self, session_id: str, conversation_id: str | None, holder: ConsoleLibraryPolicyHolder) -> None:
        """Bind one live holder for same-process committed publication."""

    def unregister_holder(self, session_id: str) -> None:
        """Remove one closed session holder."""

    async def load(self, session_id: str, conversation_id: str) -> ConsoleLibraryPolicyReadResult:
        """Read durable policy off-loop and publish its effective result."""

    async def save(self, session_id: str, candidate: ConsoleLibraryPolicyCandidate) -> ConsoleLibraryPolicyWriteResult:
        """Commit one insert/CAS and publish only the committed snapshot."""

    async def capture_for_execution(self, session_id: str) -> ConsoleLibraryPolicySnapshot:
        """Perform the execution-time durable read and return frozen authority."""
```

- [ ] **Step 1: Write RED policy repository tests.** Cover valid/absent/corrupt/error reads, conditional insert race, update CAS success/conflict, missing/deleted conversation, no candidate publication, soft-delete retention/restore, and hard-purge cascade.
- [ ] **Step 2: Write RED checkpoint codec/ownership tests.** Pin exact JSON keys/types/order/byte caps; reject request text, source snippets, credentials, bad roles, cross-conversation owners, duplicate active-path owners, invalid states, and generic upsert behavior.
- [ ] **Step 3: Write RED atomic checkpoint tests.** Inject failure at USER, assistant, checkpoint, state-CAS, terminal-content, sync-intent, checkpoint-delete, continuation-write, and handoff-delete statements. Assert all-or-nothing plus expected checkpoint revision, USER/assistant versions, matching assistant state, and `deleted = 0`.
- [ ] **Step 4: Write RED coordinator tests.** Use two holders for one conversation and two repositories over one file DB. Assert off-loop execution, same-process publication only after commit, fresh execution read defeating stale Allowed, unavailable read producing Never/Blocked, and a commit after capture affecting only the next capture.
- [ ] **Step 5: Run RED.** Run the four new test files; expected: missing modules/contracts.
- [ ] **Step 6: Implement minimal repositories and coordinator.** Use parameterized SQL and typed result variants. `settle_with_assistant` and `handoff_to_provider_continuation` must write message content/state/version/hash/sync intent and delete the expected checkpoint in one `transaction(immediate=True)`.
- [ ] **Step 7: Implement the generic contribution seam.** Contributions receive only an insert-only `ConsoleTransactionWriter`, committed conversation ID candidate, and message-ID map. The writer also exposes `next_trajectory_sequence() -> int`, a no-argument allocator bound internally to the accepted conversation. One writer is shared across the entire contribution loop and revoked after that scope. Its first allocator call reads `MAX(seq)` through the private caller cursor inside the same `BEGIN IMMEDIATE`; later calls return consecutive values across all contributions. Missing rows start at `1`; non-integer, negative, or SQLite 64-bit-overflowing maxima fail closed, and rollback leaks no reservation. The allocator accepts no conversation/table/column/count/range input and exposes no generic read. Its complete accepted SQL grammar remains one `INSERT INTO simple_table (simple_column, ...) VALUES (?, ...)` statement with ordinary whitespace, one VALUES row, unquoted/unqualified ASCII identifiers, and equal non-zero column/placeholder/tuple arity; `executemany` requires at least one same-arity tuple. It rejects conflict modifiers/clauses (including REPLACE/IGNORE and both ON CONFLICT actions), literals/comments standing in for placeholders, INSERT...SELECT, RETURNING, multiple VALUES rows, quoted/dynamic identifiers, and every extra clause. The writer exposes no raw cursor/connection, authorizer, transaction/savepoint/ATTACH/DETACH control, commit/rollback, repository/session/publication state, or connection factory. Contribution errors propagate through the caller-owned `BEGIN IMMEDIATE` transaction. This is an API capability boundary for trusted in-process components, not a hostile-code sandbox.
- [ ] **Step 8: Run GREEN, lint, and mutation probes.** Re-run Step 5; temporarily invert missing/error fail-closed and remove a checkpoint version predicate one at a time, confirm named tests fail, then restore the implementation. Run scoped Ruff.
- [ ] **Step 9: Commit.** Commit `feat(console): add Library policy and dispatch repositories`.
