# PR #1096 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all six actionable PR #1096 review findings without changing its authority, provenance, TTS, or audio.cpp behavior.

**Architecture:** Keep each amendment at its existing boundary: Console rejects an unclassifiable persistence source, the character DB uses its transaction context, the configured-target store validates its path at construction, and the reviewed public methods document their existing contracts. Regressions exercise the public seams and prove no fallback can assign false local provenance.

**Tech Stack:** Python 3.11+, pytest, Loguru, SQLite, shared `path_validation.py`, Ruff

**Design:** [PR #1096 Review Remediation](../specs/2026-07-30-pr-1096-review-remediation-design.md)

**Related task:** [TASK-617.2](<../../../backlog/tasks/task-617.2 - Establish-character-authority-and-conversation-provenance.md>)

**ADR required:** no

**ADR path:** N/A; [ADR-037](../../../backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md) remains governing.

**Reason:** The amendments enforce and document existing authority/provenance policy. They add no schema, storage owner, runtime boundary, dependency, or alternative authority rule.

---

## File map

- Modify `tldw_chatbook/Chat/console_chat_store.py`: fail loudly and observably when an invalid runtime backend reaches persistence.
- Modify `Tests/Chat/test_console_chat_store.py`: prove invalid persistence identity logs, raises, and writes nothing.
- Modify `tldw_chatbook/DB/ChaChaNotes_DB.py`: use the shared transaction cursor and complete the local-authority docstring.
- Modify `Tests/DB/test_chachanotes_character_authority_migration.py`: prove transaction-seam use and the docstring contract.
- Modify `tldw_chatbook/MCP/server_target_store.py`: validate the chosen store path and document constructor failure.
- Modify `tldw_chatbook/MCP/unified_control_models.py`: document configured-target serialization methods.
- Modify `Tests/MCP/test_server_target_store.py`: prove dangerous paths fail and serialization docs are complete.
- Modify `tldw_chatbook/Chat/chat_conversation_service.py`: document conversation creation, including authority provenance.
- Modify `tldw_chatbook/Chat/chat_persistence_service.py`: document the lower persistence creation boundary.
- Modify `Tests/Chat/test_chat_conversation_service.py`: enforce the conversation-service public contract documentation.
- Modify `Tests/Chat/test_chat_persistence_service.py`: enforce the persistence-service public contract documentation.
- Verify `Tests/MCP/test_store_default_paths.py`: ensure the validated default and explicit paths remain byte-for-byte compatible.

No files are created by the implementation and no existing module is split.

### Task 1: Make invalid Console persistence identity observable

**Files:**

- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:1966`

- [ ] **Step 1: Write the failing regression**

Add a test beside the existing `persist_session_if_needed` observability tests:

```python
def test_persist_session_if_needed_rejects_invalid_runtime_backend_observably():
    from loguru import logger as loguru_logger

    persistence = FakePersistence()
    store = ConsoleChatStore(persistence=persistence)
    session = store.create_session(
        title="Unscoped restored chat",
        runtime_backend="invalid",
    )
    messages: list[str] = []
    sink_id = loguru_logger.add(
        messages.append,
        level="WARNING",
        format="{extra[session_id]}|{extra[runtime_backend]}|{message}",
    )
    try:
        with pytest.raises(
            ValueError,
            match="runtime_backend must be 'local' or 'server'",
        ):
            store.persist_session_if_needed(session.id)
        with pytest.raises(
            ValueError,
            match="runtime_backend must be 'local' or 'server'",
        ):
            store.append_message(
                session.id,
                role=ConsoleMessageRole.USER,
                content="Keep this message in memory",
                persist=True,
            )
    finally:
        loguru_logger.remove(sink_id)

    assert persistence.created_conversations == []
    assert persistence.created_messages == []
    assert any(
        session.id in message
        and "invalid" in message
        and "persist" in message.lower()
        for message in messages
    )
```

- [ ] **Step 2: Update the existing fail-closed provenance expectation**

In the parameterized
`test_invalid_runtime_source_never_reaches_real_chat_persistence`, replace the
old `None` expectations with `pytest.raises(ValueError, ...)` around both
`persist_session_if_needed()` and `append_message(..., persist=True)`. Keep its
real-DB assertions, and inspect the in-memory message after the raised append:

```python
with pytest.raises(ValueError, match="runtime_backend must be"):
    store.persist_session_if_needed(session.id)
with pytest.raises(ValueError, match="runtime_backend must be"):
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="Keep this message in memory",
        persist=True,
    )

message = store.messages_for_session(session.id)[-1]
assert session.persisted_conversation_id is None
assert message.persisted_message_id is None
assert create_calls == []
assert db.get_all_conversation_ids() == []
assert db.count_character_cards() == character_count
```

This preserves the existing proof that malformed provenance cannot be
materialized as local while changing only the formerly silent failure
contract.

- [ ] **Step 3: Run the tests and confirm the silent-return defect**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_chat_store.py::test_persist_session_if_needed_rejects_invalid_runtime_backend_observably \
  Tests/Chat/test_console_chat_store.py::test_invalid_runtime_source_never_reaches_real_chat_persistence
```

Expected: FAIL because no exception or structured diagnostic is produced.

- [ ] **Step 4: Implement the minimal fail-loud boundary**

Replace the invalid-backend `return None` in
`ConsoleChatStore.persist_session_if_needed()` with:

```python
runtime_backend = session.runtime_backend
if type(runtime_backend) is not str or runtime_backend not in {"local", "server"}:
    logged_backend = repr(runtime_backend)[:128]
    logger.bind(
        session_id=session_id,
        runtime_backend=logged_backend,
    ).error("Cannot persist Console session with invalid runtime backend.")
    raise ValueError(
        "Cannot persist Console session: "
        "runtime_backend must be 'local' or 'server'."
    )
```

Keep the existing early returns for an already-persisted session and an absent
persistence adapter. Do not coerce or infer a backend.

- [ ] **Step 5: Run the focused Console tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_chat_store.py::test_persist_session_if_needed_rejects_invalid_runtime_backend_observably \
  Tests/Chat/test_console_chat_store.py
```

Expected: the new regression and the complete file PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  Tests/Chat/test_console_chat_store.py \
  tldw_chatbook/Chat/console_chat_store.py
git commit -m "fix: reject invalid console persistence backend"
```

### Task 2: Use the character DB transaction seam

**Files:**

- Modify: `Tests/DB/test_chachanotes_character_authority_migration.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py:2737`

- [ ] **Step 1: Write failing transaction and documentation tests**

Import `contextmanager` and `inspect`, then add:

```python
def test_local_authority_accessor_uses_shared_transaction_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(tmp_path / "authority-transaction.sqlite", "test-client")
    real_transaction = db.transaction
    calls = 0

    @contextmanager
    def recording_transaction():
        nonlocal calls
        calls += 1
        with real_transaction() as cursor:
            yield cursor

    monkeypatch.setattr(db, "transaction", recording_transaction)
    try:
        assert db.get_local_authority_id()
        assert calls == 1
    finally:
        db.close_connection()


def test_local_authority_accessor_documents_public_contract() -> None:
    docstring = inspect.getdoc(CharactersRAGDB.get_local_authority_id)

    assert docstring is not None
    assert "Returns:" in docstring
    assert "Raises:" in docstring
```

- [ ] **Step 2: Run the tests and confirm both review gaps**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/DB/test_chachanotes_character_authority_migration.py::test_local_authority_accessor_uses_shared_transaction_seam \
  Tests/DB/test_chachanotes_character_authority_migration.py::test_local_authority_accessor_documents_public_contract
```

Expected: FAIL because the accessor bypasses `transaction()` and lacks
`Returns:`.

- [ ] **Step 3: Route the lookup through the transaction cursor**

Change the lookup to:

```python
try:
    with self.transaction() as cursor:
        rows = cursor.execute(
            """
            SELECT local_authority_id
            FROM rag_identity_context
            WHERE context_name = 'default'
            LIMIT 2
            """
        ).fetchall()
except sqlite3.Error as exc:
    raise CharactersRAGDBError(
        "Local authority identity is unavailable or invalid."
    ) from exc
```

Add this docstring section before `Raises:`:

```python
Returns:
    The database-owned local authority identifier.
```

- [ ] **Step 4: Run the complete migration/authority test file**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/DB/test_chachanotes_character_authority_migration.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  Tests/DB/test_chachanotes_character_authority_migration.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py
git commit -m "fix: use transaction for local authority lookup"
```

### Task 3: Validate configured-target store paths and document serialization

**Files:**

- Modify: `Tests/MCP/test_server_target_store.py`
- Modify: `tldw_chatbook/MCP/server_target_store.py:45`
- Modify: `tldw_chatbook/MCP/unified_control_models.py:237`
- Verify: `Tests/MCP/test_store_default_paths.py`

- [ ] **Step 1: Write failing path and documentation tests**

Import `inspect`, then add:

```python
def test_target_store_rejects_dangerous_path() -> None:
    with pytest.raises(ValueError, match="dangerous pattern"):
        ConfiguredServerTargetStore("../../mcp_server_targets.json")


def test_configured_server_target_serialization_documents_public_contract() -> None:
    to_dict_doc = inspect.getdoc(ConfiguredServerTarget.to_dict)
    from_dict_doc = inspect.getdoc(ConfiguredServerTarget.from_dict)

    assert to_dict_doc is not None
    assert "Returns:" in to_dict_doc
    assert from_dict_doc is not None
    assert "Args:" in from_dict_doc
    assert "Returns:" in from_dict_doc
```

- [ ] **Step 2: Run the tests and confirm validation/docs are absent**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_server_target_store.py::test_target_store_rejects_dangerous_path \
  Tests/MCP/test_server_target_store.py::test_configured_server_target_serialization_documents_public_contract
```

Expected: FAIL because the path is accepted and the methods have no
docstrings.

- [ ] **Step 3: Validate the constructor boundary**

Import the shared helper:

```python
from tldw_chatbook.Utils.path_validation import validate_path_simple
```

Implement:

```python
def __init__(self, path: str | Path | None = None) -> None:
    """Initialize a configured-server-target store.

    Args:
        path: Optional JSON store path. Defaults to the current user data
            directory.

    Raises:
        ValueError: If the selected path contains a known security risk.
    """
    selected_path = Path(path) if path else _default_server_targets_path()
    self.path = validate_path_simple(selected_path, require_exists=False)
```

- [ ] **Step 4: Add minimal Google-style serialization docs**

Add to `ConfiguredServerTarget.to_dict()`:

```python
"""Serialize this configured target.

Returns:
    A JSON-compatible mapping containing routing, authority, authentication
    reference, and status metadata.
"""
```

Add to `ConfiguredServerTarget.from_dict()`:

```python
"""Restore a configured target from serialized data.

Args:
    data: Serialized mapping. Non-mapping values produce an empty legacy
        target.

Returns:
    The normalized configured target.
"""
```

- [ ] **Step 5: Run target-store and default-path regressions**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/MCP/test_server_target_store.py \
  Tests/MCP/test_store_default_paths.py
```

Expected: PASS, including exact default and explicit path equality.

- [ ] **Step 6: Commit**

```bash
git add \
  Tests/MCP/test_server_target_store.py \
  tldw_chatbook/MCP/server_target_store.py \
  tldw_chatbook/MCP/unified_control_models.py
git commit -m "fix: validate configured server target path"
```

### Task 4: Document both conversation-creation boundaries

**Files:**

- Modify: `Tests/Chat/test_chat_conversation_service.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py:358`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:72`

- [ ] **Step 1: Add failing contract tests**

Add to the conversation-service tests:

```python
def test_create_conversation_documents_public_contract():
    docstring = inspect.getdoc(ChatConversationService.create_conversation)

    assert docstring is not None
    assert "Args:" in docstring
    assert "assistant_authority_id" in docstring
    assert "Returns:" in docstring
    assert "Raises:" in docstring
```

Add the equivalent assertion for
`ChatPersistenceService.create_conversation` to its test class.

- [ ] **Step 2: Run the tests and confirm both docstrings are absent**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_chat_conversation_service.py::test_create_conversation_documents_public_contract \
  Tests/Chat/test_chat_persistence_service.py -k "create_conversation_documents_public_contract"
```

Expected: FAIL because neither public method has a docstring.

- [ ] **Step 3: Document `ChatConversationService.create_conversation()`**

Add a Google-style docstring covering every named argument and
`extra_fields`. Its authority entry must state that omission lets the DB apply
eligible local inference while explicit `None` preserves unproven authority.
Document the returned string ID and the existing `ValueError` when the DB
cannot create the conversation.

- [ ] **Step 4: Document `ChatPersistenceService.create_conversation()`**

Add a Google-style docstring covering every named argument. Preserve the same
authority omission-versus-null distinction, describe workspace linkage, return
the persisted ID, and document invalid workspace scope/linkage failure without
changing runtime behavior.

- [ ] **Step 5: Run both complete service test files**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chat/test_chat_persistence_service.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chat/test_chat_persistence_service.py \
  tldw_chatbook/Chat/chat_conversation_service.py \
  tldw_chatbook/Chat/chat_persistence_service.py
git commit -m "docs: complete conversation creation contracts"
```

### Task 5: Verify, review, and open the follow-up PR

**Files:**

- Review all files changed since `origin/dev`
- Do not add unrelated cleanup

- [ ] **Step 1: Run the complete focused regression union**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/DB/test_chachanotes_character_authority_migration.py \
  Tests/MCP/test_server_target_store.py \
  Tests/MCP/test_store_default_paths.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/Chat/test_console_chat_store.py
```

Expected: PASS. The pre-change five-file baseline was 260 passed; the added
default-path file and new regressions must remain green.

- [ ] **Step 2: Run task-scoped lint and formatting checks**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/chat_conversation_service.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/MCP/server_target_store.py \
  tldw_chatbook/MCP/unified_control_models.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/DB/test_chachanotes_character_authority_migration.py \
  Tests/MCP/test_server_target_store.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/chat_conversation_service.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/DB/ChaChaNotes_DB.py \
  tldw_chatbook/MCP/server_target_store.py \
  tldw_chatbook/MCP/unified_control_models.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_chat_conversation_service.py \
  Tests/Chat/test_chat_persistence_service.py \
  Tests/DB/test_chachanotes_character_authority_migration.py \
  Tests/MCP/test_server_target_store.py
```

Expected: PASS, or document an exact unchanged `origin/dev` baseline before
proceeding.

- [ ] **Step 3: Run repository-wide tests**

Run:

```bash
../../.venv/bin/python -m pytest -q
```

Expected: PASS. If an inherited failure blocks the suite, reproduce the exact
node and failure on `origin/dev`; do not repair unrelated baseline failures in
this PR.

- [ ] **Step 4: Inspect the complete diff**

Run:

```bash
git diff --check origin/dev...HEAD
git status --short
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_chatbook Tests Docs/superpowers
```

Expected: no whitespace errors, no uncommitted files, and only the reviewed
remediation.

- [ ] **Step 5: Request independent code review**

Use `superpowers:requesting-code-review` against `origin/dev...HEAD`. Address
every verified Critical or Important finding and rerun the affected gates.

- [ ] **Step 6: Rebase on the latest `origin/dev`**

```bash
git fetch origin dev
git rebase origin/dev
```

Rerun the focused union and `git diff --check` after the rebase.

- [ ] **Step 7: Push and create a ready follow-up PR**

```bash
git push -u origin codex/pr-1096-review-fixes
gh pr create \
  --base dev \
  --head codex/pr-1096-review-fixes \
  --title "fix: address PR #1096 review findings" \
  --body-file /tmp/pr-1096-review-fixes.md
```

The PR description must link PR #1096, enumerate all six resolved findings,
state that ADR-037 remains governing, list verification evidence, and call out
the explicit non-goals.
