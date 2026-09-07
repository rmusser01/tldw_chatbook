# Console Thinking Persistence Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist a selected Console assistant generation's bounded thinking evidence and conversation replay preference without mixing displayable data with ADR-063 private continuation.

**Architecture:** Introduce one strict, content-safe thinking-envelope module; add nullable message/conversation fields with complete trigger coverage; make the store's live variant and regeneration snapshot own the whole generation; write the selected projection atomically through the existing ChaChaNotes transaction seam; then extend Sync v2 hashes/envelopes and an explicit persistence capability preflight. Unsupported durable versions remain opaque on unrelated writes and block generation mutation.

**Tech Stack:** Python 3.11+, SQLite/FTS5, dataclasses/`Literal`, pytest, existing Console store and Sync v2 repositories.

**Spec:** `Docs/superpowers/specs/2026-08-26-console-thinking-blocks-design.md`

**Task:** `backlog/tasks/task-18932.1 - Persist-selected-generation-thinking-and-replay-policy.md`

## Global Constraints

- Complete this plan before TASK-18932.2 starts; provider/UI code consumes these types rather than defining parallel state.
- Use an isolated worktree and temporary database paths. Recheck the current schema version immediately before Task 2.
- Proprietary blocks have no text field in their Python type or serialized object. Displayable text and opaque JSON are `repr=False`.
- Parser errors and warnings name only the violated field/rule; never include JSON, text, provider/model values, byte samples, or hashes.
- Bounds apply to UTF-8 bytes after canonical JSON decoding: 32 blocks/envelope, 256 KiB/block text, 1 MiB/envelope, 200 chars for provider/model/protocol/source fields, and 128 chars for IDs. Centralize constants so sync/import/provider code reuses them.
- Missing/NULL means no evidence and Auto policy. Do not backfill blocks or a literal `auto` value.
- Rebuild sync triggers for both new fields but leave FTS triggers scoped to visible message content. `thinking_blocks_json` must not enter FTS tables.
- Preserve the trigger-authored source intent as authoritative. Sync v2 reconciliation happens after commit and makes no cross-database atomicity claim.
- Narrow test fakes that never opt into thinking remain compatible. A persistent path that advertises/requests thinking must expose capability version 1.

---

### Task 1: Define the bounded thinking envelope and replay policy

**Files:**
- Create: `tldw_chatbook/Chat/thinking_blocks.py`
- Create: `Tests/Chat/test_thinking_blocks.py`

**Interfaces consumed:** standard-library `json`, `dataclasses`, `enum`, and existing identifier/string validation conventions in `provider_continuation.py`.

**Interfaces produced:** `DisplayableThinkingBlock`, `ProprietaryThinkingBlock`, their `ThinkingBlock` union, `ThinkingEnvelope`, `ThinkingEnvelopeRead`, `ThinkingHistoryPolicy`, canonical parse/dump/read functions, version and size constants.

- [ ] **Step 1: Write failing happy-path and canonical-round-trip tests.** Cover one displayable block, one proprietary block, ordered mixed blocks, exact displayable whitespace, NULL, and nullable/missing policy resolving to Auto.

```python
def test_thinking_envelope_round_trips_displayable_and_proprietary() -> None:
    envelope = ThinkingEnvelope(
        blocks=(
            DisplayableThinkingBlock(
                block_id="round-0",
                round_ordinal=0,
                provider="llama_cpp",
                model="qwen3",
                protocol="openai_chat",
                source_format="start_anchored_think",
                status="complete",
                text="  deliberate\nreasoning  ",
            ),
            ProprietaryThinkingBlock(
                block_id="round-1",
                round_ordinal=1,
                provider="moonshot",
                model="kimi-k3",
                protocol="chat_completions",
                source_format="reasoning_content",
                status="complete",
            ),
        )
    )
    raw = dump_thinking_blocks_json(envelope)
    assert parse_thinking_blocks_json(raw) == envelope
    assert "deliberate" not in repr(envelope)
```

- [ ] **Step 2: Run the new tests and confirm the module is missing.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_thinking_blocks.py -q`

Expected: FAIL during import because `thinking_blocks.py` does not exist.

- [ ] **Step 3: Implement immutable, visibility-specific block types.** Put shared provenance in a private base/helper or repeated frozen fields, then use a discriminated union. `ProprietaryThinkingBlock` has no `text` field or constructor parameter, making raw proprietary content structurally unrepresentable rather than merely rejected at runtime.

```python
ThinkingVisibility = Literal["displayable", "proprietary"]
ThinkingStatus = Literal["complete", "stopped", "failed"]
ThinkingHistoryPolicy = Literal["auto", "include", "exclude"]

@dataclass(frozen=True, slots=True)
class DisplayableThinkingBlock:
    block_id: str
    round_ordinal: int
    provider: str
    model: str
    protocol: str
    source_format: str
    status: ThinkingStatus
    text: str = field(repr=False)
    visibility: Literal["displayable"] = field(default="displayable", init=False)

@dataclass(frozen=True, slots=True)
class ProprietaryThinkingBlock:
    block_id: str
    round_ordinal: int
    provider: str
    model: str
    protocol: str
    source_format: str
    status: ThinkingStatus
    visibility: Literal["proprietary"] = field(default="proprietary", init=False)

ThinkingBlock = DisplayableThinkingBlock | ProprietaryThinkingBlock
```

Share strict field validation through a helper called from both `__post_init__` methods. The parser dispatches on `visibility`, rejects `text` as an unknown key for proprietary input, and requires non-empty bounded `text` for displayable input.

- [ ] **Step 4: Write failing mutation, version, bounds, duplicate-ID, ordinal, allowed-key, and redaction tests.** Include non-string JSON, bool-as-int ordinals, unknown keys, malformed UTF-8-size boundaries, text on proprietary, text absent on displayable, empty/duplicate IDs, non-monotonic ordinals, and unknown version behavior for durable read versus direct parse.

- [ ] **Step 5: Implement strict canonical parsing and content-free safe reads.** `parse_thinking_blocks_json` accepts only supported version 1. `read_thinking_blocks_json` distinguishes malformed supported data from an unsupported durable version and preserves the latter's exact raw string opaquely.

```python
@dataclass(frozen=True, slots=True)
class ThinkingEnvelopeRead:
    envelope: ThinkingEnvelope | None = field(default=None, repr=False)
    opaque_json: str | None = field(default=None, repr=False)
    warning: str | None = None

    @property
    def generation_actions_enabled(self) -> bool:
        return self.opaque_json is None

def normalize_thinking_history_policy(value: object) -> ThinkingHistoryPolicy:
    if value is None or value == "":
        return "auto"
    if type(value) is str and value in {"auto", "include", "exclude"}:
        return cast(ThinkingHistoryPolicy, value)
    return "auto"
```

Unknown imported/sync versions are handled by callers with strict parse and rejection; only durable hydration calls the preserving reader.

- [ ] **Step 6: Run the focused parser suite.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_thinking_blocks.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the pure contract.**

```bash
git add tldw_chatbook/Chat/thinking_blocks.py Tests/Chat/test_thinking_blocks.py
git commit -m "feat: define bounded Console thinking envelopes"
```

---

### Task 2: Add the ChaChaNotes fields and genuine migration coverage

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v51_to_v52_console_thinking.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Create: `Tests/DB/test_chachanotes_console_thinking_migration.py`
- Modify: `Tests/ChaChaNotesDB/legacy_conversation_schema.py` only if its historical fixture enumerates current columns instead of loading a genuine earlier schema.

**Interfaces consumed:** current schema migrator, messages/conversations sync-trigger templates, message FTS triggers.

**Interfaces produced:** nullable columns, current schema bump, DB create/read/update projection support, conversation policy getters/setters.

- [ ] **Step 1: Recheck schema and migration collisions.**

Run: `rg -n "_CURRENT_SCHEMA_VERSION|v51_to_v52|thinking_blocks_json|thinking_history_policy" tldw_chatbook/DB backlog/decisions`

Expected on the rebased implementation baseline: version 51 and no v51-to-v52 implementation. If another task advances the version, use the next integer and rename this task's migration/test expectations without changing semantics.

- [ ] **Step 2: Write a failing historical migration test.** Start from the real schema immediately before the new migration; insert assistant/user rows and a conversation; migrate; assert both columns are NULL, schema reaches `CharactersRAGDB._CURRENT_SCHEMA_VERSION`, message FTS still excludes a canary placed only in thinking, and message/conversation sync triggers include the new fields.

```python
def test_console_thinking_migration_is_additive_without_evidence_backfill(tmp_path):
    db_path = build_historical_database(tmp_path, through_version=51)
    db = CharactersRAGDB(db_path=str(db_path))
    assert db.get_message_by_id("assistant-1")["thinking_blocks_json"] is None
    conversation = db.get_conversation_by_id("conversation-1")
    assert conversation["thinking_history_policy"] is None
    assert db.get_schema_version() == db._CURRENT_SCHEMA_VERSION
```

- [ ] **Step 3: Run the migration test and confirm missing fields fail.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/DB/test_chachanotes_console_thinking_migration.py -q`

Expected: FAIL on missing columns/current migration.

- [ ] **Step 4: Add the migration and base-schema parity.** Add the two nullable TEXT columns. Recreate `messages_sync_insert/update/delete` and `conversations_sync_insert/update/delete` using the repository's current full trigger bodies plus the new fields. Do not modify `messages_ai`, `messages_au`, `messages_ad`, or any FTS content expression.

- [ ] **Step 5: Add DB boundary methods with canonical validation.** Message create/update/select queries must read the new message field. Conversation create/update/details queries must read the policy. Add one transactional selected-projection method instead of sequencing content and thinking writes in Python.

```python
def replace_assistant_generation_projection(
    self,
    *,
    message_id: str,
    content: str,
    thinking_blocks_json: str | None,
    provider_continuation_json: str | None,
    assistant_generation_state: str | None,
    usage_json: str | None,
    expected_version: int | None = None,
) -> int:
    """Replace one assistant row's selected generation and return its version."""
```

Validate/canonicalize thinking and continuation before opening the transaction. Inside one transaction, guard `role='assistant'`, `deleted=0`, and expected version when present; update the generation fields once so trigger-authored sync state observes one projection.

- [ ] **Step 6: Add trigger and API tests.** Assert message and conversation creates/updates produce sync-log records containing their fields, unrelated local-only metadata still does not, FTS cannot find a thinking-only canary, optimistic conflicts do not partially update, and deletion/tombstone removes access to thinking with the row.

- [ ] **Step 7: Run migration and DB tests.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/DB/test_chachanotes_console_thinking_migration.py Tests/DB/test_chachanotes_provider_continuation_migration.py Tests/DB/test_chachanotes_sync_conflict_preservation_migration.py -q`

Expected: PASS.

- [ ] **Step 8: Commit schema and DB support.**

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/DB/migrations Tests/DB/test_chachanotes_console_thinking_migration.py Tests/ChaChaNotesDB/legacy_conversation_schema.py
git commit -m "feat: persist Console thinking and replay policy"
```

---

### Task 3: Make variants and store persistence generation-complete

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_dispatch_checkpoint.py`
- Modify: `tldw_chatbook/Chat/console_dispatch_repository.py`
- Modify: `Tests/Chat/test_console_variant_stream.py`
- Create: `Tests/Chat/test_console_thinking_persistence.py`
- Modify: `Tests/Chat/test_console_dispatch_continuation_handoff.py`

**Interfaces consumed:** Task 1 envelope reader/dumper and Task 2 DB projection API.

**Interfaces produced:** hydrated message thinking state, full live variants, selected projection, stop/failure settlement, opaque-version mutation gate.

- [ ] **Step 1: Write failing model/variant tests.** Prove message snapshots hide thinking text from repr, a new regenerate clears only its working generation, successful finalization retains old/new paired envelopes, selection swaps answer/thinking/usage/continuation together, and abandoned regeneration restores all prior fields.

```python
def test_select_variant_swaps_the_complete_generation(store, assistant_id):
    original = store.message(assistant_id)
    store.begin_variant_stream(assistant_id)
    store.append_thinking_delta(assistant_id, _displayable_delta("new reasoning"))
    store.append_stream_chunk(assistant_id, "new answer")
    store.finalize_variant_stream(assistant_id)
    restored = store.select_variant(assistant_id, 0)
    assert restored.content == original.content
    assert restored.thinking == original.thinking
    assert restored.provider_continuation == original.provider_continuation
    assert restored.usage == original.usage
```

- [ ] **Step 2: Run variant/persistence tests and confirm pairing failures.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_thinking_persistence.py -q`

Expected: FAIL because variants currently own content only.

- [ ] **Step 3: Extend message, variant, and regeneration snapshot state.** Add parsed/opaque/warning state to `ConsoleChatMessage`; put the actual generation fields on `ConsoleVariant`; snapshot the same fields in `_VariantStreamBase`.

```python
@dataclass(frozen=True)
class ConsoleVariant:
    content: str
    thinking: ThinkingEnvelope | None = field(default=None, repr=False)
    opaque_thinking_json: str | None = field(default=None, repr=False)
    usage: ProviderUsage | None = None
    metadata: MessageMetadata | None = None
    provider_continuation: ProviderContinuationCheckpoint | None = field(
        default=None, repr=False
    )
    assistant_generation_state: str | None = None
    id: str = field(default_factory=lambda: str(uuid4()))
```

Keep `from_contents` as a compatibility constructor that creates evidence-free variants for legacy tests/callers. Add `from_generations` for complete values.

- [ ] **Step 4: Hydrate supported and opaque durable envelopes.** On restored rows, use `read_thinking_blocks_json`. Supported data becomes render/replay state; unknown version stores exact `opaque_thinking_json`, a content-free warning, and disables regenerate/edit/generation replacement. Malformed known versions surface a safe warning and no partial envelope.

- [ ] **Step 5: Add one explicit terminal projection path.** Extend `ChatPersistenceService` and `ConsoleChatPersistence` with a declared optional `replace_assistant_generation_projection` seam. Use Task 2's DB method. Keep `update_message_content` preserving thinking and continuation by default; confirmed edit calls the explicit clear path.

```python
def persist_selected_generation(self, message_id: str) -> bool:
    message = self._message_or_raise(message_id)
    if message.opaque_thinking_json is not None:
        raise ConsoleThinkingCompatibilityError(
            "This conversation contains a newer thinking format; upgrade before regenerating it."
        )
    return self.persistence.replace_assistant_generation_projection(
        message_id=message.persisted_message_id,
        content=message.content,
        thinking_blocks_json=dump_thinking_blocks_json(message.thinking),
        provider_continuation_json=dump_provider_continuation_json(
            message.provider_continuation
        ),
        assistant_generation_state=message.assistant_generation_state,
        usage_json=message.usage.to_json() if message.usage else None,
        expected_version=message.provider_continuation_message_version,
    )
```

Actual implementation returns/refreshes the committed version and projects the exact committed row to Sync v2. Narrow fakes receive no new kwarg unless they declare the seam.

- [ ] **Step 6: Thread the field through dispatch ownership.** Add thinking JSON to assistant-owner selects, durable-turn acceptance/settlement, recovery validation, and payload hashes. Continuation-only checkpoints preserve the current thinking field; terminal settlement updates both within the existing dispatch transaction. Stopped/failed capture persists a stopped/failed envelope, not a complete one.

- [ ] **Step 7: Add preservation and explicit-clear tests.** Cover ordinary feedback/metadata/title writes, continuation discard, selected variant, terminal completion, stop, handled failure, dispatch crash recovery, assistant edit confirmation, deletion, and descendant/branch behavior. Use canaries to prove an unknown raw envelope survives unrelated writes byte-for-byte.

- [ ] **Step 8: Run store, variant, persistence, and dispatch tests.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_thinking_persistence.py Tests/Chat/test_console_dispatch_continuation_handoff.py Tests/Chat/test_console_continuation_review_fixes.py -q`

Expected: PASS.

- [ ] **Step 9: Commit generation ownership.**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_dispatch_checkpoint.py tldw_chatbook/Chat/console_dispatch_repository.py Tests/Chat/test_console_variant_stream.py Tests/Chat/test_console_thinking_persistence.py Tests/Chat/test_console_dispatch_continuation_handoff.py
git commit -m "feat: keep Console thinking paired with selected generations"
```

---

### Task 4: Extend Sync v2 and persistent capability preflight

**Files:**
- Modify: `tldw_chatbook/Sync_Interop/hashing.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_builder.py`
- Modify: `tldw_chatbook/Sync_Interop/envelope_applier.py`
- Modify: `tldw_chatbook/Sync_Interop/chat_outbox_producer.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Create: `Tests/Sync_Interop/test_console_thinking_sync.py`
- Modify: `Tests/Chat/test_console_thinking_persistence.py`

**Interfaces consumed:** canonical envelope parser, committed-intent reader, existing whole-record sync conflict policy.

**Interfaces produced:** exact field hashing/envelopes/application and version-1 persistence capability.

- [ ] **Step 1: Write failing Sync v2 round-trip and rejection tests.** Build a real source DB row, reconcile its committed intent, decode the outbox envelope, apply it to a target DB, and compare content/thinking/continuation. Cover proprietary content-free block, deletion, conflict, malformed known version, unsupported version, and policy sync through existing conversation sync-log behavior.

- [ ] **Step 2: Run the sync test and confirm thinking is absent.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Sync_Interop/test_console_thinking_sync.py -q`

Expected: FAIL because current hashes/envelopes omit the field.

- [ ] **Step 3: Add thinking to committed intent, hash, envelope, and applier as one field.** Parse/canonicalize before enqueue and before apply. Reject unsupported incoming versions before a target transaction. Include `thinking_blocks_json` whenever non-NULL, even for a content-free proprietary block. Whole-record conflicts choose one complete content/thinking/continuation projection; never merge blocks.

```python
payload = {
    "assistant_generation_state": row["assistant_generation_state"],
    "content": row["content"],
    "role": row["role"],
}
if row["thinking_blocks_json"] is not None:
    payload["thinking_blocks_json"] = canonical_thinking_json(
        row["thinking_blocks_json"]
    )
if row["provider_continuation_json"] is not None:
    payload["provider_continuation_json"] = canonical_continuation_json(
        row["provider_continuation_json"]
    )
```

- [ ] **Step 4: Inventory every real `ConsoleChatPersistence` implementation, then add explicit capability reporting.** The current production Console path is `ChatPersistenceService`; make its `thinking_round_trip_version()` return 1. If execution-time inspection finds another production adapter, it may return 1 only after tests prove exact field round-trip. An absent, legacy, or future server adapter remains unsupported and must fail preflight rather than inherit local capability. Test fakes need not implement the method unless their configured provider resolution can emit thinking. Add a store/controller preflight callable that receives `may_emit_thinking` from the provider-resolution contract in TASK-18932.2; for now test it directly.

```python
def require_thinking_persistence_support(
    persistence: ConsoleChatPersistence | None,
    *,
    persistent: bool,
    may_emit_thinking: bool,
) -> None:
    if not persistent or not may_emit_thinking:
        return
    version_reader = getattr(persistence, "thinking_round_trip_version", None)
    if not callable(version_reader) or version_reader() != 1:
        raise ConsoleThinkingCompatibilityError(
            "This persistent backend cannot preserve model thinking version 1. Upgrade it before sending."
        )
```

Do not infer `may_emit_thinking` from a capability table here; TASK-18932.2 supplies an adapter-owned fact.

- [ ] **Step 5: Prove the refusal precedes provider contact.** Inject a provider spy, run the preflight with persistent+thinking enabled and missing/0/2 capability, assert the spy has zero calls. Include an unsupported server-mode persistence fake so a future remote path cannot silently use the local guarantee. Assert local v1, ephemeral, and ignored-disposition paths continue. If no production server persistence adapter exists at execution time, record that fact in TASK-18932.1 Implementation Notes and document that server mode cannot opt into thinking until its adapter implements the contract.

- [ ] **Step 6: Run sync and reachable persistence regressions.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Sync_Interop/test_console_thinking_sync.py Tests/Chat/test_console_thinking_persistence.py Tests/Chat/test_console_provider_continuation.py Tests/Chat/test_console_dispatch_continuation_handoff.py -q`

Expected: PASS.

- [ ] **Step 7: Run static checks and inspect durable owners.**

```bash
.venv/bin/python -m ruff format --check tldw_chatbook/Chat/thinking_blocks.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_dispatch_checkpoint.py tldw_chatbook/Chat/console_dispatch_repository.py tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Sync_Interop Tests/Chat/test_thinking_blocks.py Tests/Chat/test_console_thinking_persistence.py Tests/Sync_Interop/test_console_thinking_sync.py
.venv/bin/python -m ruff check tldw_chatbook/Chat/thinking_blocks.py tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_dispatch_checkpoint.py tldw_chatbook/Chat/console_dispatch_repository.py tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Sync_Interop Tests/Chat/test_thinking_blocks.py Tests/Chat/test_console_thinking_persistence.py Tests/Sync_Interop/test_console_thinking_sync.py
git diff --check
```

- [ ] **Step 8: Commit Sync/capability support and close TASK-18932.1.**

```bash
git add tldw_chatbook/Sync_Interop tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py Tests/Sync_Interop/test_console_thinking_sync.py Tests/Chat/test_console_thinking_persistence.py
git commit -m "feat: sync Console thinking with backend capability gates"
```

Update TASK-18932.1 ACs, add Implementation Notes with schema number and exact test evidence, and set it `Done` only after all checks pass.
