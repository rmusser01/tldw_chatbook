# Console Native Reasoning Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native-only reasoning prefill to Console with independent next-send and conversation-pinned values, exact turn ownership, and qualified direct/tool execution.

**Architecture:** A pure value/codec module and a small per-session reservation owner define the shared contract. Existing Console custody, queue, persistence, and provider paths carry that contract; adapters own native serialization and generated-versus-authored attribution. Context exposes one editor and preview after the runtime and storage boundaries exist.

**Tech Stack:** Python >=3.12, Textual 8.x, SQLite, existing Pydantic/httpx and Sync v2 infrastructure; no new dependency.

**Spec:** [Approved design](../specs/2026-09-12-console-native-reasoning-prefill-design.md).

**ADR required:** yes
**ADR path:** [ADR-159](../../../backlog/decisions/159-console-native-reasoning-prefill.md)
**Reason:** Provider continuation contracts, per-turn custody, and conversation-owned persistence. Implement accepted ADR-159; do not create a duplicate ADR.

## Global Constraints

- Native means the backend starts generation at the unfinished reasoning boundary.
- There is no conversion into system/user instructions, silent omission, or automatic switch of provider, API mode, endpoint, tools, or thinking settings to make a request succeed.
- Text is literal in this version. Existing response-prefill command syntax remains compatible.
- The configured text must be nonblank and at most 4,000 Unicode code points.
- Disabled text survives; accepted whitespace is preserved; no silent truncation.
- An unknown target is Unverified and blocked while a seed is active.
- The next-send slot and accepted submission's seed snapshot are process-memory state, excluded from disk screen snapshots, sync/export, forks, and global configuration.
- Required provider continuation and existing opt-in request capture retain their established input-retention rules.
- No automatic paid compatibility probes run at startup or send time.
- Do not run the full repository suite without the user's opt-in.
- All token-covered visual values use existing `$ds-*` tokens. Edit stylesheet sources and rebuild the generated bundle if CSS changes are needed.
- Every task reads its Backlog record and ADR-159 before starting, moves to In Progress, and only then adds its task-local Implementation Plan with the ADR check.
- Finish each task with targeted evidence, self-review, Implementation Notes, and a commit containing only that task's files. Never mark live qualification passed without actual native evidence.
- Execution requires an isolated checkout/worktree because the current checkout contains concurrent unrelated changes. Recheck actual symbols on that checkout; the paths below were inspected during planning.

## Delivery map and file ownership

| Task | Backlog | Prerequisites |
| --- | --- | --- |
| 1. Define native reasoning prefill values and reservation state | [TASK-32523](../../../backlog/tasks/task-32523%20-%20Define-native-reasoning-prefill-values-and-reservation-state.md) | TASK-32521 |
| 2. Persist and exchange conversation reasoning prefill pins | [TASK-32524](../../../backlog/tasks/task-32524%20-%20Persist-and-exchange-conversation-reasoning-prefill-pins.md) | TASK-32521, TASK-32523 |
| 3. Bind reasoning prefill to Console turn and queue ownership | [TASK-32525](../../../backlog/tasks/task-32525%20-%20Bind-reasoning-prefill-to-Console-turn-and-queue-ownership.md) | TASK-32521, TASK-32523, TASK-32524 |
| 4. Add explicit DeepSeek native reasoning prefix transport | [TASK-32526](../../../backlog/tasks/task-32526%20-%20Add-explicit-DeepSeek-native-reasoning-prefix-transport.md) | TASK-32521, TASK-32523, TASK-32525 |
| 5. Add template-qualified local reasoning prefill transport | [TASK-32527](../../../backlog/tasks/task-32527%20-%20Add-template-qualified-local-reasoning-prefill-transport.md) | TASK-32521, TASK-32523, TASK-32526 |
| 6. Preserve reasoning prefill attribution through tool replay and budgets | [TASK-32528](../../../backlog/tasks/task-32528%20-%20Preserve-reasoning-prefill-attribution-through-tool-replay-and-budgets.md) | TASK-32521, TASK-32525, TASK-32526, TASK-32527 |
| 7. Expose reasoning prefill editor and preview in Console | [TASK-32529](../../../backlog/tasks/task-32529%20-%20Expose-reasoning-prefill-editor-and-preview-in-Console.md) | TASK-32521, TASK-32524, TASK-32525, TASK-32528 |
| 8. Qualify native reasoning prefill and Console recovery end to end | [TASK-32530](../../../backlog/tasks/task-32530%20-%20Qualify-native-reasoning-prefill-and-Console-recovery-end-to-end.md) | TASK-32521, TASK-32526, TASK-32527, TASK-32528, TASK-32529 |

| Unit | Create | Integrate with existing owners |
| --- | --- | --- |
| Values, strict codec, support result | `Chat/reasoning_prefill.py` | Existing text-validation and strict metadata patterns |
| Exclusive live state | `Chat/reasoning_prefill_state.py` | Session store, runtime custody, queue and terminal callbacks |
| Durable pin | `Chat/reasoning_prefill_persistence.py` | `chat_persistence_service.py`, conversation hydration, DB and exchange boundaries |
| Protected pin Sync record | `Sync_Interop/reasoning_prefill_sync.py` | Existing encrypted envelopes, source intents, outbox and conflict handling |
| Native wire projection | `LLM_Calls/reasoning_prefill.py` | Gateway, `LLM_API_Calls.py`, hosted wire only where used |
| Output attribution | `Chat/reasoning_prefill_output.py` | Thinking capture/history, protected continuation, agent bridge and budgeting |
| Editor | `Widgets/Console/console_reasoning_prefill_modal.py` | Context inspector, command grammar, screen hooks and runtime |
| Release evidence | `Helper_Scripts/QA/reasoning_prefill.py` | Real Console runtime with explicitly configured targets |

Application paths in this table are under `tldw_chatbook/`. Create focused modules,
not a general plugin framework. Keep all existing provider branches operational
when no reasoning seed is supplied. Do not broadly refactor the large controller,
gateway, or persistence service to deliver this feature.

Read before execution: `backlog/docs/lessons-console-wiring.md`,
`backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`,
and `backlog/docs/design-language.md` for the UI task. Use the selected worktree's
`.venv/bin/python`; commands below are executed from its repository root.

## Task 1: Values, codec, support result, and exclusive reservation state

**Backlog:** [TASK-32523](../../../backlog/tasks/task-32523%20-%20Define-native-reasoning-prefill-values-and-reservation-state.md). **Dependencies:** TASK-32521.

**Files:**
- Create `tldw_chatbook/Chat/reasoning_prefill.py` and `reasoning_prefill_state.py`.
- Create `Tests/Chat/test_reasoning_prefill.py` and `test_reasoning_prefill_state.py`.
- Read `Chat/console_prefill.py`, `Chat/console_generation_settings_metadata.py`, and `Utils/input_validation.py`.

**Interfaces produced:**

`PrefillValue(text: str, enabled: bool = True)` is a frozen, strict Pydantic model
with text excluded from repr. `PrefillRead(status: Literal['absent', 'valid',
'invalid', 'unknown'], value: PrefillValue | None)` contains no malformed raw body.
`read_prefill_metadata(raw: object) -> PrefillRead`,
`merge_prefill_metadata(raw: object, value: PrefillValue | None) -> str`, and
`validate_prefill_exchange(raw: object) -> None` preserve siblings and refuse
lossy/unknown writes. Absence clears the key; unknown versions cannot be rewritten
without the explicit reset action in Task 7.
`reset_prefill_metadata(raw: object) -> str` removes only the owned key after an
explicit reset. It still requires a readable outer JSON object; it never replaces
unreadable whole-conversation metadata with an empty object.

`PrefillSnapshot(submission_id: str, value: PrefillValue, source:
Literal['next_send', 'pinned'], revision: int)` is frozen and body-redacted.
`ReasoningPrefillState` exposes `set_next(value) -> int`, `set_pinned(value) -> int`,
`reserve(submission_id: str, *, use_next: bool = True) -> PrefillSnapshot | None`,
`snapshot(submission_id: str) -> PrefillSnapshot | None`, and
`settle(submission_id: str, *, outcome: Literal['complete', 'stopped', 'failed',
'unknown', 'released'], dispatched: bool) -> None`.

`PrefillTarget` freezes `provider`, `api_mode`, `endpoint_key`, `model`,
`server_version`, `template_sha256`, `thinking_enabled: bool | None`,
`tools_enabled: bool`, and `response_prefill: bool`. Optional version/template
strings default to None. `endpoint_key` is the existing resolved destination's
opaque identity, not a URL displayed or logged by this module.

`PrefillCapability` freezes the target plus `status`, `reason_code`,
`wire_format: str | None`, `echo: Literal['suffix', 'full'] | None`,
`supports_tools: bool`, `supports_response_prefill: bool`, and
`evidence_ref: str | None`. `require_native_prefill(target, capability) -> None`
raises a content-free `PrefillUnsupportedError` for unavailable or mismatched
targets and combinations. A Supported result requires a native wire format,
known echo contract, and evidence reference. No production Supported entry is
introduced in this task.

- [ ] **1. Write the red codec and ownership tests.** Start with these exact cases,
  then parameterize invalid types, 4,000/4,001 code points, blank input, controls,
  disabled pins, invalid JSON, unknown versions, and sibling metadata preservation.

```python
from tldw_chatbook.Chat.reasoning_prefill import PrefillValue
from tldw_chatbook.Chat.reasoning_prefill_state import ReasoningPrefillState

def test_preserves_whitespace_and_old_completion_cannot_clear_new_seed():
    state = ReasoningPrefillState()
    state.set_next(PrefillValue(text="  First,\n"))
    old = state.reserve("turn-a")
    assert old.value.text == "  First,\n"
    state.set_next(PrefillValue(text="  First,\n"))
    state.settle("turn-a", outcome="complete", dispatched=True)
    newer = state.reserve("turn-b")
    assert newer.value.text == old.value.text
    assert newer.revision != old.revision

def test_second_queue_entry_uses_pin_and_failure_keeps_first_seed():
    state = ReasoningPrefillState()
    state.set_pinned(PrefillValue(text="Pinned"))
    state.set_next(PrefillValue(text="Once"))
    first = state.reserve("queue-a")
    second = state.reserve("queue-b")
    assert (first.value.text, second.value.text) == ("Once", "Pinned")
    state.settle("queue-a", outcome="failed", dispatched=True)
    assert state.snapshot("queue-a") == first
```

- [ ] **2. Run red:** `.venv/bin/python -m pytest Tests/Chat/test_reasoning_prefill.py Tests/Chat/test_reasoning_prefill_state.py -q`. Expect missing new modules first; after import scaffolding, retain failing behavioral assertions until implemented.
- [ ] **3. Implement the strict value, codec, and state transitions.** Use strict
  JSON-object parsing and the existing validator in rejection-only mode; compare
  accepted text with its original value rather than accepting sanitization.

```python
def checked_seed_text(text: str) -> str:
    from tldw_chatbook.Utils.input_validation import validate_text_input
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        raise ValueError("Reasoning prefill must contain 1–4000 characters.")
    if not validate_text_input(text, max_length=4000, allow_html=True):
        raise ValueError("Reasoning prefill is invalid.")
    if any((ord(c) < 32 and c not in "\t\n\r") or
           127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF
           for c in text):
        raise ValueError("Reasoning prefill contains unsupported characters.")
    return text
```

  The state owner increments a revision on every explicit write, including
  identical text. Reserve is idempotent per submission ID and leases next-send
  only if enabled and not already leased. Fall through to an enabled pin only
  when no available next-send value exists, not when its capability fails.
  Failed/unknown settlement leaves the immutable snapshot; complete/dispatched
  stop clears next-send only when its revision still matches. Release or
  undispatched stop restores availability without overwriting a newer slot.
  Reject a second terminal mutation rather than resurrecting a retired lease.
  Use the existing queue limit plus one primary owner as the reservation cap;
  refuse new admission when retained failed owners fill it, with explicit release
  available in Task 7. Never evict an unresolved owner to meet the cap.
- [ ] **4. Run green and scoped lint:** rerun the two files; then
  `.venv/bin/python -m ruff check tldw_chatbook/Chat/reasoning_prefill.py tldw_chatbook/Chat/reasoning_prefill_state.py Tests/Chat/test_reasoning_prefill.py Tests/Chat/test_reasoning_prefill_state.py`
  and the same paths with `ruff format --check`. Fix only these files.
- [ ] **5. Record evidence and commit** the two modules, their tests, and the task record with message `feat: define native reasoning prefill ownership`.

## Task 2: Conversation persistence, portability, and ephemeral exclusions

**Backlog:** [TASK-32524](../../../backlog/tasks/task-32524%20-%20Persist-and-exchange-conversation-reasoning-prefill-pins.md). **Dependencies:** TASK-32521, TASK-32523.

**Files:**
- Create `tldw_chatbook/Chat/reasoning_prefill_persistence.py`.
- Create `tldw_chatbook/Sync_Interop/reasoning_prefill_sync.py` for the scoped encrypted pin record.
- Modify `Chat/chat_persistence_service.py`, `Chat/console_chat_store.py`, `Chat/console_conversation_hydration.py`, and `Chat/console_chat_fork.py`.
- Modify `DB/ChaChaNotes_DB.py`, `Chat/Chat_Functions.py`, `Character_Chat/Character_Chat_Lib.py`, `Chatbooks/chatbook_creator.py`, and `Chatbooks/chatbook_importer.py` at conversation metadata/exchange boundaries.
- Modify `Sync_Interop/envelope_builder.py`, `Sync_Interop/envelope_applier.py`, `Sync_Interop/local_first_sync_service.py`, and `tldw_api/sync_schemas.py` to dispatch and advertise the scoped record before message-only validation.
- Create `Tests/Chat/test_reasoning_prefill_persistence.py`, `Tests/Chatbooks/test_reasoning_prefill_round_trip.py`, and `Tests/Sync_Interop/test_reasoning_prefill_sync.py`.
- Extend `Tests/Chat/test_thinking_privacy_surfaces.py` for authored-seed canaries.

**Consumes:** Task 1's value and codec. **Produces:**
`ReasoningPrefillRepository(db: CharactersRAGDB)` with
`read(conversation_id: str) -> tuple[PrefillRead, int]` and
`write(conversation_id: str, value: PrefillValue | None, *, expected_version: int) -> int`.
The returned integer is the new conversation version; missing conversations,
conflicts, and unreadable metadata raise content-free typed errors.
`reset(conversation_id: str, *, expected_version: int) -> int` performs the
explicit owned-key reset with the same version guard.
`ChatPersistenceService.supports_reasoning_prefill_version = 1` advertises this
contract only on backends that implement it. A missing capability means unsupported.
`ConsoleChatStore.reasoning_prefill_state(session_id) -> ReasoningPrefillState`
owns live state separately from the safe generation-defaults model.

`ReasoningPrefillSyncAdapter` in the new Sync module produces/applies encrypted
`domain='chat'` records distinguished by
`routing_metadata.record_type='conversation_reasoning_prefill'`, with stable key
`<conversation_id>:reasoning_prefill`. The version-1 encrypted payload contains
only `version`, `conversation_id`, and the complete `reasoning_prefill` value
(null means clear). Source intent is the existing whole-conversation `sync_log`
record containing metadata, not a reread of today's configuration. Wire/source
IDs and hashes are content-free; no new body-bearing database column is needed.
Advertise the explicit feature `chat.reasoning_prefill.v1`; a peer without it
retains pending local sync intent and receives no lossy substituted record.
`SyncEnvelopeBuilder.build_reasoning_prefill(conversation_id: str, value:
PrefillValue | None, *, base_version: str | None, entity_version: int) -> SyncV2Envelope`
uses the existing encrypted-envelope helper and dataset key. The adapter's
`apply(envelope, *, dataset_key, local_store, record_conflict) -> dict[str, object]`
uses the existing whole-record hash/conflict pattern and the repository writer.
A missing target conversation is a retained conflict requiring restoration of
that conversation; this record does not create a synthetic conversation.

- [ ] **1. Write a real SQLite red test and exchange canary tests.** The initial
  repository test must use the DB's actual version and preserve sibling metadata.

```python
import json
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Chat.reasoning_prefill import PrefillValue
from tldw_chatbook.Chat.reasoning_prefill_persistence import ReasoningPrefillRepository

def test_disabled_pin_round_trips_without_erasing_sibling_metadata(tmp_path):
    db = CharactersRAGDB(tmp_path / "chat.db", client_id="reasoning-prefill-test")
    try:
        conversation_id = db.add_conversation({
            "title": "Pin", "metadata": json.dumps({"active_dictionaries": [7]})
        })
        row = db.get_conversation_by_id(conversation_id)
        repository = ReasoningPrefillRepository(db)
        repository.write(conversation_id, PrefillValue(text="  Consider\n", enabled=False),
                         expected_version=row["version"])
        read, version = repository.read(conversation_id)
        assert read.value == PrefillValue(text="  Consider\n", enabled=False)
        assert version > row["version"]
        assert json.loads(db.get_conversation_by_id(conversation_id)["metadata"])["active_dictionaries"] == [7]
    finally:
        db.close_connection()
```

  Clone the real round-trip
  setup from `Tests/Chatbooks/test_chatbook_thinking_round_trip.py` and
  `Tests/Sync_Interop/test_console_thinking_sync.py`; insert the seed inside the
  conversation pin, decrypt the actual envelope, import it, and check exact bytes.
  Assert the canary is absent from clear routing fields and human-readable outputs.
- [ ] **2. Run red:** `.venv/bin/python -m pytest Tests/Chat/test_reasoning_prefill_persistence.py Tests/Chatbooks/test_reasoning_prefill_round_trip.py Tests/Sync_Interop/test_reasoning_prefill_sync.py -q`.
- [ ] **3. Implement repository writes and wire hydration/exchange.** The write
  core is a transaction-owned read/check/merge/update; do not copy the permissive
  metadata reset behavior from the older answer-prefill helper.

```python
def write(self, conversation_id, value, *, expected_version):
    from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError
    from tldw_chatbook.Chat.reasoning_prefill import merge_prefill_metadata
    with self.db.transaction():
        row = self.db.get_conversation_by_id(conversation_id)
        if row is None:
            raise ValueError("Conversation is unavailable.")
        if row["version"] != expected_version:
            raise ConflictError("Conversation changed before prefill save.")
        metadata = merge_prefill_metadata(row.get("metadata"), value)
        self.db.update_conversation(conversation_id, {"metadata": metadata},
                                    expected_version=expected_version)
        return self.db.get_conversation_by_id(conversation_id)["version"]
```

  Preserve both enabled and disabled pins through promotion and resume. Keep a
  runtime pin-save revision so an older async save result cannot overwrite newer
  intent; no implicit retry against changed ownership. Validate the owned object
  before DB mutation on imports/Sync ingress, including nested duplicate JSON
  keys, unknown versions, and unexpected properties. Add this key to the existing
  whole-conversation source intent's encrypted pin projection, never a clear
  metadata mirror. Dispatch the scoped record before `_ContinuationValidatingChatStore`,
  which currently assumes every chat envelope is a message. Verify source-intent
  version/hash before publication, make re-publication idempotent, and use the
  existing Sync state/outbox ownership. A committed local change followed by
  outbox failure remains pending for reconciliation; do not claim a cross-DB
  transaction. Reject unsupported peer capability before network publication. On
  persistent servers, block pin Save/enable and seeded sends unless round-trip
  version support is advertised. Do not infer it from generic metadata acceptance.
  Explicitly exclude live state and reservations from disk screen serialization/restore,
  defaults, forks, and serializers using `asdict`; pin hydration is the sole
  restart configuration source. Current answer-prefill behavior is unchanged.
- [ ] **4. Run green**, the three new files plus `Tests/Chat/test_thinking_conversation_exchange.py` and `Tests/Chatbooks/test_chatbook_thinking_round_trip.py`. Check the modified files with Ruff and the existing exchange validators; no new SQL migration is required by this repository design.
- [ ] **5. Commit** the scoped files and task evidence with message `feat: persist conversation reasoning prefill pins`.

## Task 3: Accepted-turn, queue, retry, and recovery ownership

**Backlog:** [TASK-32525](../../../backlog/tasks/task-32525%20-%20Bind-reasoning-prefill-to-Console-turn-and-queue-ownership.md). **Dependencies:** TASK-32521, TASK-32523, TASK-32524.

**Files:**
- Modify `Chat/console_turn_context.py`, `Chat/console_turn_preparation.py`, `Chat/console_runtime.py`, `Chat/console_dispatch_checkpoint.py`.
- Modify `Chat/console_provider_gateway.py` to introduce its pure capability query, initially returning Unverified.
- Modify `Chat/console_chat_controller.py`, `Chat/console_chat_store.py`, `Chat/console_prompt_queue.py`, `Chat/console_prompt_queue_coordinator.py`, and `UI/Console_Modules/wiring.py`.
- Create `Tests/Chat/test_reasoning_prefill_lifecycle.py` and extend `Tests/Chat/test_console_turn_execution_context.py`, `test_console_prompt_queue_coordinator.py`, and `test_console_dispatch_recovery.py`.

**Consumes:** Task 1 reservations, Task 2 live store access and hydrated pins.
**Produces:** optional `reasoning_prefill: PrefillSnapshot | None` on
`ConsoleTurnCustodyRequest`, `ConsoleTurnExecutionContext`, and retained prepared
continuations, all `repr=False`. Snapshot includes frozen absence: a queued entry
accepted without a seed must not acquire a later pin. Existing configuration
custody owns the full target; it is not duplicated into a second settings object.
Controller exposes `release_reasoning_prefill(session_id: str, submission_id: str) -> bool`
for the exact failed/undispatched owner, and rejects release while dispatch is live
or delivery is unknown. Existing recovery Discard handles unknown delivery.
`ConsoleProviderGateway.resolve_reasoning_prefill(target: PrefillTarget) ->
PrefillCapability` is introduced here with a default Unverified result;
Task 4 replaces that default with the qualification-record lookup. Introduce
the optional seed keyword on gateway preparation/stream/completion signatures
here, preserving the no-seed path byte-for-byte. It cannot cross network entry
while the target remains unqualified.

- [ ] **1. Write red behavioral tests through real runtime custody.** Extend the
  existing queue fixture (`_arm_controller`, `_queue`, `SequencedGateway`) and
  assert two entries retain different snapshots without starting the drain.

```python
from Tests.Chat.test_console_prompt_queue_coordinator import _arm_controller, _queue, SequencedGateway
from tldw_chatbook.Chat.reasoning_prefill import PrefillValue, PrefillCapability

def test_queue_acceptance_reserves_only_one_next_seed():
    gateway = SequencedGateway()
    gateway.resolve_reasoning_prefill = lambda target: PrefillCapability(
        target=target, status="supported", reason_code="qualified",
        wire_format="deepseek_prefix", echo="suffix", supports_tools=False,
        supports_response_prefill=False, evidence_ref="test-fixture")
    controller, store, session_id = _arm_controller(gateway)
    state = store.reasoning_prefill_state(session_id)
    state.set_pinned(PrefillValue(text="Pin"))
    state.set_next(PrefillValue(text="Once"))
    first = _queue(controller, session_id, "First")
    second = _queue(controller, session_id, "Second")
    assert state.snapshot(first).value.text == "Once"
    assert state.snapshot(second).value.text == "Pin"
```

  Use queue entry IDs as submission IDs for queued work; map them to existing
  preparation and assistant identities rather than inventing a second queue ID.
  Add barrier-based mounted/runtime tests for readiness rejection, failed enqueue,
  clear/rearm during a run, stopped-after-dispatch, canceled-before-dispatch,
  provider failure, failed later tool round, and unknown-delivery recovery.
  Inject a Supported test capability using the gateway interface defined below;
  do not accidentally make negative assertions pass on an unrelated support error.
- [ ] **2. Run red:** `.venv/bin/python -m pytest Tests/Chat/test_reasoning_prefill_lifecycle.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_prompt_queue_coordinator.py -q`.
- [ ] **3. Thread snapshots through existing admission and terminal owners.**
  Queue mutation and reservation commit occur without an intervening await; roll
  back the reservation if registry insertion fails. For manual admission reserve
  before the accepted turn is published, after compatibility succeeds. A failed
  support check must leave no transcript row and no lost lease. Pass snapshots
  down both direct and agent continuations rather than rereading live state.

```python
def settle_reasoning_prefill(store, session_id, submission_id, *,
                             terminal_status, provider_started):
    outcome = {
        "complete": "complete", "stopped": "stopped", "failed": "failed",
        "unknown": "unknown", "released": "released",
    }[terminal_status]
    store.reasoning_prefill_state(session_id).settle(
        submission_id, outcome=outcome, dispatched=provider_started)
```

  Call this from the owning run's existing terminal callback, never the
  `staged_input_clearing` postcommit effect or the first tool-call return. Retry
  uses retained custody, including an explicit no-seed value; regenerate captures
  the current pin with `use_next=False`; Continue uses no fresh seed. Parent
  fields are cleared when creating child/autonomous work. Queue execution must
  revalidate its frozen target; do not use the queue's existing generic
  "use current context" recovery to silently change seed or target. A deliberate
  replacement is a new submission with a new identity.
  Reuse the dispatch checkpoint's existing `prefill_reconstructable=False` when
  a seed snapshot cannot survive restart; do not persist its text or a reversible
  encoding in checkpoint metadata. Live retry is possible; cold retry is refused
  with explicit new-generation/discard actions. Test Capture On and Off separately.
- [ ] **4. Run green** plus `Tests/Chat/test_console_dispatch_recovery.py` and `Tests/UI/test_console_runtime_ownership.py`; verify seeded and successful unseeded controls both reach their intended paths. Run scoped Ruff checks.
- [ ] **5. Commit** with message `feat: bind reasoning prefill to accepted Console turns`.

## Shared native transport interface for Tasks 4–6

Create `tldw_chatbook/LLM_Calls/reasoning_prefill.py` in Task 4. It owns:

- `resolve_native_prefill(target: PrefillTarget) -> PrefillCapability`: a bounded
  lookup in application-owned qualification records. Unknown combinations never
  gain support by guessing from their model name.
- `build_native_prefill_payload(payload: Mapping[str, object], *, target:
  PrefillTarget, capability: PrefillCapability, seed: PrefillSnapshot,
  response_prefill: str | None = None) -> dict[str, object]`: detached final wire
  projection. Requires Supported target-matching capability. No input mutation.
- `ConsoleProviderGateway.resolve_reasoning_prefill(target: PrefillTarget) ->
  PrefillCapability`: the Task 3 method now delegates the lookup; tests can inject a fake implementation.
- `reasoning_prefill: PrefillSnapshot | None = None` is a keyword argument on
  gateway preparation, streaming, completion, and the bridge's captured first
  primary call. Its typed internal object never leaks into outgoing JSON.

Candidate serializers can be tested with explicitly constructed Supported test
capabilities. Production qualification records remain empty until Task 8 records
live proof. This avoids adding a user-facing "force support" escape hatch.

## Task 4: DeepSeek native prefix wire and refusal behavior

**Backlog:** [TASK-32526](../../../backlog/tasks/task-32526%20-%20Add-explicit-DeepSeek-native-reasoning-prefix-transport.md). **Dependencies:** TASK-32521, TASK-32523, TASK-32525.

**Files:**
- Create `tldw_chatbook/LLM_Calls/reasoning_prefill.py`.
- Modify `Chat/console_provider_gateway.py` at `prepare_chat_request`, `_stream_generic_chat`, `_chat_api_kwargs_from_prepared`, and `LLM_Calls/LLM_API_Calls.py` at `chat_with_deepseek` and the common caller's parameter propagation.
- Modify `LLM_Calls/hosted_chat.py` only if this execution route uses its allowlists; test the actual route rather than an unused sibling.
- Create `Tests/LLM_Calls/test_deepseek_reasoning_prefill.py`.

**Consumes/produces:** Task 1 types, Task 3 gateway keyword/query seams, and the shared native transport interface above.
The capability's DeepSeek candidate wire format is `deepseek_prefix`. The exact
supported models, echo mode, tools, and simultaneous response-prefill combinations
are evidence-backed records added only by Task 8.

- [ ] **1. Write red tests at both the pure projection and real transport call.**

```python
from tldw_chatbook.Chat.reasoning_prefill import PrefillValue, PrefillSnapshot, PrefillTarget, PrefillCapability
from tldw_chatbook.LLM_Calls.reasoning_prefill import build_native_prefill_payload

def test_deepseek_prefix_is_one_native_assistant_message():
    target = PrefillTarget(provider="deepseek", api_mode="chat_completions",
        endpoint_key="test-beta", model="test-model", thinking_enabled=True,
        tools_enabled=False, response_prefill=False)
    capability = PrefillCapability(target=target, status="supported",
        reason_code="qualified", wire_format="deepseek_prefix", echo="suffix",
        supports_tools=False, supports_response_prefill=False, evidence_ref="test-fixture")
    seed = PrefillSnapshot("turn-a", PrefillValue(text=" First,\n"), "next_send", 1)
    source = {"messages": [{"role": "user", "content": "Solve this"}]}
    wire = build_native_prefill_payload(source, target=target,
        capability=capability, seed=seed)
    assert wire["messages"][-1] == {
        "role": "assistant", "content": "", "prefix": True,
        "reasoning_content": " First,\n",
    }
    assert len(source["messages"]) == 1
```

  The synthetic model is deliberately test-only. Capture the actual HTTP request
  from `chat_with_deepseek`, after all common normalization, for stream and
  non-stream calls. Refuse a normal endpoint when the contract requires beta;
  assert the request URL is not rewritten. Test API-mode, thinking-off, tools,
  response-prefill, and evidence mismatch with a successful compatible control.
- [ ] **2. Run red:** `.venv/bin/python -m pytest Tests/LLM_Calls/test_deepseek_reasoning_prefill.py -q`.
- [ ] **3. Implement the native branch and final-wire propagation.**

```python
def deepseek_prefix_message(seed, response_prefill=None):
    return {"role": "assistant", "content": response_prefill or "",
            "prefix": True, "reasoning_content": seed.value.text}
```

  Only call this after capability and current endpoint validation. The projection
  owns both prefix fields when combining is qualified; it must not also call
  `_provider_messages_with_prefill`. Keep the configured URL and normal HTTP
  credential resolution. Use the real DeepSeek route found in this checkout;
  ADR descriptions of an adapter do not prove that a separate `deepseek.py`
  module exists. Verify the final stream and non-stream response retain reasoning
  for Task 6. Keep Responses mode Unverified unless its independent native
  contract is implemented and qualified; do not reuse the Chat payload shape.
- [ ] **4. Run green**, `Tests/LLM_Calls/test_hosted_chat.py`, and the existing affected gateway tests selected by the new keyword's call sites. Run scoped Ruff. No live provider call is needed for this candidate task.
- [ ] **5. Commit** with message `feat: add gated DeepSeek reasoning prefix transport`.

## Task 5: Local template-qualified native prefix projection

**Backlog:** [TASK-32527](../../../backlog/tasks/task-32527%20-%20Add-template-qualified-local-reasoning-prefill-transport.md). **Dependencies:** TASK-32521, TASK-32523, TASK-32526.

**Files:**
- Extend `LLM_Calls/reasoning_prefill.py` and `Chat/console_provider_gateway.py` at `build_llamacpp_chat_payload`, local streaming/completion, and prepared-request rebuilding.
- Inspect `Chat/console_provider_support.py`, `Chat/local_reasoning.py`, and `LLM_Calls/Local_Summarization_Lib.py` for the actual vLLM call path.
- Create `Tests/Chat/test_local_reasoning_prefill.py`; use pinned fixtures in `Tests/fixtures/reasoning_templates/` and add new source/fixture pairs only from verified official templates.

**Consumes:** Task 4's shared transport module and interface. **Produces:** candidate wire formats
`vllm_reasoning` and `llamacpp_reasoning`, each tied to a known server/template
contract. Local/custom endpoint identities reuse `custom_endpoint_registry.py`;
never apply a format merely because the endpoint presents as OpenAI-compatible.

- [ ] **1. Write red local boundary tests.** In `test_local_reasoning_prefill.py`,
  exercise the full projection with a synthetic test-only qualification record:

```python
from tldw_chatbook.Chat.reasoning_prefill import PrefillValue, PrefillSnapshot, PrefillTarget, PrefillCapability
from tldw_chatbook.LLM_Calls.reasoning_prefill import build_native_prefill_payload

def test_local_native_seed_keeps_thinking_and_uses_one_open_assistant():
    target = PrefillTarget(provider="vllm", api_mode="chat_completions",
        endpoint_key="local-test", model="test-template-model",
        server_version="test-pinned", template_sha256="f" * 64,
        thinking_enabled=True, tools_enabled=False, response_prefill=False)
    capability = PrefillCapability(target=target, status="supported",
        reason_code="qualified", wire_format="vllm_reasoning", echo="suffix",
        supports_tools=False, supports_response_prefill=False, evidence_ref="test-fixture")
    seed = PrefillSnapshot("turn-local", PrefillValue(text=" First,\n"), "next_send", 1)
    source = {"messages": [{"role": "user", "content": "Solve"}],
              "chat_template_kwargs": {"enable_thinking": True}}
    wire = build_native_prefill_payload(source, target=target,
                                       capability=capability, seed=seed)
    assert wire["continue_final_message"] is True
    assert wire["add_generation_prompt"] is False
    assert wire["chat_template_kwargs"]["enable_thinking"] is True
    assert len(wire["messages"]) == 2
    assert wire["messages"][-1]["reasoning"] == " First,\n"
    assert len(source["messages"]) == 1
```

  The meaningful integration test renders the complete request using the pinned
  real template and tokenizer interface for that backend. Assert the seed appears
  once at the open reasoning suffix, before neither a closing-thinking delimiter
  nor an answer-generation prefix. A flag-only test cannot qualify a template.
  Add rejects for absent/mismatched template/server identity and test ordinary
  answer-prefill behavior still forces thinking off only on its old path.
- [ ] **2. Run red:** `.venv/bin/python -m pytest Tests/Chat/test_local_reasoning_prefill.py -q`.
- [ ] **3. Implement candidates against verified template contracts.**

```python
def vllm_continuation_fields() -> dict[str, bool]:
    return {"continue_final_message": True, "add_generation_prompt": False}
```

  This is only the common flag pair. Resolve which reasoning field and delimiter
  behavior the pinned template consumes from actual render evidence before
  registering its serializer. Do not guess that the last message's answer field
  and reasoning field share the same continuation boundary. llama.cpp currently
  has a response-prefill thinking-off branch; add an explicit native-seed path
  that retains thinking and uses only verified server controls. If the server
  cannot represent an open reasoning prefix, retain an Unverified/Unsupported
  result with a precise reason and test its refusal. Do not patch user templates,
  raw-completion endpoints, or server binaries as a workaround. Check both the
  initial wire and the non-stream fallback/reprepared wire.
- [ ] **4. Run green**, `Tests/Chat/test_local_thinking_wire_formats.py`, `Tests/Chat/test_local_reasoning_gateway.py`, and `Tests/Chat/test_local_qwen_reasoning.py`. Run scoped Ruff. Record any unsupported candidate accurately instead of treating absence of a live server as proof of support.
- [ ] **5. Commit** with message `feat: gate local reasoning prefill on template capability`.

## Task 6: Generated attribution, tool history, and common input budgeting

**Backlog:** [TASK-32528](../../../backlog/tasks/task-32528%20-%20Preserve-reasoning-prefill-attribution-through-tool-replay-and-budgets.md). **Dependencies:** TASK-32521, TASK-32525, TASK-32526, TASK-32527.

**Files:**
- Create `Chat/reasoning_prefill_output.py` and `Tests/Chat/test_reasoning_prefill_output.py`.
- Modify `Chat/console_agent_bridge.py` (`_StreamingModelAdapter` and `run_reply`), `Chat/console_provider_gateway.py`, `Chat/console_thinking_capture.py`, `Chat/console_thinking_history.py`, `Chat/provider_continuation.py`, and `Chat/console_history_budget.py`.
- Modify `Chat/thinking_blocks.py` for the versioned optional replay-eligibility field and `Sync_Interop/chat_outbox_producer.py` only where its message capability/version handling changes.
- Modify `Agents/agent_service.py` only at reasoning history/counting boundaries actually used by the bridge.
- Create `Tests/Chat/test_reasoning_prefill_tool_replay.py` and `test_reasoning_prefill_budget.py`.

**Consumes:** captured seed and native projection from Tasks 1, 3–5.
**Produces:** `PrefillOutputProjector(seed: str, echo: Literal['suffix', 'full'])`
with `feed(delta: str) -> str`, `finish() -> None`, and
`continuation_text: str`. Display receives only feed's generated suffix;
continuation_text contains the exact seed-plus-generated record when required.
Full-echo mismatch or incomplete echo raises `PrefillEchoError` with no body.
`PrefillEchoError` subclasses ValueError in the same module.
`prefill_replay_eligible: bool = True` is an additive displayable-thinking block
field carried by a new thinking-envelope version; false marks a suffix that is
not a complete optional replay source. Version-1 records default true. This flag
contains no seed bytes and cannot override required provider continuation.

- [ ] **1. Write red streaming attribution and replay tests.**

```python
from tldw_chatbook.Chat.reasoning_prefill_output import PrefillOutputProjector

def test_full_echo_split_across_chunks_is_removed_only_by_declared_contract():
    output = PrefillOutputProjector(seed=" First,\n", echo="full")
    assert output.feed(" Fi") == ""
    assert output.feed("rst,\nnext") == "next"
    output.finish()
    assert output.continuation_text == " First,\nnext"

def test_suffix_contract_does_not_strip_generated_text_that_matches_seed():
    output = PrefillOutputProjector(seed="First", echo="suffix")
    assert output.feed("First again") == "First again"
    output.finish()
    assert output.continuation_text == "FirstFirst again"
```

  Add a real bridge fixture with one native tool call and a final answer. Capture
  both outbound model calls; the first contains one open seed, the second contains
  completed provider history and no newly appended seed. Include a child call
  before the primary call to prove a shared adapter counter cannot steal the seed.
  A successful first call followed by tool failure must not consume next-send.
- [ ] **2. Run red:** `.venv/bin/python -m pytest Tests/Chat/test_reasoning_prefill_output.py Tests/Chat/test_reasoning_prefill_tool_replay.py Tests/Chat/test_reasoning_prefill_budget.py -q`.
- [ ] **3. Implement explicit echo projection and exact replay ownership.**

```python
class PrefillEchoError(ValueError):
    pass

def check_echo_prefix(expected: str, received: str) -> None:
    if not expected.startswith(received):
        raise PrefillEchoError("Provider reasoning echo did not match its contract.")
```

  Keep a bounded buffer only until the declared full echo is matched; thereafter
  stream deltas directly. In suffix mode never deduplicate content. Bind first
  injection to the exact primary call identity and normal call-entry gate, not
  global model-call count. A pre-entry retry may reuse the snapshot; an ambiguous
  dispatch may not silently re-enter it. Do not turn a tool-associated reasoning
  record into a fresh standalone assistant prefix on later calls.
  Add the content-free replay-eligibility field to `thinking_blocks.py`, update
  its strict serializer/version readers and capability advertisements, and extend
  `Tests/Chat/test_thinking_blocks.py`, `Tests/Chatbooks/test_chatbook_thinking_round_trip.py`,
  and `Tests/Sync_Interop/test_console_thinking_sync.py`. Version-1 readers remain
  supported; unknown versions preserve on unrelated local writes and reject on
  incoming exchange. Auto omits an ineligible suffix; Include refuses it unless
  compatible required continuation already supplies the complete owner record.
  Use the prepared final wire projection for Context estimate, direct request,
  and agent budgets. Record provider usage unchanged; do not add seed length to
  completion tokens or display it as generated Thinking. Selected-variant
  restore/edit/delete must retain or remove the matching eligibility and replay
  owner together. Keep proprietary continuation out of the display projector.
- [ ] **4. Run green**, the three new files and `Tests/Chat/test_console_agent_call_thinking.py`, `Tests/Chat/test_console_thinking_history.py`, `Tests/Chat/test_console_thinking_capture.py`, and the three version/round-trip files above. Add canary assertions to privacy tests and run scoped Ruff.
- [ ] **5. Commit** with message `feat: preserve reasoning prefill attribution and replay`.

## Task 7: Context editor, source summary, and exact Next Send preview

**Backlog:** [TASK-32529](../../../backlog/tasks/task-32529%20-%20Expose-reasoning-prefill-editor-and-preview-in-Console.md). **Dependencies:** TASK-32521, TASK-32524, TASK-32525, TASK-32528.

**Files:**
- Create `Widgets/Console/console_reasoning_prefill_modal.py` and `Tests/UI/test_console_reasoning_prefill_modal.py`.
- Modify `Widgets/Console/console_conversation_inspector.py`, `console_send_authority_summary.py`, `Chat/console_command_grammar.py`, `Chat/console_command_suggestions.py`, `UI/Screens/chat_screen.py`, `UI/Console_Modules/wiring.py`, and `Chat/console_runtime.py`.
- Create `Tests/UI/test_console_reasoning_prefill_wiring.py`; extend `Tests/UI/test_console_runtime_ownership.py`.
- Update `Docs/User_Guide/console.md` and `Docs/User_Guide/console/chat-basics.md`.
- Change only needed source CSS modules; `css/tldw_cli_modular.tcss` is generated.

**Consumes:** state, persistence, frozen snapshots, support and budgeting.
**Produces:** `ReasoningPrefillEdit` with `session_id`, `expected_revision`,
`scope: Literal['next_send', 'pinned']`, and `value: PrefillValue | None`;
the value is repr-redacted. `ConsoleReasoningPrefillModal` calls an async host
apply callback and closes only on successful application. A failed save or stale
revision keeps the draft and focus. The host handles worker dispatch and origin
checks; the modal has no DB or provider calls. Store editor mutation has one
per-session revision covering pin and next-send writes so stale modals cannot
write into a different owner.

- [ ] **1. Write mounted red tests.** Create a minimal Textual harness that
  opens the real modal with a captured originating session, using an apply
  callback that records a `ReasoningPrefillEdit` and returns success. Then verify
  dirty dismissal and a callback failure retain the exact TextArea text. The
  production wiring test must launch the editor through the actual command and
  Context entry rather than calling its apply callback directly.

```python
from dataclasses import dataclass, field
from typing import Literal
from tldw_chatbook.Chat.reasoning_prefill import PrefillValue

@dataclass(frozen=True)
class ReasoningPrefillEdit:
    session_id: str
    expected_revision: int
    scope: Literal["next_send", "pinned"]
    value: PrefillValue | None = field(repr=False)

def test_editor_result_repr_never_contains_authored_seed():
    edit = ReasoningPrefillEdit("chat-a", 3, "next_send",
                               PrefillValue(text="PRIVATE-SEED-CANARY"))
    assert "PRIVATE-SEED-CANARY" not in repr(edit)
```

  Put the dataclass in production and import it from the test; the combined block
  specifies the exact contract, not a duplicate type to leave in the test file.
  Mounted assertions must cover both lifetimes, override revealing the pin,
  Disable retaining text, unsupported active state, reserved owner summary,
  failure release, model change, and switching chats during Save.
- [ ] **2. Run red:** `.venv/bin/python -m pytest Tests/UI/test_console_reasoning_prefill_modal.py Tests/UI/test_console_reasoning_prefill_wiring.py -q`.
- [ ] **3. Compose the editor and wire the existing host.** Use a stacked
  section for each lifetime with standard TextArea, enabled Checkbox, Save, and
  Clear. The relevant construction pattern is:

```python
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Static, TextArea

def prefill_section(scope: str, label: str, text: str, enabled: bool) -> Vertical:
    return Vertical(
        Static(label, classes="form-section-title", markup=False),
        Checkbox("Enabled", value=enabled, id=f"reasoning-{scope}-enabled"),
        TextArea(text, id=f"reasoning-{scope}-text"),
        Horizontal(Button("Save", id=f"reasoning-{scope}-save"),
                   Button("Clear", id=f"reasoning-{scope}-clear")),
        id=f"reasoning-{scope}-section",
    )
```

  Add the same modal entry under Context and `/reasoning-prefill` command metadata,
  help, and suggestions. Keep existing `/prefill reasoning...` as literal answer
  prefix text. Declare any new view hook in `CONSOLE_VIEW_HOOK_SLOTS` with its
  viewless behavior; keep its declaration and binding in the same commit.
  Summary rows show lifetime, enabled/reserved status, length, and compatibility,
  not text excerpts. The explicit Next Send panel uses the frozen preview seed
  and final projected estimate; preview never reserves or consumes a slot.
  Changes after preview invalidate/recompute it before send. Provide explicit
  Reset for unreadable/unknown pin data, preserving the raw object until the
  user chooses that destructive-to-the-pin action. Explain when clearing the
  override exposes the pin. Use the existing dirty-modal guard and token classes;
  inspect compositor output/focus at narrow and normal terminal sizes.
- [ ] **4. Run green**, `Tests/UI/test_console_runtime_ownership.py`, `Tests/UI/test_design_token_governance.py`, and `Tests/UI/test_css_bundle_sync_guard.py`. If CSS changed, first run `.venv/bin/python tldw_chatbook/css/build_css.py`. Check keyboard focus and dismissal in a real Console session; run scoped Ruff.
- [ ] **5. Commit** with message `feat: expose native reasoning prefill in Console Context`.

## Task 8: Qualification inventory and end-to-end release evidence

**Backlog:** [TASK-32530](../../../backlog/tasks/task-32530%20-%20Qualify-native-reasoning-prefill-and-Console-recovery-end-to-end.md). **Dependencies:** TASK-32521, TASK-32526, TASK-32527, TASK-32528, TASK-32529.

**Files:**
- Create `Helper_Scripts/QA/reasoning_prefill.py` and `Tests/integration/test_reasoning_prefill_end_to_end.py`.
- Create `backlog/docs/console-native-reasoning-prefill-qualification.md`.
- Update the application-owned qualification records in `LLM_Calls/reasoning_prefill.py`, affected user-guide copy, and this plan's checklist only after evidence exists.
- Read `Chat/console_provider_support.py`, `Chat/provider_readiness.py`, and custom endpoint identities for the complete inventory; test that every offered Console execution family gets a support result.

**Consumes:** the complete production path. **Produces:** recorded exact native
qualification constraints, a reproducible test command, and an honest support
inventory. No new provider configuration, credentials, network scan, or startup
probe is introduced. The helper uses existing configured provider resolution,
credential handling, and ConsoleRuntime; an explicit `--live` flag is required.

- [ ] **1. Add a hermetic production-path test before live work.** Its fixtures
  use real SQLite and ConsoleRuntime with a fake network responder, never a fake
  controller or direct serialization-only invocation. Drive editor application,
  submit, a tool round, terminal settlement, close/reopen, and the next turn.
  Capture the final HTTP bodies and assert seed injection once, complete required
  replay, generated-only display, exact persistence, and no stale consumption.
  Add this input-only inventory test to the candidate module's existing tests:

```python
from tldw_chatbook.Chat.reasoning_prefill import PrefillTarget
from tldw_chatbook.LLM_Calls.reasoning_prefill import resolve_native_prefill

def test_unrecognized_endpoint_stays_unverified():
    target = PrefillTarget(provider="custom", api_mode="chat_completions",
        endpoint_key="unregistered", model="unknown", thinking_enabled=True,
        tools_enabled=False, response_prefill=False)
    assert resolve_native_prefill(target).status == "unverified"
```

- [ ] **2. Run the hermetic release slice:** `.venv/bin/python -m pytest Tests/integration/test_reasoning_prefill_end_to_end.py Tests/Chat/test_reasoning_prefill_lifecycle.py Tests/Chat/test_reasoning_prefill_tool_replay.py Tests/UI/test_console_reasoning_prefill_wiring.py -q`. Add the pin exchange and version tests from Tasks 2 and 6 once; do not repeat the whole repo suite.
- [ ] **3. Build the explicit qualification helper and run it against configured targets.**
  Its CLI contract is `--provider`, `--model`, `--live`, `--tools`, and `--output-dir`.
  Without `--live`, validate configuration and print the required test matrix
  without sending a request. With `--live`, use a temporary test conversation
  and synthetic seed through the real runtime, with bounded token budgets and
  one existing read-only native tool. Never save credentials in evidence.
  To qualify an otherwise gated candidate, the QA helper injects a gateway
  subclass whose `resolve_reasoning_prefill` returns a test-only candidate
  capability for exactly the captured test target. All other targets delegate
  to the normal resolver. This is test dependency injection, not a saved
  capability, product setting, or ordinary-send override. Its evidence reference
  is labelled qualification-only, and candidate echo/native claims are assertions
  the run must test. Production remains Unverified until reviewed proof is added.

```python
import argparse

def qualification_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--tools", action="store_true")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()
```

  For each explicitly configured target, run both stream and non-stream cases,
  seed-only thinking through final answer, two tool rounds when tools are claimed,
  stop and retry, and response-prefill coexistence when claimed. Record exact
  provider/API mode/model/server/template, raw synthetic wire evidence with
  credentials removed, rendered suffix where available, echo contract, and final
  Console outcome. Hosted evidence also links the official native contract.
  The helper writes an evidence report, never self-promotes a capability from a
  single successful response. Inspect each report and add only the proven
  constraints to the application-owned records. Supported records must not claim
  untested arbitrary custom URLs or version ranges. If no eligible target is
  available, leave the task In Progress and all unqualified candidates gated;
  report the missing target rather than shipping an enabled claim.
- [ ] **4. Perform mounted/live acceptance and final scoped checks.** In Console,
  save and disable/re-enable both lifetimes, queue two sends, rearm while the first
  runs, navigate to a second chat, return after completion, retry a failed owner,
  and restart with a disabled pin. Confirm exact-origin persistence and no cold
  reconstruction of an ephemeral seed. Run new-file Ruff and touched-file checks,
  `git diff --check`, relevant token/bundle checks, and the existing response
  prefill regression slice. Record unrelated baseline failures separately with
  an unchanged control; never weaken tests to label the feature complete.
- [ ] **5. Commit** qualification records, evidence, tests, guide updates, and task
  notes with message `test: qualify native Console reasoning prefill`. Mark only
  genuinely completed tasks Done through Backlog CLI.

## Coverage and execution handoff

| Approved requirement | Owning tasks |
| --- | --- |
| Native-only, no silent fallback/settings change | 1, 4, 5, 8 |
| Next-send/pinned precedence and Disable | 1, 2, 7 |
| Queue reservation, failure retention, retry and cold recovery | 1, 3, 8 |
| Whitespace, size and owned metadata validation | 1, 2 |
| Pins in lossless exchange and protected Sync; ephemeral exclusion | 2, 6, 8 |
| Inject once, required replay, no child/automatic inheritance | 3, 6 |
| Generated-only thinking and exact echo contract | 4, 5, 6, 8 |
| Token budgets from final wire projection | 6, 7, 8 |
| Origin-safe editor, Context preview, keyboard/token rules | 3, 7, 8 |
| Provider inventory and actual live qualification | 4, 5, 8 |

Planning self-review covers local file paths, earlier-only Backlog dependencies,
code-block syntax, interface names, and this coverage table. The new
thinking-envelope eligibility field is the only planned versioned replay-format
extension. The scoped Sync pin record is a separate versioned wire contract;
no SQL schema change is planned. Do not conflate the pinned input object with
safe generation defaults.

This document is an implementation plan, not evidence that the feature exists.
No feature tests or live qualification ran during planning. Choose inline
execution with checkpoints or subagent-driven task execution before starting.
