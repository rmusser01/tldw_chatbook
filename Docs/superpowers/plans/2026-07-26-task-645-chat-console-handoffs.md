# Chat and Console Handoff Ownership Implementation Plan (TASK-645)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three raw Chat/Console pending fields with typed, memory-only, revisioned single-slot handoffs that cannot lose replacements and cannot leave duplicate partial Chat tabs after failure or cancellation.

**Architecture:** Add a navigation-layer `PendingHandoffStore` with independent typed channels, one in-flight claim and at most one latest replacement per channel. The store normalizes and structurally detaches values at staging and again at claim delivery, retaining an internal pristine value for release. Producers stage before navigation; consumers claim only when their lifecycle prerequisites are available, then acknowledge terminal/success outcomes or release transient/cancelled outcomes.

**Tech Stack:** Python 3.11+, dataclasses/generics, `StrEnum`, `copy.deepcopy`, `threading.get_ident`, asyncio cancellation, Textual mount/resume lifecycle, pytest/pytest-asyncio.

**Backlog:** [TASK-645](../../../backlog/tasks/task-645%20-%20Move-Chat-and-Console-handoffs-behind-revisioned-single-slot-ownership.md)

**Specification:** [Application Session State Ownership Design](../specs/2026-07-26-application-session-state-ownership-design.md)

**Depends on:** TASK-644

**ADR required:** yes

**ADR path:** `backlog/decisions/026-application-session-state-ownership.md`

**Reason:** ADR-026 already defines the cross-screen single-slot delivery, replacement, settlement, privacy, and thread-affinity contract.

---

## Execution Environment

This worktree has no `.venv`, and `/usr/bin/python3` is Python 3.9. Before
running any command in this plan:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate
python -c "import pathlib, tldw_chatbook; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

The printed path must be inside
`.../.worktrees/privacy-lifecycle-eval-wheel-hardening/tldw_chatbook`, not the
main checkout or site-packages. The verified environment is Python 3.12.11,
pytest 8.4.2, and Ruff 0.15.22.

## File Structure

- Create `tldw_chatbook/UI/Navigation/pending_handoff_store.py`: channel enum, opaque claims, typed normalization/copying, and revisioned claim settlement.
- Modify `tldw_chatbook/UI/Navigation/__init__.py`: export the intended handoff types if this package uses explicit exports.
- Modify `tldw_chatbook/app.py`: construct the store and migrate Chat context, Console live work, and Console prompt producers.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: migrate all three consumers and add exact ephemeral-tab rollback.
- Modify `tldw_chatbook/Chat/console_live_work.py`: ensure nested launch payload reconstruction is structurally detached; do not change visible serialization.
- Create `Tests/UI/test_pending_handoff_store.py`: protocol, copying, thread-affinity, memory-only, and privacy tests.
- Modify `Tests/UI/test_chat_first_handoffs.py`: Chat producer/consumer settlement and rollback/cancellation tests.
- Modify `Tests/UI/test_console_live_work_handoffs.py`: Console launch producer/transfer and mounted behavior.
- Modify `Tests/UI/test_console_command_composer.py`: prompt success, terminal, transient, replacement, and resume behavior.
- Modify `Tests/UI/test_ux_audit_smoke.py` and `Tests/UI/test_product_maturity_phase1_core_loop.py`: mounted smoke fixtures use the owner.
- Modify `Tests/test_application_state_ownership.py`: reject the three raw app fields and direct bypasses.

## Task 1: Implement the Revisioned Single-Slot Protocol

**Files:**

- Create: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Modify: `tldw_chatbook/UI/Navigation/__init__.py`
- Modify: `tldw_chatbook/Chat/console_live_work.py`
- Create: `Tests/UI/test_pending_handoff_store.py`

- [ ] **Step 1: Write failing protocol tests**

Cover the state machine without sleeps:

```python
def test_release_does_not_overwrite_newer_replacement() -> None:
    store = PendingHandoffStore()
    first = _chat_payload("first")
    second = _chat_payload("second")
    store.stage(HandoffChannel.CHAT, first)
    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None

    store.stage(HandoffChannel.CHAT, second)

    assert store.release(claim) is True
    replacement = store.claim(HandoffChannel.CHAT)
    assert replacement is not None
    assert replacement.value.title == "second"
    assert store.acknowledge(claim) is False


def test_claim_value_and_released_value_are_detached_from_mutation() -> None:
    source = _chat_payload("context")
    source.metadata["nested"] = {"items": ["original"]}
    store = PendingHandoffStore()
    store.stage(HandoffChannel.CHAT, source)
    source.metadata["nested"]["items"].append("producer-change")

    claim = store.claim(HandoffChannel.CHAT)
    assert claim is not None
    claim.value.metadata["nested"]["items"].append("consumer-change")
    assert store.release(claim) is True
    retry = store.claim(HandoffChannel.CHAT)

    assert retry.value.metadata["nested"]["items"] == ["original"]
```

Also test: stage returns monotonic channel-local revisions; stage replaces an unclaimed value; second claim while in flight returns `None`; acknowledge removes only the identical current claim; releasing with no replacement makes the same revision claimable again; only the latest of several in-flight replacements remains; stale claim objects cannot settle a re-claimed same revision; invalid values leave no partial slot; Chat mappings rebuild through `to_dict`/`from_dict`; Console nested payloads are deep-copied; prompt accepts only non-empty strings while preserving the user's text; all mutations reject off-owner threads.
Add a `clear_pending()` case: clearing while an older claim is in flight
advances the channel revision, leaves no pending value, and prevents release
from resurrecting the older retained value.

- [ ] **Step 2: Run tests and verify the store is absent**

Run:

```bash
pytest Tests/UI/test_pending_handoff_store.py -q
```

Expected: FAIL on import.

- [ ] **Step 3: Implement channel, claim, and private slot state**

Start with the three TASK-645 channels:

```python
class HandoffChannel(StrEnum):
    CHAT = "chat"
    CONSOLE_LIVE_WORK = "console_live_work"
    CONSOLE_PROMPT_INSERT = "console_prompt_insert"


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class HandoffClaim(Generic[T]):
    channel: HandoffChannel
    revision: int
    value: T = field(repr=False, compare=False)


@dataclass(slots=True)
class _InFlight:
    claim: HandoffClaim[Any]
    retained_value: Any


@dataclass(slots=True)
class _Slot:
    revision: int = 0
    pending: tuple[int, Any] | None = None
    in_flight: _InFlight | None = None
```

`PendingHandoffStore` creates one `_Slot` per enum member and captures `threading.get_ident()`. Do not provide serialization, iteration over payloads, a raw backing-map property, or loggable claim representations containing values.

- [ ] **Step 4: Implement normalize, stage, claim, acknowledge, and release**

Normalization rules:

```python
def _copy_value(channel: HandoffChannel, value: Any) -> Any:
    if channel is HandoffChannel.CHAT:
        copied = ChatHandoffPayload.from_dict(value)
        if copied is None:
            raise ValueError("invalid Chat handoff")
        return copied
    if channel is HandoffChannel.CONSOLE_LIVE_WORK:
        copied = ConsoleLiveWorkLaunch.from_pending(value)
        if copied is None:
            raise ValueError("invalid Console launch")
        return ConsoleLiveWorkLaunch.from_values(
            source=copied.source,
            title=copied.title,
            payload=copy.deepcopy(copied.payload),
            status=copied.status,
            recovery=copied.recovery,
            action_label=copied.action_label,
        )
    if channel is HandoffChannel.CONSOLE_PROMPT_INSERT:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Console prompt must be non-empty text")
        return value
    raise ValueError("unsupported handoff channel")
```

`stage()` stores a normalized copy and advances only that channel's revision.
`clear_pending()` advances that revision and sets `pending=None`; it does not
cancel an existing in-flight consumer, but marks its value as superseded.
`claim()` retains the stored value privately but gives the consumer a second
`_copy_value()` result. `acknowledge()` and `release()` must compare the exact
current claim object by identity, not only its forgeable channel/revision
fields:

```python
def release(self, claim: HandoffClaim[Any]) -> bool:
    self._assert_owner_thread()
    slot = self._slot_for(claim.channel)
    current = slot.in_flight
    if current is None or current.claim is not claim:
        return False
    slot.in_flight = None
    if slot.revision == claim.revision:
        slot.pending = (claim.revision, current.retained_value)
    return True
```

When a newer replacement exists, release clears the old in-flight claim and
preserves the replacement. When a newer clear revision exists, release clears
the old in-flight claim and leaves the channel empty. This is one old
in-flight value plus one latest replacement or clear, never a queue.

- [ ] **Step 5: Make Console launch reconstruction deep-copy nested payloads**

Change only the copy boundary in `ConsoleLiveWorkLaunch.from_values()`/`to_pending_payload()` as needed so callers cannot mutate nested structures through a frozen dataclass's mutable `payload`. Preserve all normalized strings and visible output.

- [ ] **Step 6: Run protocol and Console model tests**

Run:

```bash
pytest Tests/UI/test_pending_handoff_store.py Tests/UI/test_console_live_work_handoffs.py -q -k "model or payload or pending_handoff_store"
```

Expected: PASS.

- [ ] **Step 7: Commit the protocol**

```bash
git add tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/Chat/console_live_work.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_console_live_work_handoffs.py
git commit -m "feat(navigation): add revisioned handoff owner (task-645)"
```

## Task 2: Migrate Chat and Console Producers

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Tests/UI/test_console_command_composer.py`

- [ ] **Step 1: Update producer tests to require stage-before-navigation**

Construct `app.pending_handoffs = PendingHandoffStore()` in app-like fixtures. Assert each producer:

1. stages the correct normalized channel;
2. posts `NavigateToScreen(TAB_CHAT)` only after successful staging;
3. preserves the Chat tabs-enabled gate before staging;
4. warns and does not navigate if normalization/copying fails;
5. replaces an existing unclaimed value rather than queueing.

Patch `post_message` with a side effect that claims the channel, proving the value exists before navigation publication.

- [ ] **Step 2: Run producer tests to verify raw fields fail the contract**

Run:

```bash
pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py -q -k "stores_payload or producer or stage or navigate"
```

Expected: FAIL while producers assign raw app attributes.

- [ ] **Step 3: Construct the store and migrate producers**

In `TldwCli.__init__`:

```python
self.pending_handoffs = PendingHandoffStore()
```

Delete initialization of `pending_chat_handoff`, `pending_console_launch`, and `pending_console_prompt_insert`.

Migrate:

```python
self.pending_handoffs.stage(HandoffChannel.CHAT, payload)
self.pending_handoffs.stage(HandoffChannel.CONSOLE_PROMPT_INSERT, text)
self.pending_handoffs.stage(
    HandoffChannel.CONSOLE_LIVE_WORK,
    ConsoleLiveWorkLaunch.from_values(...),
)
```

Catch only normalization/copy errors at this boundary, emit bounded recovery without the value, and do not post navigation on failure. Off-thread production callers must use Textual's existing `app.call_from_thread()` boundary; do not weaken store affinity.

- [ ] **Step 4: Run producer suites**

Run:

```bash
pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py -q -k "open_chat or open_console or stage_console"
```

Expected: PASS.

- [ ] **Step 5: Commit producer migration**

```bash
git add tldw_chatbook/app.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py
git commit -m "refactor(app): stage Chat and Console handoffs through owner (task-645)"
```

## Task 3: Migrate Console Launch and Prompt Consumers

**Files:**

- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_console_live_work_handoffs.py`
- Modify: `Tests/UI/test_console_command_composer.py`

- [ ] **Step 1: Write failing settlement tests**

Console live work:

- a claim transfers to `_pending_console_launch_context` and acknowledges immediately;
- normalization is already complete before the screen receives it;
- a second consumer cannot claim while transfer is in flight;
- a newer replacement staged during transfer remains pending.

Prompt insert:

- success acknowledges after append;
- setup blocked warns and releases for an existing mount/resume or
  user-triggered retry after setup is completed;
- missing composer/readiness releases for mount/resume retry;
- cancellation releases and re-raises;
- unexpected failure releases and logs only channel/revision/exception category;
- a replacement staged while the consumer holds a claim survives the older settlement.

- [ ] **Step 2: Run consumer tests to verify direct clearing**

Run:

```bash
pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py -q -k "pending or replacement or release or acknowledge or resume"
```

Expected: new replacement-race cases FAIL because consumers clear raw fields.

- [ ] **Step 3: Transfer Console live work through a claim**

Replace `_consume_pending_console_launch()` raw reads with:

```python
claim = self.app_instance.pending_handoffs.claim(
    HandoffChannel.CONSOLE_LIVE_WORK
)
if claim is None:
    return self._pending_console_launch_context
self._pending_console_launch_context = claim.value
self._pending_console_launch_auto_open_inspector = True
self.app_instance.pending_handoffs.acknowledge(claim)
return self._pending_console_launch_context
```

If assignment unexpectedly fails, release and emit a metadata-only diagnostic.

- [ ] **Step 4: Settle prompt insertion by outcome**

Claim at method entry. A blank claim should be unreachable but remains
terminal. Keep `_sync_console_session_draft()` immediately before insertion.
A blocked setup is an incomplete-readiness outcome: warn and release, leaving
the claim available for an existing mount/resume or explicit user-triggered
retry after configuration changes. Missing composer likewise releases.
Success acknowledges before focus work. Update the existing blocked-setup
test so it asserts the draft remains untouched and the released handoff can be
claimed and applied after setup becomes ready; do not preserve the old
raw-field assertion that setup permanently consumes the intent. Use:

```python
except asyncio.CancelledError:
    store.release(claim)
    raise
except Exception as exc:
    store.release(claim)
    logger.warning(
        "Console prompt handoff failed "
        "(channel={}, revision={}, exception_category={})",
        claim.channel.value,
        claim.revision,
        type(exc).__name__,
    )
```

Never log `claim`, `claim.value`, composer text, or traceback locals.

- [ ] **Step 5: Run Console mounted flows**

Run:

```bash
pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Console consumption**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py
git commit -m "refactor(console): settle owned handoff claims (task-645)"
```

## Task 4: Make Chat Consumption Transactional with Exact Tab Rollback

**Files:**

- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `Tests/UI/test_ux_audit_smoke.py`

- [ ] **Step 1: Write deterministic failure/cancellation injection tests**

Use `asyncio.Event` barriers or direct async side effects, never sleeps, for each boundary:

1. cancellation before `create_new_tab()` returns: release, no cleanup;
2. false/empty tab ID: warn and release;
3. switch failure after exact ID creation: close that ID, then release;
4. cancellation during switch/apply: close exact ID, release, re-raise;
5. missing created session: close exact ID, release;
6. apply failure: close exact ID, release;
7. `close_tab()` raises or leaves the exact ID in `sessions`: acknowledge terminally, warn once, no retry duplicate;
8. success acknowledges immediately after apply, before an injected unrelated awaited UI continuation;
9. native Console transfer acknowledges after `_stage_handoff_as_console_live_work()`;
10. replacement staged while the old claim awaits remains the next pending value.

Assert cleanup never closes a pre-existing tab and always receives the exact ID returned by creation.

- [ ] **Step 2: Run rollback tests and verify the current partial application**

Run:

```bash
pytest Tests/UI/test_chat_first_handoffs.py -q -k "cancel or rollback or cleanup or replacement or partial"
```

Expected: FAIL because current code retains the raw pending value and partial tab after post-creation failures.

- [ ] **Step 3: Add one exact rollback helper**

```python
async def _rollback_chat_handoff_tab(
    self,
    tab_container: ChatTabContainer,
    tab_id: str,
) -> bool:
    try:
        await tab_container.close_tab(tab_id)
    except Exception as exc:
        logger.warning(
            "Chat handoff tab cleanup failed (exception_category={})",
            type(exc).__name__,
        )
        return False
    return tab_id not in tab_container.sessions
```

This uses the public `close_tab()` contract. It does not scan by title, payload, active tab, or "latest" tab.

- [ ] **Step 4: Implement explicit claim settlement**

Claim once after the existing mount/setup gate. Track `created_tab_id: str | None` and `settled = False`. On success:

```python
await self._apply_handoff_to_chat_session(session, payload)
store.acknowledge(claim)
settled = True
```

For failures/cancellation after `created_tab_id` exists, call the rollback helper first. If cleanup succeeds, release. If cleanup fails, acknowledge and show bounded recovery so the same intent cannot create another tab. Before a tab ID exists, release directly. Re-raise `CancelledError` after settlement. Native Console transfer acknowledges immediately after screen-local ownership is established.

Do not use a `finally` block that can settle a newer claim; settle only the exact local claim object.

- [ ] **Step 5: Run Chat unit and mounted smoke tests**

Run:

```bash
pytest Tests/UI/test_chat_first_handoffs.py Tests/UI/test_ux_audit_smoke.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit transactional Chat consumption**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_ux_audit_smoke.py
git commit -m "fix(chat): roll back failed owned handoffs (task-645)"
```

## Task 5: Add Ownership and Privacy Guards

**Files:**

- Modify: `Tests/UI/test_pending_handoff_store.py`
- Modify: `Tests/UI/test_chat_first_handoffs.py`
- Modify: `Tests/UI/test_console_command_composer.py`
- Modify: `Tests/test_application_state_ownership.py`
- Modify: `Tests/UI/test_product_maturity_phase1_core_loop.py`

- [ ] **Step 1: Add log-redaction sentinels**

Stage unique secrets in nested Chat metadata, Console payload, and prompt text. Force normalization, consumer, rollback, and cleanup failures. Capture Loguru/standard logs and assert the sentinel is absent while stable channel/revision/outcome/exception-category metadata remains.

- [ ] **Step 2: Extend the AST ownership guard**

Reject production:

- attributes named `pending_chat_handoff`, `pending_console_launch`, or `pending_console_prompt_insert`;
- direct access to `PendingHandoffStore._slots`;
- payload persistence or serialization calls in the handoff module;
- logging calls whose arguments include `claim`, `.value`, payload, prompt text, or object reprs in handoff exception paths.

Allow the owner module's private internals and test fixtures only.

- [ ] **Step 3: Update the product-maturity smoke fixture**

Replace its raw pending field with `PendingHandoffStore`, stage through `HandoffChannel.CHAT`, and retain the visible first-send assertions. Do not change the product behavior being tested.

- [ ] **Step 4: Run guard and sentinel tests**

```bash
pytest Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_command_composer.py Tests/UI/test_product_maturity_phase1_core_loop.py Tests/test_application_state_ownership.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit guards**

```bash
git add Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_command_composer.py Tests/UI/test_product_maturity_phase1_core_loop.py Tests/test_application_state_ownership.py
git commit -m "test(handoffs): guard Chat and Console ownership (task-645)"
```

## Task 6: Verify TASK-645 and Hold Final Reconciliation

**Files:**

- No production or Backlog status changes expected; fix only verified
  regressions within TASK-645 acceptance criteria.

- [ ] **Step 1: Run focused and mounted verification**

```bash
pytest Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py Tests/UI/test_ux_audit_smoke.py Tests/UI/test_product_maturity_phase1_core_loop.py Tests/test_application_state_ownership.py -q
python -m compileall -q tldw_chatbook/UI/Navigation tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_live_work.py
python -m ruff check tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_live_work.py tldw_chatbook/app.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py Tests/test_application_state_ownership.py
python -m ruff format --check tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Navigation/__init__.py tldw_chatbook/Chat/console_live_work.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_chat_first_handoffs.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_console_command_composer.py Tests/test_application_state_ownership.py
git diff --check
```

Expected: all commands exit 0. The format gate intentionally excludes the
verified pre-tranche unformatted `app.py` and `chat_screen.py`; do not create a
large unrelated formatting diff.

- [ ] **Step 2: Self-review all six acceptance criteria**

Confirm exact evidence for structural detachment, exclusive claims, replacement survival, stale settlement rejection, owner-thread enforcement, cancellation propagation, exact tab rollback, terminal cleanup failure, mounted Chat/Console behavior, memory-only storage, and sentinel redaction.

- [ ] **Step 3: Preserve the In Progress status until integrated gates**

Use `backlog task 645 --plain` to confirm the plan and acceptance criteria
still match the implemented code, but leave all criteria unchecked and keep
TASK-645 In Progress. Do not add final Implementation Notes or update the
design status yet. Final reconciliation waits for TASK-646's shared
installed-wheel, product-maturity, static, and full-suite evidence.
