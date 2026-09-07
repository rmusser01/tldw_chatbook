# Console Cost Ticker PR3 — Cost Chip + Cache-Break Alert Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the user-facing ticker: a cost chip in the Console status strip showing real conversation spend, cache state, and a passive cache-break alert, plus a cost-breakdown modal.

**Architecture:** A new pure module `Chat/console_cost_tracker.py` owns all math (cost aggregation from PR1 usage rows with estimator fallback, payload fingerprinting, break-reason diffing, TTL derivation, label formatting into a frozen `ConsoleCostState`). The store gains a per-session payload-revision counter (the lazy-recompute trigger). The controller records a fingerprint baseline at each dispatch and cache warm/TTL ground truth in `_attach_stream_usage`, and exposes accessors. The chips widget gains a 10th chip with its own equality-guarded `sync_cost_state`. The screen wires it via `_sync_console_cost_chip` (OUTSIDE the existing control-state guard), a 10-second audited TTL repaint timer, and a breakdown modal.

**Tech Stack:** Python ≥3.11, Textual (Static chips, ModalScreen), pytest.

**Spec:** `Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md` PR3 section. **Exploration dossier with verbatim current-code excerpts: `.superpowers/sdd/pr3-exploration.md` in this worktree — every task brief cites sections; implementers MUST read the cited sections before coding.**

## Global Constraints

- Worktree `/private/tmp/tldw-cost-ticker`, branch `feat/console-cost-chip` (off dev @ `db493a89d` incl. PR1+PR2). pytest ONLY via `/private/tmp/tldw-cost-ticker/.venv/bin/pytest`, FOREGROUND. NEVER `git stash`.
- **Spec-stale corrections (from the dossier — trust these over the spec's UI anchors):** chips live in `Widgets/Console/console_status_chips.py` (NOT console_control_bar.py); there are already 9 chip slots; the strip is hard-pinned to **height 1**; the container is `#console-status-chips`. The cost chip is the 10th, placed last (after `#console-scope-chip`).
- The ticker must NEVER block or delay a send: every failure degrades (catalog → tokens-only; fingerprint → suppress alert + `logger.warning`).
- Alert renders ONLY while a warm cache exists (last-send ground truth). TTL countdown + timed WARM→EXPIRED flip are Anthropic-only; monotonic clock. Warm-until = last successful Anthropic prompt-cached send + 300s.
- Fingerprint recompute: lazily, only when the session's payload revision differs from the memoized one, and NEVER while the session's run status is in `CONSOLE_ACTIVE_RUN_STATUSES`. Baseline recorded at dispatch. **Both baseline and recompute fingerprint the `_provider_messages_for_session` stage (pre-compaction/pre-window) — documented consistency decision: window/compaction drops don't alert (conservative: fewer false alarms).**
- Break-reason priority: model > system > history. Sampling params never alert. Draft typing never alerts. Tools component is deliberately omitted from the v1 fingerprint (the Console direct path sends no tools; documented in code).
- Chip label formats (height-1 strip, compact by necessity): normal `$0.4821 ●`; alert `$0.48 ⚠ +$0.13` (compact fallback `$0.48 ⚠` when the strip is narrow); cold `$0.48 ○`; no pricing `12.3k tok`; local `$0.00`. `markup=False` like every chip. Never store dollars anywhere — display-time computation only.
- New timer name `"console-cost-ttl"`: registered with `_record_ui_timer_created/_stopped`, stopped in `on_unmount` next to the transcript timer.
- CSS in SOURCE `.tcss` files only (never the generated bundle); new class `console-chip-cold`.
- No new config knob (dossier §10 confirmed).
- Line anchors verified at `db493a89d`; re-locate by symbol on drift.

---

### Task 1: Per-session payload-revision counter in the store

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Test: `Tests/Chat/test_console_chat_store.py` (append)

**Interfaces:**
- Consumes: dossier §5g (the mutation choke-point table + `_bump_message_speech_revision` precedent at :3407).
- Produces: `ConsoleChatStore.payload_revision(session_id: str) -> int` (0 for unknown/new sessions) and private `_bump_payload_revision(session_id: str) -> None`, bumped at every payload-affecting mutation listed in Step 3. Task 6 consumes `payload_revision`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Chat/test_console_chat_store.py`)

```python
def test_payload_revision_bumps_on_payload_mutations():
    from tldw_chatbook.Chat.provider_usage import ProviderUsage

    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    r0 = store.payload_revision(session.id)

    message = store.append_message(
        session.id, role=ConsoleMessageRole.USER, content="hi"
    )
    r1 = store.payload_revision(session.id)
    assert r1 > r0

    store.update_message_content(message.id, "edited")
    r2 = store.payload_revision(session.id)
    assert r2 > r1

    reply = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content="yo"
    )
    r3 = store.payload_revision(session.id)
    store.set_message_usage(
        reply.id, ProviderUsage(uncached_input=1, provider="anthropic", model="m")
    )
    # usage attach is NOT payload-affecting
    assert store.payload_revision(session.id) == r3


def test_payload_revision_bumps_on_settings_and_system_prompt():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    r0 = store.payload_revision(session.id)
    store.set_session_system_prompt(session.id, "be terse")
    r1 = store.payload_revision(session.id)
    assert r1 > r0
    store.replace_session_settings(
        session.id, replace(session.settings, model="claude-sonnet-4-6")
    )
    assert store.payload_revision(session.id) > r1


def test_payload_revision_not_bumped_per_stream_chunk():
    store = ConsoleChatStore()
    session = store.ensure_session(title="Chat 1")
    message = store.append_message(
        session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    r0 = store.payload_revision(session.id)
    store.append_stream_chunk(message.id, "a")
    store.append_stream_chunk(message.id, "b")
    assert store.payload_revision(session.id) == r0  # chunks don't churn
    store.mark_message_complete(message.id)
    assert store.payload_revision(session.id) > r0  # completion does
```

(Match the file's local import idioms; `replace` is `dataclasses.replace` — check how sibling tests build modified settings and copy that idiom exactly. `set_session_system_prompt` / `replace_session_settings` signatures: dossier §5g.)

- [ ] **Step 2: Run to verify failure**

Run: `cd /private/tmp/tldw-cost-ticker && .venv/bin/pytest Tests/Chat/test_console_chat_store.py -k payload_revision -v`
Expected: FAIL — `payload_revision` doesn't exist.

- [ ] **Step 3: Implement**

In `console_chat_store.py`: add `self._payload_revisions: dict[str, int] = {}` in `__init__` near `self._message_speech_revisions` (:468 area); add next to `_bump_message_speech_revision` (:3407):

```python
    def _bump_payload_revision(self, session_id: str) -> None:
        """Mark the session's provider payload as changed (cost-ticker PR3).

        Bumped at every mutation that can change what a future send would
        transmit; the cost chip recomputes its cache-break fingerprint only
        when this moves, so a missed bump means a stale chip (annoying), not
        a wrong send (impossible from here).
        """
        self._payload_revisions[session_id] = (
            self._payload_revisions.get(session_id, 0) + 1
        )

    def payload_revision(self, session_id: str) -> int:
        """Monotonic per-session counter of payload-affecting mutations."""
        return self._payload_revisions.get(session_id, 0)
```

Call `self._bump_payload_revision(<owning session id>)` at these sites (dossier §5g table has line anchors; message-scoped methods resolve the session via the message node — follow how `_bump_message_speech_revision` callers find ids): `append_message`, `create_sibling`, `append_generation_message`, `append_generation_variant`, `keep_generation_variant`, `replace_deferred_terminal_body`, `update_message_content`, `delete_message`, `set_active_leaf`, `set_session_context_summary`, `mark_message_complete`, `mark_message_stopped`, `mark_message_failed`, `mark_message_send_blocked`, `prepare_message_retry`, `add_variant`, `begin_variant_stream`, `finalize_variant_stream`, `select_variant`, `replace_session_settings`, `set_session_system_prompt`, `set_session_pinned_prefill`, `restore_state` (bump all restored sessions), `restore_persisted_session`. Do NOT bump in `append_stream_chunk`/`reset_stream_content` (mid-stream churn; completion covers it) or `set_message_usage`.

- [ ] **Step 4: Run to verify pass + store regression**

Run: `.venv/bin/pytest Tests/Chat/test_console_chat_store.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): per-session payload-revision counter for the cost chip"
```

---

### Task 2: `console_cost_tracker.py` — cost math + chip state formatting

**Files:**
- Create: `tldw_chatbook/Chat/console_cost_tracker.py`
- Test: `Tests/Chat/test_console_cost_tracker.py` (new)

**Interfaces:**
- Consumes: `ProviderUsage` (Chat/provider_usage.py), `get_pricing_catalog().cost_for_usage(usage) -> CostBreakdown | None` (LLM_Calls/pricing_catalog.py), `_estimate_tokens_locally(messages, model, provider) -> int` (Chat/console_session_settings.py; dossier §4).
- Produces (Tasks 4/5/6 rely on — exact):

```python
class ConsoleCacheState(str, Enum): NONE; WARM; EXPIRED

@dataclass(frozen=True)
class ConsoleCostSnapshot:
    total_usd: float | None      # None when pricing unknown for ALL priced rows
    total_tokens: int
    pricing_known: bool
    has_estimated_entries: bool
    row_count: int

def build_cost_snapshot(
    messages: Sequence[Any], *, provider: str, model: str | None
) -> ConsoleCostSnapshot

@dataclass(frozen=True)
class ConsoleCostState:
    label: str            # full chip text
    compact_label: str    # narrow-strip fallback
    tooltip: str
    alert: bool
    cold: bool

def build_cost_state(
    snapshot: ConsoleCostSnapshot,
    *,
    cache_state: "ConsoleCacheState",
    break_reason: str | None,
    projected_delta_usd: float | None,
    ttl_remaining_s: float | None,
    pricing_as_of: str | None,
) -> ConsoleCostState
```

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_cost_tracker.py
"""Cost math + chip-state formatting for the Console cost chip (PR3).

House pattern: the state dataclass owns ALL label formatting; the widget
only renders (see ConsoleControlState). Never store dollars — compute at
display time from usage rows via the pricing catalog.
"""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_cost_tracker import (
    ConsoleCacheState,
    ConsoleCostSnapshot,
    build_cost_snapshot,
    build_cost_state,
)
from tldw_chatbook.Chat.provider_usage import ProviderUsage


def _msg(content="hi", usage=None, role="assistant"):
    return SimpleNamespace(content=content, usage=usage, role=role)


def test_snapshot_sums_priced_usage_rows():
    usage = ProviderUsage(
        uncached_input=1_000_000, output=1_000_000,
        provider="anthropic", model="claude-sonnet-4-6",
    )
    snap = build_cost_snapshot(
        [_msg(usage=usage)], provider="anthropic", model="claude-sonnet-4-6"
    )
    assert snap.pricing_known is True
    assert snap.has_estimated_entries is False
    assert snap.total_usd == 18.0  # $3 in + $15 out per MTok
    assert snap.total_tokens == 2_000_000


def test_rows_without_usage_fall_back_to_estimates():
    snap = build_cost_snapshot(
        [_msg(content="x" * 400, usage=None)],
        provider="anthropic",
        model="claude-sonnet-4-6",
    )
    assert snap.has_estimated_entries is True
    assert snap.total_tokens > 0


def test_unknown_model_yields_tokens_only():
    usage = ProviderUsage(
        uncached_input=100, provider="anthropic", model="mystery-9000"
    )
    snap = build_cost_snapshot(
        [_msg(usage=usage)], provider="anthropic", model="mystery-9000"
    )
    assert snap.pricing_known is False
    assert snap.total_usd is None
    assert snap.total_tokens == 100


def test_state_normal_warm():
    snap = ConsoleCostSnapshot(0.4821, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.WARM, break_reason=None,
        projected_delta_usd=None, ttl_remaining_s=240.0, pricing_as_of="2026-08-02",
    )
    assert state.label == "$0.4821 ●"
    assert state.alert is False and state.cold is False
    assert "2026-08-02" in state.tooltip and "4:00" in state.tooltip


def test_state_alert_carries_delta_and_reason():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.WARM, break_reason="system prompt changed",
        projected_delta_usd=0.13, ttl_remaining_s=120.0, pricing_as_of="2026-08-02",
    )
    assert state.label == "$0.48 ⚠ ~+$0.13"
    assert state.compact_label == "$0.48 ⚠"
    assert state.alert is True
    assert "system prompt changed" in state.tooltip


def test_state_alert_requires_warm_cache():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.NONE, break_reason="system prompt changed",
        projected_delta_usd=0.13, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.alert is False  # no warm cache -> nothing to break


def test_state_expired_is_cold_not_alert():
    snap = ConsoleCostSnapshot(0.48, 12000, True, False, 3)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.EXPIRED, break_reason=None,
        projected_delta_usd=0.13, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.label == "$0.48 ○"
    assert state.cold is True and state.alert is False
    assert "expired" in state.tooltip.lower()


def test_state_no_pricing_shows_tokens():
    snap = ConsoleCostSnapshot(None, 12_345, False, False, 2)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.NONE, break_reason=None,
        projected_delta_usd=None, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.label == "12.3k tok"
    assert "[pricing]" in state.tooltip


def test_estimated_entries_marked_in_tooltip_and_label():
    snap = ConsoleCostSnapshot(0.10, 5000, True, True, 2)
    state = build_cost_state(
        snap, cache_state=ConsoleCacheState.NONE, break_reason=None,
        projected_delta_usd=None, ttl_remaining_s=None, pricing_as_of=None,
    )
    assert state.label.startswith("~$0.10")
    assert "estimated" in state.tooltip.lower()
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/pytest Tests/Chat/test_console_cost_tracker.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement** `tldw_chatbook/Chat/console_cost_tracker.py` (pure module; Google docstrings; loguru logger). Core behaviors the tests pin: sum `cost_for_usage` totals for rows with usage (pricing miss on ANY priced row → `pricing_known=False`, `total_usd=None` but keep token totals); rows without usage but with content → estimator tokens via `_estimate_tokens_locally([{"role": role, "content": content}], model or "", provider)` priced at the CURRENT session model's rates when known, flagging `has_estimated_entries`; formats: `$X.XXXX` 4 decimals under $1, 2 decimals ≥ $1, `~` prefix when estimated; tokens `12.3k tok` (one decimal, k at ≥1000); TTL `M:SS` in tooltip; tooltip lines: total (+estimated note), token total, cache state (+TTL / expired note / break reason + `~+$delta`), `prices as of <date>` when known, `add a [pricing] override for <model>` when unknown. Alert only when `cache_state is WARM and break_reason`; cold when EXPIRED.

- [ ] **Step 4: Run to verify pass** — `.venv/bin/pytest Tests/Chat/test_console_cost_tracker.py -v` → all PASS.

- [ ] **Step 5: Commit** — `git add tldw_chatbook/Chat/console_cost_tracker.py Tests/Chat/test_console_cost_tracker.py && git commit -m "feat(console): cost snapshot + chip state formatting module"`

---

### Task 3: Controller fingerprint baseline + cache/TTL ground truth

**Files:**
- Modify: `tldw_chatbook/Chat/console_cost_tracker.py` (fingerprint functions)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Test: `Tests/Chat/test_console_cost_tracker.py` + `Tests/Chat/test_console_chat_controller.py` (append)

**Interfaces:**
- Consumes: dossier §5a/§5b (`_provider_messages_for_session`, the dispatch site + strip site :5905-5935), §3b (`_attach_stream_usage` :6007-6065 — the single hook with resolution + summed usage), §6.
- Produces:

```python
# console_cost_tracker.py
@dataclass(frozen=True)
class PayloadFingerprint:
    provider_model: str      # digest of (provider, model)
    system: str              # digest of leading system row content ("" if none)
    history: tuple[str, ...] # per-row digest of (role, canonical-json content)

def fingerprint_payload(
    provider: str, model: str | None, provider_messages: Sequence[Mapping[str, Any]]
) -> PayloadFingerprint

def fingerprint_break_reason(
    baseline: PayloadFingerprint, current: PayloadFingerprint
) -> str | None   # None = match; else "model or provider changed" |
                  # "system prompt changed" | "earlier history changed"
                  # (priority model > system > history; a longer current
                  # history whose prefix matches baseline.history is NOT a
                  # break — that's the normal appended turn)

# controller
def payload_fingerprint_baseline(self, session_id: str) -> PayloadFingerprint | None
def compute_current_fingerprint(self, session_id: str) -> PayloadFingerprint  # via _provider_messages_for_session (pre-compaction; see Global Constraints)
def cache_ttl_snapshot(self, session_id: str) -> tuple[float | None, bool]
    # (monotonic warm_until or None, last_send_had_cache_activity)
```

- [ ] **Step 1: Write the failing tests**

Tracker tests (append to test_console_cost_tracker.py):

```python
from tldw_chatbook.Chat.console_cost_tracker import (
    PayloadFingerprint,
    fingerprint_break_reason,
    fingerprint_payload,
)


def _fp(messages, provider="anthropic", model="m"):
    return fingerprint_payload(provider, model, messages)


BASE = [
    {"role": "system", "content": "be terse"},
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
]


def test_appended_turn_is_not_a_break():
    baseline = _fp(BASE)
    current = _fp(BASE + [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}])
    assert fingerprint_break_reason(baseline, current) is None


def test_each_component_yields_its_reason_with_priority():
    baseline = _fp(BASE)
    assert fingerprint_break_reason(baseline, _fp(BASE, model="other")) == "model or provider changed"
    changed_system = [{"role": "system", "content": "be verbose"}] + BASE[1:]
    assert fingerprint_break_reason(baseline, _fp(changed_system)) == "system prompt changed"
    edited = [BASE[0], {"role": "user", "content": "EDITED"}, BASE[2]]
    assert fingerprint_break_reason(baseline, _fp(edited)) == "earlier history changed"
    # model beats system when both changed
    assert (
        fingerprint_break_reason(baseline, _fp(changed_system, model="other"))
        == "model or provider changed"
    )


def test_truncated_history_is_a_break():
    baseline = _fp(BASE)
    assert fingerprint_break_reason(baseline, _fp(BASE[:2])) == "earlier history changed"


def test_list_content_rows_hash_stably():
    rows = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    assert _fp(rows) == _fp([dict(r) for r in rows])
```

Controller test (append to test_console_chat_controller.py, reusing the local StreamingGateway idiom + UsageEmittingGateway shape from PR1's tests):

```python
@pytest.mark.asyncio
async def test_dispatch_records_fingerprint_baseline_and_cache_snapshot():
    class CacheUsageGateway(StreamingGateway):
        async def resolve_for_send(self, selection):
            resolution = await super().resolve_for_send(selection)
            resolution.provider = "anthropic"
            resolution.prompt_caching = True
            return resolution

        async def stream_chat(self, resolution, messages, **kwargs):
            signals = kwargs.get("signals")
            yield "hi"
            if signals is not None:
                signals.record_usage_payload(
                    {"input_tokens": 10, "output_tokens": 2,
                     "cache_creation_input_tokens": 900}
                )

    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=CacheUsageGateway())
    session = store.ensure_session(title="Chat 1")
    assert controller.payload_fingerprint_baseline(session.id) is None

    result = await controller.submit_draft("hello")
    assert result.accepted

    baseline = controller.payload_fingerprint_baseline(session.id)
    assert baseline is not None
    warm_until, had_activity = controller.cache_ttl_snapshot(session.id)
    assert had_activity is True
    assert warm_until is not None  # monotonic deadline stamped

    current = controller.compute_current_fingerprint(session.id)
    from tldw_chatbook.Chat.console_cost_tracker import fingerprint_break_reason
    assert fingerprint_break_reason(baseline, current) is None
```

(Adapt the gateway stub base-resolution mutation to how the file's stubs build resolutions — if the type is frozen/`type(...)`-built, construct a fresh namespace with the extra attrs instead of assigning.)

- [ ] **Step 2: Run to verify failure** — both `-k fingerprint` selections FAIL.

- [ ] **Step 3: Implement.** Tracker: digests via `hashlib.sha1(json.dumps(..., sort_keys=True, default=str).encode())`; system row = `provider_messages[0]` when its role is "system" (excluded from `history`); history digests over `(role, content)`; `fingerprint_break_reason` compares provider_model, then system, then: current.history must have baseline.history as a PREFIX (else "earlier history changed"). Controller: `self._payload_fingerprint_baselines: dict[str, PayloadFingerprint] = {}`, `self._cache_warm_until: dict[str, float] = {}`, `self._cache_last_activity: dict[str, bool] = {}`; record the baseline in `_stream_assistant_response_inner` right where `provider_messages` is final pre-compaction (dossier §5b strip-site context — record BEFORE `_apply_context_summary_compaction`, keyed by owner session id; both direct and agent paths flow through it); in `_attach_stream_usage`, after a successful attach, when `getattr(resolution, "provider", "")` normalizes to anthropic AND `getattr(resolution, "prompt_caching", None)`: set `self._cache_last_activity[sid] = total.cache_read + total.cache_write > 0` and when True `self._cache_warm_until[sid] = time.monotonic() + 300.0` (import time). Accessors per the Interfaces block; `compute_current_fingerprint` calls `_provider_messages_for_session(session_id)` + `fingerprint_payload(self.provider, self.model or self.configured_model, msgs)`. All best-effort: wrap recording in try/except with `logger.warning("cost_fingerprint_record_failed ...")` — never affect the send.

- [ ] **Step 4: Run** — `.venv/bin/pytest Tests/Chat/test_console_cost_tracker.py Tests/Chat/test_console_chat_controller.py -q` → all PASS.

- [ ] **Step 5: Commit** — `git commit -m "feat(console): fingerprint baselines and cache TTL ground truth in the controller"` (add both files + tests).

---

### Task 4: Chip in `ConsoleStatusChips` + CSS

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_status_chips.py`
- Modify: source CSS that styles `.console-control-chip` (dossier §8 names the file(s) — `css/components/_agentic_terminal.tcss` region ~:1943/2106 + wherever `console-chip-alert` is defined; add `.console-chip-cold` next to it; NEVER touch the generated bundle)
- Test: `Tests/Chat/test_console_status_chips_cost.py` (new; copy the mount/assert idiom from the existing chips tests — dossier §9 names them)

**Interfaces:**
- Consumes: `ConsoleCostState` (Task 2).
- Produces: 10th chip `#console-cost-chip` composed LAST; `ConsoleStatusChips.__init__` gains `cost_state: ConsoleCostState | None = None` (F1 precedent: initial state at compose, dossier §2e); `sync_cost_state(self, state: ConsoleCostState | None) -> None` with its own equality guard (`self._cost_state`); a `ConsoleCostChip(ConsoleChip)` subclass whose click posts a `ConsoleCostChipPressed` message (follow exactly how the existing clickable chip classes — ConsoleModelChip etc. — post theirs; dossier §1); label picks `state.compact_label` when the strip is narrow (use the same width the widget can see: `self.size.width < 120` at sync time, falling back to full label pre-layout); classes: dim by default, `console-chip-alert` when `state.alert`, `console-chip-cold` when `state.cold`; tooltip = `state.tooltip`; `None` state hides the chip (display=False) — non-Console-native contexts.

- [ ] **Step 1: Write failing tests** — mount `ConsoleStatusChips` with a cost_state (copy the harness from the existing chips test file); assert: chip exists, label text, alert class toggling across two `sync_cost_state` calls, cold class, equality guard (same state twice → single update; assert via label unchanged after mutating a copy), hidden when None, click posts the message (pilot.click + message capture, same as the existing clickable-chip test).
- [ ] **Step 2: Run** → FAIL (no chip).
- [ ] **Step 3: Implement** per Interfaces (compose after `_scope_chip()`; `_cost_chip()` builder mirroring `_chip` but with the cost classes; keep `markup=False`).
- [ ] **Step 4: Run** the new file + the whole existing chips test file → all PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(console): cost chip in the status strip"`.

---

### Task 5: Screen wiring — sync, TTL timer, breakdown modal

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Create: `tldw_chatbook/Widgets/Console/console_cost_modal.py` (pattern: ConsoleContextModal, dossier §7)
- Test: `Tests/UI/test_console_cost_chip_screen.py` (new; harness = the screen-level test idiom from dossier §9)

**Interfaces:**
- Consumes: everything above; dossier §2c (`_sync_console_control_bar` :18634 — the cost push goes OUTSIDE the `control_state_changed` guard, mirroring the inspector block), §3e/§3f/§3g (tick, timer audit, on_unmount), §2e (compose site).
- Produces:
  - `ChatScreen._build_console_cost_state() -> ConsoleCostState | None` — None for non-native sessions; else: snapshot = `build_cost_snapshot(store.messages_for_session(sid), provider=..., model=...)`; fingerprint compare ONLY when `controller.run_state_for(sid).status not in CONSOLE_ACTIVE_RUN_STATUSES` AND `store.payload_revision(sid) != self._console_cost_fp_revisions.get(sid)` (then memoize revision + reason); cache state from `controller.cache_ttl_snapshot(sid)` + `time.monotonic()` (warm_until None or activity False → NONE; deadline in future → WARM; past → EXPIRED); projection = estimator tokens over current history × (cache_write − cache_read rate) for the session model when priced, else None; `pricing_as_of` from the catalog entry when known.
  - `ChatScreen._sync_console_cost_chip()` — builds state, equality-memo `self._last_console_cost_state`, pushes via `status_chips.sync_cost_state(...)`; called at the END of `_sync_console_control_bar` (outside the guard) and from the TTL timer.
  - TTL timer: `_start_console_cost_ttl_timer` / `_stop_console_cost_ttl_timer` — `self.set_interval(10.0, ...)`, name `"console-cost-ttl"` through `_record_ui_timer_created/_stopped`; started when a built state is WARM, stopped when not (and in `on_unmount` next to `_stop_console_transcript_sync_timer()`); the tick just calls `_sync_console_cost_chip()`.
  - Modal: `ConsoleCostModal(ModalScreen)` listing per-message rows (index, role, model, bucket tokens, cost or `~est`) + totals row incl. variants; opened by the `ConsoleCostChipPressed` handler on the screen; dismiss = the context-modal idiom. Pass rows precomputed by a `build_cost_rows(messages, provider, model)` helper added to console_cost_tracker.py (returns list of frozen row dataclasses — test it in Tests/Chat/test_console_cost_tracker.py, not through the modal).
  - Compose site passes the initial cost state into `ConsoleStatusChips(...)` (F1 precedent).

- [ ] **Step 1: Write failing tests** — screen-level (copy the harness): (a) after a stub-gateway send completes, the cost chip label shows a real dollar figure (stub emits usage with priced model); (b) editing an earlier message (store.update_message_content) then syncing flips the chip to alert with "earlier history changed" in tooltip — requires a WARM cache: drive the controller cache snapshot via the stub usage carrying cache_creation tokens and prompt_caching resolution; (c) reverting the edit clears the alert (self-clearing); (d) `_build_console_cost_state` returns None when no native session; (e) TTL: monkeypatch time.monotonic forward past warm_until → state EXPIRED/cold and the timer stops; (f) fingerprint recompute is SKIPPED while run status is STREAMING (set run state, bump revision, assert memo unchanged). Plus a tracker-level test for `build_cost_rows`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** per Interfaces. Best-effort everywhere: any exception inside `_build_console_cost_state` → `logger.warning("cost_chip_state_failed ...")` and return the last-known state (or None) — never raise into the sync path.
- [ ] **Step 4: Run** the new test file + `Tests/UI/test_console_resume_active_path.py` + the chips tests → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat(console): wire the cost chip, TTL repaint, and breakdown modal"`.

---

### Task 6: Gates + live verification + push + PR

**Files:** none (verification only; screenshots land nowhere tracked).

- [ ] **Step 1: Suites** — `.venv/bin/pytest Tests/Chat/ Tests/UI/test_console_cost_chip_screen.py Tests/UI/test_console_resume_active_path.py Tests/LLM_Calls/ -q` → green (known env skips only).
- [ ] **Step 2: Live verify (REQUIRED — house rule: UI ships only after a real-terminal look).** Use the project `verify` skill / tmux recipe (launch with a scratch `TLDW_CONFIG_PATH` — NEVER the live config; stub or real provider): (a) normal size: chip renders last in the strip, dollar label after a send, tooltip on hover/focus; (b) 80×24: strip doesn't wrap/clip the cost chip off entirely — if it clips, verify the compact label engages (resize threshold) and record what was seen; (c) edit an earlier message → alert color; revert → clears. Capture findings (text notes) in the task report; any misrender is a fix-loop finding, not a note.
- [ ] **Step 3: Push + PR**

```bash
git push -u origin feat/console-cost-chip
gh pr create --base dev --title "feat(console): cost chip + cache-break alert (cost ticker PR3)" --body "PR3 of the Console cost-ticker program: the user-facing chip. Real conversation cost from PR1 usage rows (estimator-marked fallback), cache warm/cold/alert states from PR2 ground truth, fingerprint-based break detection (revision-lazy, run-suppressed, self-clearing, model>system>history reasons), Anthropic-only 10s TTL repaint (audited timer), cost-breakdown modal, height-1-strip compact labels. Live-verified incl. 80x24.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Spec-coverage checklist (PR3 section, with stale-anchor corrections)

| Spec requirement | Task |
|---|---|
| Frozen ConsoleCostState, equality-guarded sync | 2 (state) + 4 (guarded sync) |
| Chip states: warm ●, alert ⚠+delta, cold ○, tokens-only, $0 local | 2 (formats) + 4 (render) |
| Chip placed last, compact fallback, height-1 reality | 4 + 6 live verify |
| Breakdown modal with per-message rows + variants + ~ markers | 5 |
| Cost from persisted usage; estimator fallback flagged | 2 |
| Revision counter + lazy fingerprint via real builder | 1 + 3 + 5 |
| Recompute suppressed during runs; baseline at dispatch | 3 + 5(f) |
| Alert gating on WARM; self-clearing; reason priority; sampling excluded | 2 + 3 + 5 |
| TTL Anthropic-only, monotonic, 10s audited timer, per-session, unmount teardown | 3 + 5 |
| OpenAI: ground-truth warm/cold, no countdown | 3 (cache activity is provider-agnostic; countdown only stamps for anthropic+prompt_caching) |
| Projection with ~ | 5 |
| Never block sends; degrade paths | 2/3/5 (best-effort + logs) |
| No mid-stream cost animation | 5 (revision not bumped per chunk + run-suppressed recompute; cost rebuild rides existing syncs) |
| CSS in source .tcss; console-chip-cold | 4 |
| 80×24 live check | 6 |
