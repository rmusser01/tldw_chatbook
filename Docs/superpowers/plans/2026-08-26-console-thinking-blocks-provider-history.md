# Console Thinking Provider and History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn only adapter-approved current-response reasoning into explicit Console evidence and replay complete compatible displayable thinking through the exact prepared request used for token accounting and dispatch.

**Architecture:** Evolve the local start-anchored filter into a two-channel splitter, add two narrow provider stream event types, and make each adapter declare evidence disposition. Direct and agent consumers feed one round-aware accumulator owned by the assistant generation. History preparation joins visible messages, optional displayable thinking, and mandatory ADR-063 continuation into owner-atomic units; a target-specific serializer freezes one wire payload that both accounting and dispatch consume.

**Tech Stack:** Python 3.11+, async generators, dataclasses/`Literal`, existing hosted/local LLM adapters, immutable prepared requests, pytest/pytest-asyncio.

**Spec:** `Docs/superpowers/specs/2026-08-26-console-thinking-blocks-design.md`

**Task:** `backlog/tasks/task-18932.2 - Normalize-provider-thinking-events-and-history-replay.md`

## Global Constraints

- TASK-18932.1 must be complete; import its envelope/policy/capability contracts directly.
- Stream strings remain visible answer content. Thinking and proprietary occurrence travel only as typed events.
- Default adapter disposition is `ignored`. Shared gateway code never interprets a generic `reasoning_content`, `analysis`, `thinking`, timing, token count, or empty answer as evidence.
- Initial explicit classifications are: llama.cpp/vLLM-compatible start-anchored think sections are displayable; Moonshot and Z.ai `reasoning_content` handled by ADR-063 is proprietary evidence; every other field remains ignored until its own adapter contract test proves a user-displayable API surface.
- A proprietary event contains provider/model/protocol/source identity and occurrence only. It cannot carry raw text, length, hash, token count, or excerpt.
- Keep existing Moonshot/Z.ai private continuation replay authoritative. Optional thinking history never duplicates private reasoning from continuation.
- Complete displayable blocks alone are optional replay candidates. Stopped, failed, proprietary, malformed, opaque-version, or incompatible blocks remain durable/UI facts but never optional history.
- `include` is strict only for otherwise replay-eligible blocks. Incompatible historical blocks from unrelated providers are omitted; a block that the fully resolved target claims as compatible but cannot safely serialize blocks before provider contact.
- One frozen provider-prepared artifact must be counted and dispatched. No second transformation after accounting.
- Keep the compatibility `StartAnchoredThinkFilter` wrapper until every existing caller/test migrates; it delegates to the splitter's visible channel and preserves current non-Console behavior.

---

### Task 1: Replace start-anchored stripping with a safe streaming splitter

**Files:**
- Modify: `tldw_chatbook/Chat/llamacpp_think_filter.py`
- Modify: `Tests/Chat/test_llamacpp_think_filter.py`
- Create: `Tests/Chat/test_llamacpp_think_splitter.py`

**Interfaces consumed:** existing `StartAnchoredThinkFilter.feed/flush` behavior.

**Interfaces produced:** `ThinkSplitChunk`, `StartAnchoredThinkSplitter`, terminal capture status, compatibility filter wrapper.

- [ ] **Step 1: Write failing chunk-boundary matrix tests.** Parametrize every split position across `<think>`, `<thinking>`, and both closing tags. Cover leading spaces/newlines, empty thinking, answer in same/later chunk, literal mid-answer tag, partial opening tag, mismatched close, and unclosed EOF.

```python
@pytest.mark.parametrize("cut", range(len("<think>reason</think>answer") + 1))
def test_splitter_is_chunk_boundary_invariant(cut: int) -> None:
    raw = "<think>reason</think>answer"
    splitter = StartAnchoredThinkSplitter()
    chunks = [splitter.feed(raw[:cut]), splitter.feed(raw[cut:]), splitter.flush()]
    assert "".join(item.thinking for item in chunks) == "reason"
    assert "".join(item.content for item in chunks) == "answer"
    assert chunks[-1].status == "complete"
```

- [ ] **Step 2: Run splitter tests and confirm missing API failure.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_llamacpp_think_splitter.py Tests/Chat/test_llamacpp_think_filter.py -q`

Expected: FAIL because the splitter/result type does not exist.

- [ ] **Step 3: Implement a bounded finite-state splitter.** Never emit a pending partial opener. Before confirming a start-anchored opener, buffer only leading whitespace plus the maximum tag length. Once visible answer starts, pass all later `<think>` text literally.

```python
ThinkCaptureStatus = Literal["pending", "complete", "failed"]

@dataclass(frozen=True, slots=True)
class ThinkSplitChunk:
    thinking: str = field(default="", repr=False)
    content: str = field(default="", repr=False)
    status: ThinkCaptureStatus = "pending"

class StartAnchoredThinkSplitter:
    def feed(self, chunk: str) -> ThinkSplitChunk:
        if type(chunk) is not str:
            raise TypeError("Thinking stream chunks must be strings.")
        return self._consume(chunk, terminal=False)

    def flush(self) -> ThinkSplitChunk:
        return self._consume("", terminal=True)
```

On unclosed start-anchored thinking, `flush()` returns status `failed`, no visible reclassification, and any captured thinking remains available only to settle a failed block. Enforce live envelope byte bounds while accumulating; exceeding them raises the content-free capture error defined by TASK-18932.1.

- [ ] **Step 4: Retain the old filter as a wrapper.** It returns only `.content`. At terminal failure it preserves today's privacy behavior by dropping the unclosed thinking section rather than leaking it into visible content.

- [ ] **Step 5: Add non-streaming helper parity.** `split_start_anchored_thinking(text)` must feed once then flush, returning the same combined result/status as every streaming partition.

- [ ] **Step 6: Run both splitter and compatibility suites.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_llamacpp_think_splitter.py Tests/Chat/test_llamacpp_think_filter.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the splitter.**

```bash
git add tldw_chatbook/Chat/llamacpp_think_filter.py Tests/Chat/test_llamacpp_think_filter.py Tests/Chat/test_llamacpp_think_splitter.py
git commit -m "feat: split local model thinking from visible answers"
```

---

### Task 2: Add explicit provider evidence events and adapter dispositions

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py`
- Modify: `tldw_chatbook/LLM_Calls/moonshot.py`
- Modify: `tldw_chatbook/LLM_Calls/zai.py`
- Modify: `Tests/Chat/test_console_provider_gateway.py`
- Modify: `Tests/LLM_Calls/test_hosted_chat.py`
- Modify: `Tests/LLM_Calls/test_moonshot.py`
- Modify: `Tests/LLM_Calls/test_zai.py`
- Modify: `Tests/Chat/test_kimi_zai_native_tools.py`
- Modify: `Tests/Chat/test_kimi_zai_provider_contract.py`
- Modify: `Tests/Chat/test_local_adapter_thinking_dispatch.py`
- Modify: `Tests/Chat/test_local_thinking_wire_formats.py`

**Interfaces consumed:** Task 1 splitter; frozen `ProviderResolution`; hosted finish policies.

**Interfaces produced:** typed stream items, adapter disposition contract, adapter-owned `may_emit_thinking` persistence-preflight fact.

- [ ] **Step 1: Write failing event contract tests.** Assert a displayable event keeps text out of repr, a proprietary event has no text-shaped fields, all events freeze provider/model/protocol/source, and `ProviderStreamItem` accepts current strings/tool calls unchanged.

```python
@dataclass(frozen=True, slots=True)
class ProviderThinkingDelta:
    text: str = field(repr=False)
    provider: str
    model: str
    protocol: str
    source_format: str

@dataclass(frozen=True, slots=True)
class ProviderProprietaryThinkingEvidence:
    provider: str
    model: str
    protocol: str
    source_format: str
```

- [ ] **Step 2: Run gateway contract tests and confirm missing event failure.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py -k thinking -q`

Expected: FAIL.

- [ ] **Step 3: Add an explicit disposition policy.** Extend provider-owned finish/stream policy with a content-free enum and no default inference.

```python
ReasoningDisposition = Literal["displayable", "proprietary", "ignored"]

class HostedChatFinishPolicy(Protocol):
    reasoning_disposition: ReasoningDisposition

    def validate_reasoning_content(self, value: object) -> str | None:
        """Validate provider wire data without deciding generic visibility."""
```

Moonshot/Z.ai set `reasoning_disposition="proprietary"`. Existing private reasoning remains in `HostedChatTurn.reasoning_content` with `repr=False` for continuation construction, while the Console adapter emits exactly one content-free evidence event when that current turn contains validated non-empty reasoning. Do not expose the value through the event.

- [ ] **Step 4: Convert direct llama.cpp/vLLM-compatible paths to displayable events.** Replace both `StartAnchoredThinkFilter` instances in `console_provider_gateway.py` with the splitter. Yield each non-empty reasoning channel as `ProviderThinkingDelta`; yield visible content unchanged. On terminal splitter failure, emit no answer reclassification and surface a content-free terminal capture failure to the consumer.

- [ ] **Step 5: Add adapter-owned capability declaration.** The resolved provider target carries `thinking_stream_disposition` and `thinking_round_trip_version`. Set displayable for direct local start-anchored adapters, proprietary for Moonshot/Z.ai paths that can return validated reasoning, and ignored otherwise. The foundation preflight uses `disposition != "ignored"`; it does not inspect the model name or request settings.

- [ ] **Step 6: Add no-evidence and negative controls.** For every disposition, test a capable/enabled response with no actual reasoning field/tag and assert no event. Mutate a generic unknown provider to return `reasoning_content` and assert ignored. Verify logs/caplog contain neither displayable nor private canaries.

- [ ] **Step 7: Run provider suites.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_kimi_zai_native_tools.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_local_thinking_wire_formats.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py -q`

Expected: PASS.

- [ ] **Step 8: Commit provider evidence.**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/zai.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_kimi_zai_native_tools.py Tests/Chat/test_kimi_zai_provider_contract.py Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/test_local_thinking_wire_formats.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_moonshot.py Tests/LLM_Calls/test_zai.py
git commit -m "feat: emit explicit provider thinking evidence"
```

---

### Task 3: Accumulate round-owned thinking in direct and agent turns

**Files:**
- Create: `tldw_chatbook/Chat/console_thinking_capture.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Create: `Tests/Chat/test_console_thinking_capture.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`

**Interfaces consumed:** provider stream events and foundation selected-generation APIs.

**Interfaces produced:** ordered block lifecycle events/state for store and UI; identical direct/agent accumulation.

- [ ] **Step 1: Write failing pure accumulator tests.** Cover multiple deltas in one model round, multiple rounds separated by tool events, proprietary occurrence de-duplication, answer-first collapse boundary fact, tool-first boundary, terminal-only proprietary evidence, completion, stop, failure, byte/block overflow, and no evidence.

```python
def test_no_event_means_no_recorded_evidence() -> None:
    capture = ThinkingCapture(provider="local", model="plain", protocol="chat")
    capture.observe_answer("answer")
    assert capture.settle("complete").blocks == ()
```

- [ ] **Step 2: Run the accumulator test and confirm missing module failure.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_thinking_capture.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement an event-driven accumulator.** Only `observe_thinking_delta` and `observe_proprietary_evidence` open a block. `observe_answer`/`observe_tool` record the first collapse boundary without fabricating a block. Round ordinal advances at the same primary model-round seam used by tool-call accumulation.

```python
@dataclass(frozen=True, slots=True)
class ThinkingCaptureUpdate:
    envelope: ThinkingEnvelope = field(repr=False)
    changed_block_id: str | None = None
    collapse_boundary_reached: bool = False
    terminal: bool = False

class ThinkingCapture:
    def observe(self, item: ProviderStreamItem) -> ThinkingCaptureUpdate:
        if isinstance(item, ProviderThinkingDelta):
            return self._append_displayable(item)
        if isinstance(item, ProviderProprietaryThinkingEvidence):
            return self._record_proprietary(item)
        if isinstance(item, ProviderToolCalls):
            return self._mark_boundary("tool")
        return self._mark_boundary("answer") if item else self.snapshot()
```

Stable block IDs are generated internally from trusted assistant owner + round ordinal + monotonic sequence, not provider/import text.

- [ ] **Step 4: Thread direct Console streams.** In both controller streaming loops, route typed items before string concatenation. Update the existing assistant row/store state in place for each capture update; persist only at terminal settlement. Pass the adapter-owned `may_emit_thinking` fact to the foundation backend preflight before invoking `stream_chat`.

- [ ] **Step 5: Thread agent-provider streams.** In `console_agent_bridge.py`, process the same typed items inside `_make_call_model`/gateway consumption. Associate each block with the primary model round and preserve tool ordering. The persisted final Assistant owner receives the selected generation's envelope; session-only TOOL markers remain separate.

- [ ] **Step 6: Settle terminal states.** Normal finish maps live blocks to complete. User cancellation maps open/partial blocks to stopped. Protocol/capture/provider failure maps them to failed. No event still yields NULL/no envelope rather than an empty fabricated record. Call the foundation terminal projection once.

- [ ] **Step 7: Run direct/agent tests.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_thinking_capture.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_agent_bridge.py -k "thinking or stream or stop or provider" -q`

Expected: PASS.

- [ ] **Step 8: Commit accumulation.**

```bash
git add tldw_chatbook/Chat/console_thinking_capture.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_thinking_capture.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat: accumulate model thinking by Console round"
```

---

### Task 4: Resolve optional replay and mandatory continuation in one owner projection

**Files:**
- Create: `tldw_chatbook/Chat/console_thinking_history.py`
- Modify: `tldw_chatbook/Chat/console_history_budget.py`
- Modify: `tldw_chatbook/Chat/console_prepared_request.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Create: `Tests/Chat/test_console_thinking_history.py`
- Modify: `Tests/Chat/test_console_history_budget.py`
- Modify: `Tests/Chat/test_console_prepared_request.py`
- Modify: `Tests/Chat/test_provider_continuation_history.py`

**Interfaces consumed:** conversation policy, supported thinking envelopes, provider resolution, continuation sidecars, prepared-request pipeline.

**Interfaces produced:** `ProviderThinkingSidecar`, owner-atomic resolved history, target serializers, one counted/dispatched artifact.

- [ ] **Step 1: Write failing policy-resolution tests.** Cover legacy/NULL Auto, Include, Exclude, effective Required, saved optional preference surviving Required, target-compatible/incompatible blocks, complete/stopped/failed/proprietary/opaque blocks, and strict Include serialization failure before provider spy invocation.

```python
@dataclass(frozen=True, slots=True)
class ProviderThinkingSidecar:
    owner_message_id: str
    envelope: ThinkingEnvelope = field(repr=False)

@dataclass(frozen=True, slots=True)
class ResolvedThinkingBlock:
    owner_message_id: str
    source_format: str
    text: str = field(repr=False)
```

- [ ] **Step 2: Run the new history tests and confirm missing resolver failure.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_thinking_history.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement target compatibility and exact serializers.** Target resolution, not a model-name substring, chooses a serializer. Initial serializers are:

  - llama.cpp/vLLM same compatible local chat mode + `start_anchored_think`: reconstruct the exact start-anchored protocol representation around the stored displayable text and visible assistant answer;
  - provider-native explicit displayable field: populate only that adapter's documented field when an adapter opts in later;
  - Moonshot/Z.ai private continuation: no optional displayable serializer, because ADR-063 already owns mandatory private replay.

The local serializer may place `<think>...</think>` in its provider-wire assistant content because that is the retained exact source encoding; semantic `message.content` remains the answer only and generic providers never receive this representation.

- [ ] **Step 4: Merge thinking and continuation into conversation units.** Extend `ConsoleConversationUnit` with `thinking_groups` and retain `continuation_groups`. Unit construction attaches both by exact assistant owner ID. Trimming drops a complete unit with its visible messages, thinking, and continuation.

```python
@dataclass(frozen=True, slots=True)
class ConsoleConversationUnit:
    messages: tuple[FrozenMessage, ...] = field(repr=False)
    thinking_groups: tuple[ThinkingOwnerGroup, ...] = field(default=(), repr=False)
    continuation_groups: tuple[ContinuationOwnerGroup, ...] = field(
        default=(), repr=False
    )
```

- [ ] **Step 5: Freeze one provider wire payload.** `prepare_chat_request` resolves policy and serializes thinking before token counting. `PreparedProviderRequest` stores the final messages/tools and owner facts. Accounting reads only this artifact; `stream_chat` dispatches the same artifact without another history transform.

- [ ] **Step 6: Add duplicate/count/eviction tests.** Use canaries to assert each displayable block appears exactly once in compatible wire data and token input, private continuation remains exactly once, proprietary application copy is absent, and removing the oldest unit removes all three owner components. Assert the counted frozen payload equals the provider spy payload.

- [ ] **Step 7: Add controller and agent sidecar construction, plus auxiliary-stream discard.** Build thinking sidecars only from active-path assistant owners with supported envelopes. Read policy from the session conversation. Both direct and agent preparation pass the same sidecars. Title, summary, compaction, and other auxiliary gateway streams intentionally pass no sidecars and explicitly consume/discard any typed thinking event their adapters emit; tests prove those events create no conversation block, title text, or summary text.

- [ ] **Step 8: Run prepared-request and history suites.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_thinking_history.py Tests/Chat/test_console_history_budget.py Tests/Chat/test_console_prepared_request.py Tests/Chat/test_provider_continuation_history.py Tests/Chat/test_console_provider_gateway.py -q`

Expected: PASS.

- [ ] **Step 9: Run provider/history static checks.**

```bash
.venv/bin/python -m ruff format --check tldw_chatbook/Chat/llamacpp_think_filter.py tldw_chatbook/Chat/console_thinking_capture.py tldw_chatbook/Chat/console_thinking_history.py tldw_chatbook/Chat/console_history_budget.py tldw_chatbook/Chat/console_prepared_request.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/zai.py Tests/Chat/test_llamacpp_think_splitter.py Tests/Chat/test_console_thinking_capture.py Tests/Chat/test_console_thinking_history.py
.venv/bin/python -m ruff check tldw_chatbook/Chat/llamacpp_think_filter.py tldw_chatbook/Chat/console_thinking_capture.py tldw_chatbook/Chat/console_thinking_history.py tldw_chatbook/Chat/console_history_budget.py tldw_chatbook/Chat/console_prepared_request.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/moonshot.py tldw_chatbook/LLM_Calls/zai.py Tests/Chat/test_llamacpp_think_splitter.py Tests/Chat/test_console_thinking_capture.py Tests/Chat/test_console_thinking_history.py
git diff --check
```

- [ ] **Step 10: Commit replay/accounting and close TASK-18932.2.**

```bash
git add tldw_chatbook/Chat/console_thinking_history.py tldw_chatbook/Chat/console_history_budget.py tldw_chatbook/Chat/console_prepared_request.py tldw_chatbook/Chat/console_provider_gateway.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_thinking_history.py Tests/Chat/test_console_history_budget.py Tests/Chat/test_console_prepared_request.py Tests/Chat/test_provider_continuation_history.py
git commit -m "feat: replay compatible model thinking exactly once"
```

Update TASK-18932.2 ACs, add Implementation Notes with the explicit adapter disposition table and exact tests, and set it `Done` only after all checks pass.
