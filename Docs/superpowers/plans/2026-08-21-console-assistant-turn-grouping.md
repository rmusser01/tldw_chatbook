# Console Assistant Turn Grouping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render each Console agent response as one Assistant-owned turn whose collapsed reasoning/tool activity precedes the visible final answer inside the same surface.

**Architecture:** Preserve the persisted message tree and display-only TOOL marker contract. Add structured, session-only activity presentation at the bridge seam; derive Assistant turn ownership from contiguous transcript messages; and render each derived turn through a focused composite widget while the transcript retains reconciliation, selection, expansion, pruning, and export ownership. Live and resumed runs share status and Thinking helpers so their activity order cannot drift.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich, dataclasses/`Literal`, pytest/pytest-asyncio, generated TCSS.

**Spec:** `Docs/superpowers/specs/2026-08-21-console-assistant-turn-grouping-design.md`

**Task:** `backlog/tasks/task-19426 - Group-Console-tool-activity-inside-assistant-turns.md`

**ADR required:** no
**ADR path:** N/A
**Reason:** This is presentation-only. The conversation tree, display-only marker ownership, provider/runtime contracts, run log, and database schema stay unchanged. ADR-031 still governs keybinding/footer-hint truthfulness.

---

## File map and invariants

- Create `tldw_chatbook/Chat/console_turn_grouping.py`: pure, Textual-free grouping and visual-order helpers.
- Create `tldw_chatbook/Widgets/Console/console_assistant_turn.py`: Assistant container and activity disclosure rendering/events.
- Modify `tldw_chatbook/Chat/console_chat_models.py`, `console_agent_bridge.py`, and `console_chat_store.py`: structured session-only activity metadata, safe Thinking derivation, live/resume parity.
- Modify `tldw_chatbook/Widgets/Console/console_transcript.py`: composite row planning/reconciliation, selection, expansion, pruning, windowing, export.
- Modify `_agentic_terminal.tcss` and regenerate `tldw_cli_modular.tcss`.
- Add focused Chat/UI tests and extend the existing bridge, transcript, disclosure, pruning, and CSS suites.

Preserve these invariants in every task:

- TOOL markers remain non-persisted, non-parent display messages anchored to the Assistant node.
- Store order remains `USER -> Assistant placeholder -> TOOL`; only rendering/navigation project activities before the answer.
- Structured headers never parse rendered marker content.
- Thinking never exposes provider-private reasoning or hidden chain-of-thought.
- Expansion remains view-only, per-marker, collapsed initially, and keyed by the original TOOL id.
- Existing ids remain the Inspector/action/selection identities.
- No new screen binding or footer hint is added.
- Streaming sync and activity-stack changes both preserve the Assistant container and answer widget; only the changed turn's nested activity stack may rebuild.

---

### Task 1: Structured activity presentation and status classification

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py:536-636`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:2150-2285`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:500-930,3747-3845,5488-5630,5818-5850`
- Create: `Tests/Chat/test_console_activity_presentation.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py:2631-3065`

- [ ] **Step 1: Write failing model/store tests.** Cover valid metadata, invalid/empty/multiline/>200-character labels, pass-through on a display-only TOOL marker, and absence from persistence/restore.

```python
def test_activity_presentation_is_session_only(store, session_id):
    presentation = ConsoleActivityPresentation("tool", "fs_list", "success")
    marker = store.append_message(
        session_id,
        role=ConsoleMessageRole.TOOL,
        content="⚙ fs_list → src/",
        activity_presentation=presentation,
    )
    assert marker.activity_presentation == presentation
    assert marker.persisted_message_id is None
```

- [ ] **Step 2: Run the new test and confirm failure.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py -q`
Expected: FAIL because the model/field do not exist.

- [ ] **Step 3: Add the minimal bounded contract and store keyword.**

```python
ConsoleActivityKind = Literal[
    "thinking", "tool", "spawn", "tasks", "changes", "feedback", "warning", "activity"
]
ConsoleActivityStatus = Literal["success", "blocked", "failed", "done"]

@dataclass(frozen=True)
class ConsoleActivityPresentation:
    kind: ConsoleActivityKind
    label: str
    status: ConsoleActivityStatus

    def __post_init__(self) -> None:
        if not self.label or len(self.label) > 200 or "\n" in self.label or "\r" in self.label:
            raise ValueError("activity label must be a non-empty single line <= 200 chars")
```

Add `activity_presentation: ConsoleActivityPresentation | None = None` to `ConsoleChatMessage` and `append_message`. Pass it only to the in-memory dataclass; do not add it to DB metadata, trajectory payloads, provider history, or restore serialization.

- [ ] **Step 4: Write failing classifier tests.** Parametrize successful results, unknown `ERROR:` failures, approval timeout, `STEP_ERROR`, direct controller review-hook verdicts (`USER_DENIED_REFUSAL`, `KILL_SWITCH_REFUSAL`), and runtime-enveloped canonical refusals imported from builtin/local/MCP providers (deny, timeout, kill switch, unresolved permission, root changed).

```python
@pytest.mark.parametrize("refusal", BLOCKED_PROVIDER_REFUSALS)
def test_enveloped_provider_refusals_are_blocked(refusal):
    assert classify_activity_status(STEP_TOOL_RESULT, f"ERROR: {refusal}") == "blocked"

@pytest.mark.parametrize("verdict", [USER_DENIED_REFUSAL.format(name="fs_list"), KILL_SWITCH_REFUSAL])
def test_direct_controller_verdicts_are_blocked(verdict):
    assert classify_activity_status(STEP_TOOL_RESULT, verdict) == "blocked"

def test_unknown_enveloped_error_is_failed():
    assert classify_activity_status(STEP_TOOL_RESULT, "ERROR: disk exploded") == "failed"
```

- [ ] **Step 5: Run status tests and confirm the missing helper fails.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py -k status -q`
Expected: FAIL.

- [ ] **Step 6: Implement one bridge-owned classifier, step presentation builder, and explicit non-step presentation inventory.** Match direct controller review verdicts before the success path, then unwrap `ERROR:` before matching dispatched-provider refusal constants. Use exported controller/provider constants and pinned builtin prefixes only where the tool name is appended. Attach explicit structured presentations at every known non-step TOOL-marker builder: `append_todo_marker`; live diff-feedback append; live `_append_change_markers`; and resume change summary, sub-agent-post-turn summary, concurrent-sub-agent warning, change-tracking failure, and diff-feedback block construction. Use bounded literal labels such as `Tasks updated`, `Changes`, `Sub-agent changes`, `Concurrent sub-agent`, `Change tracking`, and `Feedback delivered`; use `failed` for tracking failure and `done` for informational notices.

```python
def classify_activity_status(kind: str, result: Any = None) -> ConsoleActivityStatus:
    if kind == STEP_APPROVAL_TIMEOUT:
        return "blocked"
    if kind == STEP_ERROR:
        return "failed"
    if kind != STEP_TOOL_RESULT:
        return "done"
    text = str(result if result is not None else "")
    if _is_direct_controller_block(text):
        return "blocked"
    if not text.startswith("ERROR:"):
        return "success"
    error = text.removeprefix("ERROR:").strip()
    return "blocked" if _is_blocked_tool_refusal(error) else "failed"
```

Thread presentation through live `_append_marker` and `resume_marker_messages`; derive step labels from bounded `tool_name`/step kind, never from marker text. For non-step builders, pass their explicit presentation alongside the same formatter call that creates their body.

- [ ] **Step 7: Add live/resume presentation inventory tests.** Assert every known builder above produces the intended non-generic kind/label/status, and that live/resumed change summaries, failures, concurrent/sub-agent notices, and diff-feedback disclosures have identical presentation metadata as well as identical content. Task snapshots are live-only and should be tested as such. Keep one legacy marker-without-metadata test proving the neutral `Activity · done` fallback remains available.

- [ ] **Step 8: Run focused tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py -k "activity or marker or resume" -q`
Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat: add structured Console activity presentation"
```

---

### Task 2: Safe intermediate Thinking markers with live/resume parity

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:500-930,3747-3845,5488-5630`
- Modify: `Tests/Chat/test_console_activity_presentation.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py:2905-3065,5985-6040`

- [ ] **Step 1: Write failing sanitizer tests.** Safe visible preamble survives; `<thinking>/<analysis>/<reasoning>` shapes, explicit tool/function payloads, fenced JSON, controls, empty text, and over-cap text are rejected/flattened/truncated conservatively.

- [ ] **Step 2: Run sanitizer tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py -k thinking -q`
Expected: FAIL because the sanitizer is missing.

- [ ] **Step 3: Implement the safe-summary helper and Thinking message builder.** Reject private tags before transformation, take only visible pre-fence text, reject tool payload keys, flatten terminal controls/whitespace, apply the existing display cap, and return `None` if nothing safe remains.

```python
def safe_intermediate_thinking_summary(summary: str | None) -> str | None:
    raw = str(summary or "")
    if _PRIVATE_REASONING_TAG_RE.search(raw):
        return None
    visible = raw.split("```", 1)[0]
    if _TOOL_PAYLOAD_KEY_RE.search(visible):
        return None
    visible = _sanitize_task_marker_label(visible).strip()
    return _truncate_step_text(visible, limit=_console_tool_result_display_cap()) if visible else None
```

Use `kind="thinking"`, label `Thinking`, status `done`. With no safe summary, create no expandable detail (`tool_output_full=None`) and no dead chevron.

- [ ] **Step 4: Write failing sequence tests.** Cover `MODEL -> TOOL_CALL -> RESULT -> MODEL(final)`, `MODEL -> SPAWN`, `MODEL -> direct refused TOOL_RESULT`, multiple calls in one round, final-model-only, model-then-error, and interleaved sub-agent steps. Assert one Thinking row per tool-producing primary model round, before its first activity/result, and none for the final answer. Compare live append captures with resume blocks.

- [ ] **Step 5: Run sequence tests and confirm current omission fails.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py -k "thinking or live_resume" -q`
Expected: FAIL because `STEP_MODEL` is always omitted.

- [ ] **Step 6: Buffer only the pending primary `STEP_MODEL`.** Flush it as Thinking when the next primary step is `STEP_TOOL_CALL`, `STEP_SPAWN`, or a direct `STEP_TOOL_RESULT` (the review/continuation refusal shape); clear without output for other next primary steps. Sub-agent callbacks neither flush nor clear it. In resume, perform equivalent look-ahead with the same proving-step predicate and call the same builder.

- [ ] **Step 7: Run full bridge tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py -q`
Expected: PASS.

- [ ] **Step 8: Commit.**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat: derive safe Console thinking activity"
```

---

### Task 3: Pure Assistant-turn grouping and visual order

**Files:**
- Create: `tldw_chatbook/Chat/console_turn_grouping.py`
- Create: `Tests/Chat/test_console_turn_grouping.py`

- [ ] **Step 1: Write failing grouping tests.** Cover `USER, ASSISTANT, TOOL, TOOL, USER`; orphan TOOL; SYSTEM/new ASSISTANT closing ownership; empty Assistant body; absent off-branch marker; and visual order `USER, TOOL, TOOL, ASSISTANT` while owned ids retain causal membership.

- [ ] **Step 2: Run tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_turn_grouping.py -q`
Expected: FAIL because the module is missing.

- [ ] **Step 3: Implement immutable grouping records and one O(n) scan.**

```python
@dataclass(frozen=True)
class ConsoleAssistantTurn:
    assistant: ConsoleChatMessage
    activities: tuple[ConsoleChatMessage, ...] = ()

    @property
    def owned_message_ids(self) -> tuple[str, ...]:
        return (self.assistant.id, *(item.id for item in self.activities))
```

Use a one-of `ConsoleTranscriptUnit(message | assistant_turn)`. Only consume contiguous TOOL messages immediately after an Assistant. Add `visual_messages(units)` yielding activities before the Assistant; keep the module free of Textual imports.

- [ ] **Step 4: Run tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_turn_grouping.py -q`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add tldw_chatbook/Chat/console_turn_grouping.py Tests/Chat/test_console_turn_grouping.py
git commit -m "feat: derive Console assistant turn ownership"
```

---

### Task 4: Focused Assistant-turn and activity-disclosure widgets

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_assistant_turn.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:1120-1485,1580-1775`
- Modify if required by import convention: `tldw_chatbook/Widgets/Console/__init__.py`
- Create: `Tests/UI/test_console_assistant_turn.py`

- [ ] **Step 1: Write failing focused-widget tests.** Assert independent collapsed defaults; chevron/status/detail behavior; no chevron or toggle for detail-free Thinking; click/Enter/Space activation with original marker id; selected action row visible between header and hidden detail; and container order `Assistant header -> activities -> answer -> Assistant adjuncts`.

```python
async def test_disclosures_are_independently_collapsed_by_default():
    async with DisclosureHarness(tool_a, tool_b).run_test() as pilot:
        rows = list(pilot.app.query(ConsoleActivityDisclosure))
        assert [row.expanded for row in rows] == [False, False]
        await pilot.click("#console-activity-header-t1")
        assert [row.expanded for row in rows] == [True, False]
```

- [ ] **Step 2: Run focused tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_assistant_turn.py -q`
Expected: FAIL because the widget module is missing.

- [ ] **Step 3: Implement a focusable activity header and activation event.** Use a focusable `Static`, not a disabled Button, so a detail-free Thinking row stays selectable. Handle only click and Enter/Space; do not add a screen binding.

```python
class ConsoleActivityActivated(Message):
    def __init__(self, message_id: str, *, toggle_requested: bool) -> None:
        super().__init__()
        self.message_id = message_id
        self.toggle_requested = toggle_requested

class ConsoleActivityHeader(Static):
    can_focus = True

    def _activate(self) -> None:
        self.post_message(ConsoleActivityActivated(
            self.message_id, toggle_requested=self.has_detail
        ))
```

- [ ] **Step 4: Implement disclosure/container composition.** `ConsoleActivityDisclosure` receives prebuilt header/action/detail/diff widgets and a boolean from transcript-owned expansion state. It never owns a second expansion set. `ConsoleAssistantTurnWidget` receives the Assistant header, activity widgets, headerless answer widget, and Assistant adjunct widgets. Use `console-assistant-turn-<assistant-id>` for the container and retain original message ids for all nested operations.

- [ ] **Step 5: Add `show_header: bool = True` to existing plain/Markdown message widgets.** Standalone output remains unchanged. When false, omit `ConsoleMessageHeader`; refactor `sync_message` to treat the header query as optional while always continuing body/footer streaming sync.

```python
header = self.query_one(ConsoleMessageHeader) if self._show_header else None
if header is not None:
    header.sync_header(message, presentation, speech_state)
# body/footer sync always continues
```

- [ ] **Step 6: Run focused widget and existing row tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_assistant_turn.py Tests/UI/test_console_native_transcript.py -k "markdown or message_widget or assistant_turn or disclosure" -q`
Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add tldw_chatbook/Widgets/Console/console_assistant_turn.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/__init__.py Tests/UI/test_console_assistant_turn.py Tests/UI/test_console_native_transcript.py
git commit -m "feat: add Console assistant turn widgets"
```

---

### Task 5: Transcript integration, reconciliation, selection, and action parity

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:833-860,2415-2605,3543-3715,4222-4388,5369-6190`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/Chat/test_tool_output_disclosure.py`

- [ ] **Step 1: Write failing hierarchy tests.** Mount `USER, ASSISTANT, TOOL, TOOL` and assert one top-level Assistant container, both activities nested before the answer, no top-level owned TOOL row, and orphan TOOL remaining standalone. Assert Assistant selection styles only the answer; activity selection styles its header and exposes actions while still collapsed.

- [ ] **Step 2: Write failing identity tests.** Capture turn/answer/activity-stack widget objects; stream answer deltas and terminal status changes; then append an activity. Both streaming and marker-set changes must preserve the container and answer objects; marker-set change may replace only the nested activity stack. Unrelated turns retain identity. Session switch clears expansion.

- [ ] **Step 3: Run integration tests and confirm detached layout failure.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py -k "assistant_turn or owned_activity or activity_reconcile" -q`
Expected: FAIL because Assistant/TOOL rows are siblings.

- [ ] **Step 4: Add an `assistant-turn` transcript row/spec.** Iterate grouped visible units. Standalone messages keep the current path; Assistant units produce one rule plus one composite row keyed by Assistant id. Its signature covers the Assistant signature, ordered activity signatures/ids, selection, owned expanded ids, and Assistant adjuncts—but no unrelated global state.

```python
(
    "assistant-turn",
    assistant_signature,
    tuple(activity_signatures),
    tuple(activity.id for activity in turn.activities),
    selected_message_id,
    tuple(sorted(expanded_ids & set(turn.owned_message_ids))),
    assistant_adjunct_signatures,
)
```

- [ ] **Step 5: Reuse existing transcript services for nested children.** Extract small helpers that build the current header, answer, actions/guide, citations, annotations, original attempt, image/generation/video, and diff widgets. Do not duplicate action/media logic. Structured metadata supplies activity header/status; legacy/unknown metadata uses neutral `Activity · done`. `content`, expanded `tool_output_full`, and `tool_diff` remain detail payloads. No extra detail means no chevron.

- [ ] **Step 6: Reconcile the nested activity stack without remounting the answer.** Always keep the Assistant container/header/answer widget mounted and sync current presentation, speech, body, and status in place. When ordered activity ids change, reconcile or replace only `.console-assistant-activity-stack` from transcript-owned expansion/selection state. Stable top-level and answer ids preserve streaming buffers and text selection; unrelated rows remain mounted.

- [ ] **Step 7: Route every disclosure control through existing seams.** Handle `ConsoleActivityActivated` by selecting the original message, then calling `toggle_tool_output` only when requested. Refine `toggle_tool_output` to ignore non-expandable messages. Existing Full-output button and `o` already call this seam; collapsing must not clear selection.

```python
@on(ConsoleActivityActivated)
def _on_activity_activated(self, event: ConsoleActivityActivated) -> None:
    self.select_message(event.message_id)
    if event.toggle_requested:
        self.toggle_tool_output(event.message_id)
```

- [ ] **Step 8: Run integration/action tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_assistant_turn.py Tests/Chat/test_tool_output_disclosure.py -q`
Expected: PASS.

- [ ] **Step 9: Commit.**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_native_transcript.py Tests/Chat/test_tool_output_disclosure.py
git commit -m "feat: render activity inside Console assistant turns"
```

---

### Task 6: Turn-aware navigation, windowing, pruning, and plain export

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:2878-2955,3916-4340,4282-4320,5315-5395`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/UI/test_console_transcript_pruning.py`
- Modify: `Tests/UI/test_console_transcript_selection_prune_bound.py`

- [ ] **Step 1: Write failing navigation/export tests.** Assert `j/k` order `USER -> Thinking -> tool -> Assistant answer -> next USER`; nested id remains Inspector selection; and plain export is expansion-independent, ordered as Assistant heading/activity headers with bounded current previews/final answer, with no `tool_output_full` tail or diff body.

- [ ] **Step 2: Write failing pruning/window tests.** Assert pruning adds Assistant plus all owned activity ids atomically; selected nested activity and streaming Assistant protect the whole unit; no orphan fragment remains; initial/far-jump windows do not split a unit; revealing an activity reveals its owner.

- [ ] **Step 3: Run tests and confirm direct-child assumptions fail.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_pruning.py Tests/UI/test_console_transcript_selection_prune_bound.py -k "turn or activity or visual_order or plain" -q`
Expected: FAIL.

- [ ] **Step 4: Use derived visual messages for navigation.** Filter pruned/hidden ids first, group units, then flatten via `visual_messages`. Update reveal/recenter/turn-alignment helpers to map nested marker id to its owner and align to the unit's first causal message.

- [ ] **Step 5: Prune top-level units atomically.** Map a composite row key to `owned_message_ids`; protect if Assistant streams or any owned id is selected. Commit all ids from a candidate unit together while retaining existing actual-height scroll compensation. Do not delete store messages.

- [ ] **Step 6: Export from grouping, never expansion state.** For each Assistant unit emit heading, structured activity headers plus existing bounded `content` previews, then answer/status/actions. Never read `tool_output_full`, `tool_diff`, or `_expanded_tool_output_ids`.

- [ ] **Step 7: Run full navigation/pruning suites.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_pruning.py Tests/UI/test_console_transcript_selection_prune_bound.py -q`
Expected: PASS.

- [ ] **Step 8: Commit.**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_pruning.py Tests/UI/test_console_transcript_selection_prune_bound.py
git commit -m "fix: keep Console turn navigation and pruning atomic"
```

---

### Task 7: Shared Assistant surface and nested-disclosure styling

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:4259-4620`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_console_agent_tool_row_css.py`
- Modify: `Tests/UI/test_console_assistant_turn.py`

- [ ] **Step 1: Write failing stylesheet/geometry tests.** Require authored and bundled selectors for the Assistant surface/ownership accent, activity stack/disclosure, focus/selection, text-plus-color terminal statuses, expanded detail/diff, and narrow overflow. Under the real bundle at representative wide/narrow Console sizes, assert headers/details/answer have nonzero geometry and no horizontal overflow or clipped status.

- [ ] **Step 2: Run CSS tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_agent_tool_row_css.py Tests/UI/test_console_assistant_turn.py -k "stylesheet or geometry or narrow" -q`
Expected: FAIL because selectors are absent.

- [ ] **Step 3: Add token-based TCSS.** Use existing `$ds-*` panel/focus/status tokens. Give the Assistant one auto-height shared surface and one ownership accent; keep disclosures visually quiet. Make the activity label flexible with ellipsis and the text status slot fixed/readable. Selected/focused states must work in dark/light themes and not rely on color alone.

- [ ] **Step 4: Rebuild the production CSS bundle.**

Run: `./build_css.sh`
Expected: bundle regenerates without errors.

- [ ] **Step 5: Run focused CSS/widget tests.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_agent_tool_row_css.py Tests/UI/test_console_assistant_turn.py -q`
Expected: PASS.

- [ ] **Step 6: Commit source, bundle, and tests.**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_agent_tool_row_css.py Tests/UI/test_console_assistant_turn.py
git commit -m "style: group Console activity within assistant turns"
```

---

### Task 8: Regression, live verification, review, and Backlog completion

**Files:**
- Modify: `backlog/tasks/task-19426 - Group-Console-tool-activity-inside-assistant-turns.md`
- Modify only if a real reusable incident warrants it: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`
- Review: every implementation file changed above

- [ ] **Step 1: Run the complete focused feature set.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_turn_grouping.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_tool_output_disclosure.py Tests/UI/test_console_assistant_turn.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_pruning.py Tests/UI/test_console_transcript_selection_prune_bound.py Tests/UI/test_console_agent_tool_row_css.py -q
```

Expected: PASS.

- [ ] **Step 2: Run adjacent Console integration coverage.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_agent_controller.py Tests/integration/test_console_agent_marker_anchoring_e2e.py -q
```

Expected: PASS. Record exact optional skips; do not treat an unexecuted integration as evidence.

- [ ] **Step 3: Run lint, format, static, and repository checks.**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Chat/console_turn_grouping.py tldw_chatbook/Widgets/Console/console_assistant_turn.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Widgets/Console/console_transcript.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Chat/console_turn_grouping.py tldw_chatbook/Widgets/Console/console_assistant_turn.py tldw_chatbook/Widgets/Console/console_transcript.py Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_turn_grouping.py Tests/Chat/test_console_agent_bridge.py Tests/Chat/test_tool_output_disclosure.py Tests/UI/test_console_assistant_turn.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_pruning.py Tests/UI/test_console_transcript_selection_prune_bound.py Tests/UI/test_console_agent_tool_row_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check tldw_chatbook/Chat/console_turn_grouping.py tldw_chatbook/Widgets/Console/console_assistant_turn.py Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_turn_grouping.py Tests/UI/test_console_assistant_turn.py
git diff --check
```

Expected: all commands exit 0. The format gate intentionally covers every new Python file; clean `origin/dev` already reports nine existing modified-target files as needing whole-file Ruff formatting, so broad formatting would create unrelated churn. `ruff check` still covers every changed Python file and `git diff --check` guards the complete patch.

- [ ] **Step 4: Run the full suite.**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q`
Expected: PASS, or reproduce/document any pre-existing failure against clean `origin/dev` before claiming completion.

- [ ] **Step 5: Perform real Console verification with an isolated scratch profile.** Read `backlog/docs/lessons-live-verification.md` first. At supported wide and narrow terminal sizes:

1. Send a query causing a real local `fs_*` call.
2. Confirm one Assistant surface contains collapsed Thinking/tool rows, then the visible final answer.
3. Expand/collapse by mouse and keyboard; confirm selected-tool `o` uses the same state.
4. Confirm collapsed selected activity actions and Inspector attribution remain usable.
5. Exercise a blocked/failed call and inspect status/detail.
6. Observe a streaming answer without unrelated remounts/scroll jumps.
7. Resume the conversation; verify identical activity order and collapsed defaults.
8. Capture wide/narrow screenshots in a temporary/task evidence path; do not add them unless repository convention requires it.

- [ ] **Step 6: Self-review the diff against the spec.** Search specifically for marker-string parsing; accidental metadata persistence; raw private reasoning; orphan/cross-branch attribution; duplicate widget ids; nested text-selection breakage; divergent expansion paths; remaining direct-child pruning assumptions; full-output/diff leaks in export; source/bundle CSS drift; and unrelated files from the original dirty checkout.

- [ ] **Step 7: Request code review.** Use `superpowers:requesting-code-review`, address substantive findings, and rerun affected tests plus `git diff --check`.

- [ ] **Step 8: Complete Backlog hygiene only after evidence exists.** Add concise Implementation Notes (approach, changed files, tradeoffs, exact test/live results, ADR decision, deviations) using `backlog task edit 19426 --notes`. Check all ACs only with evidence. Add a lesson only for a concrete reusable incident. Then set Done via Backlog CLI only after the full Definition of Done:

```bash
backlog task edit 19426 -s Done
```

- [ ] **Step 9: Commit final task documentation.**

```bash
git add 'backlog/tasks/task-19426 - Group-Console-tool-activity-inside-assistant-turns.md'
git commit -m "docs: complete Console assistant turn task"
```

If a lessons file genuinely changed, stage it explicitly before the commit; otherwise leave both lesson files untouched.
