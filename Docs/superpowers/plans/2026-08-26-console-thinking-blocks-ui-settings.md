# Console Thinking Disclosure and Settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render actual model thinking as honest, keyboard-accessible Assistant-turn disclosures with expanded-live to one-time auto-collapsed behavior, plus default-on device visibility and conversation-owned replay controls.

**Architecture:** Reuse `ConsoleActivityDisclosure` and the existing transcript expansion set rather than add a second disclosure system. A pure presentation projection maps supported generation blocks to trusted internal activity identities and owner references; transcript-owned state handles lazy detail, selection, and auto-collapse. Canonical F9 Settings owns the device toggle/default policy, while the existing Console Context & memory modal edits the current conversation policy. Existing safe model-step markers are renamed Planning and suppressed per round when real Thinking exists.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich `Content`, existing Settings immediate-toggle and persistence seams, generated TCSS, pytest/Pilot.

**Spec:** `Docs/superpowers/specs/2026-08-26-console-thinking-blocks-design.md`

**Task:** `backlog/tasks/task-18932.3 - Render-collapsible-Console-thinking-and-settings.md`

## Global Constraints

- TASK-18932.1 and TASK-18932.2 must be complete.
- Impeccable mode is **Operate**: dense, calm, precise, keyboard-first. Reuse the incumbent Assistant-turn header/detail grammar and semantic tokens; add no decorative border/color/motion vocabulary.
- Ponytail full mode: extend `ConsoleActivityDisclosure`, `_expanded_tool_output_ids`, and existing Settings/Context surfaces. Add no dependency, animation framework, parallel disclosure widget, or new Console controller.
- `Show model thinking` controls rendering only. Never gate capture, persistence, provider continuation, policy resolution, token accounting, or export on it.
- `Thinking · unavailable` appears only for a stored/live proprietary evidence block. Its body is exactly `Proprietary thinking obfuscated - not available`.
- Historical rows start collapsed. Live rows open only on the first evidence event and auto-collapse at most once. Manual action removes the pending automatic transition.
- If the answer/tool boundary preceded terminal-only evidence, mount that evidence collapsed. Do not flash it expanded for one paint.
- Collapsed historical disclosures do not mount full thinking text. Expansion, copy, and Inspector resolve from the envelope sidecar on demand.
- Stable imported block IDs may participate in a namespace hash but never appear raw in Textual DOM IDs, selection IDs, or CSS selectors.
- Preserve the Assistant turn widget, answer widget, scroll anchor, selection, focus, and tool expansion during live thinking deltas and toggle refreshes.
- No new global binding or footer hint. Mouse, Enter, Space, and existing `o` semantics remain the whole interaction surface.
- Immediately before implementation UI edits, read Impeccable `reference/craft-floor.md`. Perform one batched visual inspection across supported narrow/normal/wide terminal sizes, one batched fix, and at most one confirmation pass.

---

### Task 1: Project generation blocks into trusted Assistant activities

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Modify: `tldw_chatbook/Chat/console_turn_grouping.py`
- Create: `Tests/Chat/test_console_thinking_presentation.py`
- Modify: `Tests/Chat/test_console_turn_grouping.py`

**Interfaces consumed:** supported `ThinkingEnvelope`, assistant/session IDs, capture boundary/lifecycle facts from TASK-18932.2.

**Interfaces produced:** `ConsoleThinkingActivityRef`, deterministic internal IDs, ordered activity presentations and statuses.

- [ ] **Step 1: Write failing pure presentation tests.** Cover displayable/proprietary/no-evidence, multiple rounds interleaved with TOOL activities, status mapping, exact unavailable label/body constant, duplicate block IDs, imported hostile IDs, and same block ID in different sessions/assistant owners.

```python
PROPRIETARY_THINKING_NOTICE = "Proprietary thinking obfuscated - not available"

@dataclass(frozen=True, slots=True)
class ConsoleThinkingActivityRef:
    activity_id: str
    assistant_message_id: str
    block_id: str
    label: str
    status: ConsoleActivityStatus
```

- [ ] **Step 2: Run the presentation test and confirm missing projection failure.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_thinking_presentation.py -q`

Expected: FAIL.

- [ ] **Step 3: Expand the status vocabulary and pure mapping.**

```python
ConsoleActivityStatus = Literal[
    "success", "blocked", "failed", "done", "live", "stopped", "unavailable"
]

def thinking_activity_id(
    *, session_id: str, assistant_message_id: str, block_id: str
) -> str:
    identity = f"{session_id}\0{assistant_message_id}\0{block_id}"
    return f"thinking-{uuid5(NAMESPACE_URL, identity).hex}"
```

Map displayable status `complete -> done`, `stopped -> stopped`, `failed -> failed`; a live capture supplied by the session state maps to `live`. Proprietary always maps to `unavailable`. Label is `Thinking` for displayable and `Thinking` plus separate unavailable status for proprietary; never bake the status into the label.

- [ ] **Step 4: Merge model thinking into Assistant activity order.** Reuse each block's model-round ordinal and existing TOOL marker sequence facts. A model block precedes the first tool activity belonging to that round. No envelope means no synthetic activity. Planning marker suppression is completed in Task 4.

- [ ] **Step 5: Run pure grouping/presentation tests.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_thinking_presentation.py Tests/Chat/test_console_turn_grouping.py -q`

Expected: PASS.

- [ ] **Step 6: Commit the presentation model.**

```bash
git add tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_turn_grouping.py Tests/Chat/test_console_thinking_presentation.py Tests/Chat/test_console_turn_grouping.py
git commit -m "feat: project model thinking into Assistant activities"
```

---

### Task 2: Reuse disclosures for lazy live thinking and manual-win collapse

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_assistant_turn.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Widgets/Console/__init__.py` only for an exported presentation type if a current import needs it
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_console_assistant_turn.py`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/UI/test_console_transcript_pruning.py`
- Modify: `Tests/UI/test_console_transcript_selection_contract.py`
- Modify: `Tests/UI/test_console_transcript_windowing.py`
- Create: `Tests/UI/test_console_thinking_disclosures.py`

**Interfaces consumed:** Task 1 activity refs and block sidecars; existing disclosure activation/selection/expansion APIs.

**Interfaces produced:** lazy thinking detail, in-place stream updates, one-time auto-collapse state, owner mapping for selection/copy/Inspector.

- [ ] **Step 1: Write failing disclosure lifecycle tests.** Use a mounted `ConsoleAssistantTurnWidget` to prove first delta expanded; later delta updates the same disclosure instance; first answer/tool boundary collapses; terminal fallback collapses; manual expand/collapse removes pending auto action; historical starts collapsed; terminal-only late proprietary starts collapsed; no evidence mounts nothing.

```python
async def test_manual_toggle_wins_over_pending_auto_collapse(app, pilot):
    disclosure = app.query_one(ConsoleActivityDisclosure)
    await pilot.click(f"#{disclosure.header.id}")
    assert disclosure.expanded is False
    app.transcript.observe_thinking_boundary(disclosure.activity_message_id)
    assert disclosure.expanded is False
```

- [ ] **Step 2: Run the new UI suite and confirm missing model activities fail.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_console_thinking_disclosures.py Tests/UI/test_console_assistant_turn.py -q`

Expected: FAIL.

- [ ] **Step 3: Make disclosure detail availability independent of mounted detail.** Keep the existing widget. Add `detail_available` and one async child-replacement method so a collapsed block can show a chevron without carrying full text.

```python
async def replace_detail_widgets(self, detail_widgets: Iterable[Widget]) -> None:
    replacements = tuple(detail_widgets)
    if self.detail_stack.children:
        await self.detail_stack.remove_children()
    if replacements:
        await self.detail_stack.mount(*replacements)
    self._has_detail = self.detail_available
    self.header.sync_header(
        self.label,
        self.status,
        expanded=self.expanded,
        expandable=self.detail_available,
        selected=self.selected,
    )
```

Collapsed thinking passes `detail_available=True` and no full child. Expansion resolves a literal terminal-safe `Static`/existing message-body projection, mounts it, then reveals. Collapse removes the body after state/paint reconciliation. Tool disclosures retain current eager behavior unless they already use the same helper safely.

- [ ] **Step 4: Generalize transcript expansion ownership minimally.** Keep `_expanded_tool_output_ids` for compatibility with existing tests/callers but allow it to contain trusted model activity IDs. Add:

```python
self._thinking_activity_refs: dict[str, ConsoleThinkingActivityRef] = {}
self._pending_thinking_auto_collapse: set[str] = set()
self._manual_thinking_disclosures: set[str] = set()
```

On first live evidence before a boundary, add ID to expanded + pending. On manual `ConsoleActivityActivated`, add ID to manual and discard pending before toggling. On boundary, collapse only IDs still pending. At terminal fallback, do the same. Pruning/windowing removes mounted rows but preserves valid session expansion state; session switch prunes refs not owned by that session.

- [ ] **Step 5: Build model disclosure rows inside the existing Assistant stack.** Resolve full bodies from the owning message's envelope only for expansion/copy/Inspector. Use the hashed activity ID for widgets and explicit map back to assistant/block owners for actions. Proprietary detail always comes from `PROPRIETARY_THINKING_NOTICE`, never storage.

- [ ] **Step 6: Update live rows in place.** Extend the existing Assistant-turn signature/reconciliation so text/status change calls `sync_activity` and replaces only the lazy detail when expanded. Do not replace `ConsoleAssistantTurnWidget` or its answer widget. Add object-identity, scroll-offset, and selected-answer tests.

- [ ] **Step 7: Add keyboard, selection, pruning, and copy/Inspector tests.** Enter/Space header and existing `o` toggle one disclosure. Navigation includes visible thinking and skips hidden rows. Hiding/removing selected thinking clears its selection only. Copy/Inspector returns full displayable text even while collapsed; answer-copy and speech remain answer-only. Pruning removes/unmounts the whole Assistant turn without orphaned model rows.

- [ ] **Step 8: Style statuses with incumbent semantic tokens.** Extend only existing status selectors. `live` uses active/focus semantics, `done` current neutral/success behavior, `stopped` muted warning, `failed` error, `unavailable` muted/warning with readable text. Keep header dimensions stable and detail literal/wrapped/dim. No new borders or animation.

- [ ] **Step 9: Regenerate and run disclosure suites.**

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_console_thinking_disclosures.py Tests/UI/test_console_assistant_turn.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_pruning.py Tests/UI/test_console_transcript_selection_contract.py Tests/UI/test_console_transcript_windowing.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit disclosures.**

```bash
git add tldw_chatbook/Widgets/Console/console_assistant_turn.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/__init__.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_thinking_disclosures.py Tests/UI/test_console_assistant_turn.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_pruning.py Tests/UI/test_console_transcript_selection_contract.py Tests/UI/test_console_transcript_windowing.py
git commit -m "feat: render collapsible Console thinking disclosures"
```

---

### Task 3: Add default-on device visibility and conversation replay controls

**Files:**
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/settings_context_memory.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_adapter.py`
- Modify: `tldw_chatbook/Widgets/Console/console_context_controls.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/Widgets/Console/console_edit_message_modal.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py` only if the session controller needs an existing refresh callback exposed
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` only for the existing bounded message delegate/result application; do not add policy logic
- Modify: `Tests/UI/test_settings_context_memory_controls.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_console_context_controls.py`
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/Chat/test_console_context_policy_lifecycle.py`
- Modify: `Tests/Chat/test_console_edit_message_modal.py`

**Interfaces consumed:** foundation conversation policy getter/setter and provider-history effective Required resolution.

**Interfaces produced:** `[console].show_model_thinking`, `[console].thinking_history_policy_default`, immediate transcript refresh, current-conversation policy editing.

- [ ] **Step 1: Write failing config/default tests.** Missing `show_model_thinking` resolves True; explicit false resolves False; invalid hand-edited value resolves True with safe fallback. Missing/invalid default history policy resolves `auto`; valid Auto/Include/Exclude round-trips. Neither appears in a conversation export/sync payload merely because it is a device default.

- [ ] **Step 2: Run settings model tests and confirm missing keys fail.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_settings_context_memory_controls.py Tests/UI/test_settings_configuration_hub.py -k thinking -q`

Expected: FAIL.

- [ ] **Step 3: Add canonical config defaults and coercion.** Under `[console]` in `CONFIG_TOML_CONTENT` add:

```toml
show_model_thinking = true  # Presentation only; capture and replay are unchanged
thinking_history_policy_default = "auto"  # auto, include, exclude for new conversations
```

Use existing bool coercion. Use `normalize_thinking_history_policy` for the default enum. Do not add these to `[chat_defaults]`, provider defaults, conversation exchange, or Sync v2.

- [ ] **Step 4: Add `Show model thinking` to canonical F9 Console Behavior.** Use one Checkbox with explicit On/Off state wording in its label/help and the existing immediate-toggle pattern already used by Settings presentation controls. Add field guidance, search indexing, loaded/current/reset mappings, and tests. Its event handler persists `{"console": {"show_model_thinking": value}}` off the event loop and updates the live in-memory setting without waiting for the category Save button.

- [ ] **Step 5: Apply the toggle immediately and handle persistence failure honestly.** On interaction, update live `app_config` and call the existing Console appearance refresh seam immediately; persist through the same bounded async adapter used by other immediate presentation settings. If persistence fails, restore the prior checkbox/config value and show the existing content-free Settings error. The transcript recomputes model activities only, preserves scroll/tool expansion/answer focus, and clears selection if the selected model activity becomes hidden. Re-enable reconstructs supported historical rows collapsed; a still-live block may resume live expanded only if its pending lifecycle remains active.

- [ ] **Step 6: Extend `ConsoleContextControlState` with replay presentation.**

```python
@dataclass(frozen=True, slots=True)
class ThinkingHistoryControlState:
    saved_policy: ThinkingHistoryPolicy
    effective_label: Literal["Auto", "Include", "Exclude", "Required"]
    required_reason: str | None = None
```

Embed this state in the existing context snapshot. Required disables the Select and shows the specific continuation reason while retaining the saved optional value.

- [ ] **Step 7: Add current conversation and new-conversation default controls.** In the Context & memory section, add a compact Select for Auto/Include/Exclude, an effective state line, and `Save as default for new conversations`. The ordinary modal Save writes the conversation field through the store; the default action writes only `[console].thinking_history_policy_default`. Creating a new conversation copies the resolved device default into the new conversation field; existing NULL conversations continue to mean Auto and are not rewritten.

- [ ] **Step 8: Preserve modal result compatibility.** Add `thinking_history_policy` to `ConsoleSettingsResult` with a default for old constructor sites. Existing context overrides and model Save-as-default behavior remain unchanged. Session controller owns applying/persisting the conversation value; `ChatScreen` remains a bounded delegate.

- [ ] **Step 9: Make assistant edit provenance loss explicit.** When the selected assistant generation owns thinking or provider continuation, the existing edit modal's context copy states that saving the edit clears model thinking and provider continuation for that answer. Saving confirms the clear through the foundation's explicit edit path; Cancel leaves both intact. User-message edits and evidence-free assistant edits retain their current copy and behavior.

- [ ] **Step 10: Run Settings/Context/edit tests.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_settings_context_memory_controls.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_context_controls.py Tests/UI/test_console_session_settings.py Tests/Chat/test_console_context_policy_lifecycle.py Tests/Chat/test_console_edit_message_modal.py -k "thinking or context or default or provenance" -q`

Expected: PASS.

- [ ] **Step 11: Commit settings.**

```bash
git add tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/settings_context_memory.py tldw_chatbook/UI/Screens/settings_config_adapter.py tldw_chatbook/Widgets/Console/console_context_controls.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/Widgets/Console/console_edit_message_modal.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_settings_context_memory_controls.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_console_context_controls.py Tests/UI/test_console_session_settings.py Tests/Chat/test_console_context_policy_lifecycle.py Tests/Chat/test_console_edit_message_modal.py
git commit -m "feat: add Console thinking visibility and history controls"
```

---

### Task 4: Rename safe synthetic activity to Planning and complete visual verification

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `Tests/Chat/test_console_activity_presentation.py`
- Modify: `Tests/Chat/test_console_agent_bridge.py`
- Modify: `Tests/UI/test_console_thinking_disclosures.py`
- Modify: `Tests/UI/test_console_narrow_layout.py`
- Modify: `Tests/UI/test_console_transcript_window_reconcile.py`

**Interfaces consumed:** actual model block round ownership from TASK-18932.2; existing safe-summary sanitizer.

**Interfaces produced:** honest Planning markers, no duplicate round presentation, painted evidence.

- [ ] **Step 1: Write failing Planning distinction tests.** A safe intermediate primary `STEP_MODEL.summary` with no actual block yields `ConsoleActivityPresentation("planning", "Planning", "done")`. The same round with a displayable or proprietary block yields model Thinking only. Final model round still yields neither synthetic Planning nor duplicate Thinking.

- [ ] **Step 2: Run bridge tests and confirm current Thinking label fails.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py -k "planning or thinking" -q`

Expected: FAIL because safe markers are currently labeled Thinking.

- [ ] **Step 3: Rename presentation only and suppress by round ownership.** Keep `safe_intermediate_thinking_summary`'s conservative rejection logic unchanged for compatibility; rename the builder/deriver where practical and output kind/label `planning`/`Planning`. Pass the set of model-round ordinals with actual thinking into the live/resume derivation so it omits synthetic activity for those rounds.

```python
def build_intermediate_planning_marker(summary: str | None) -> ConsoleChatMessage | None:
    safe_summary = safe_intermediate_thinking_summary(summary)
    if safe_summary is None:
        return None
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content=safe_summary,
        activity_presentation=ConsoleActivityPresentation(
            "planning", "Planning", "done"
        ),
    )
```

Keep a deprecated internal alias only if existing callers require one during the same commit; remove it when all callers/tests move.

- [ ] **Step 4: Run bridge and UI regressions.**

Run: `PYTHONPATH=. .venv/bin/python -m pytest Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py Tests/UI/test_console_thinking_disclosures.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_window_reconcile.py -q`

Expected: PASS.

- [ ] **Step 5: Run the bounded Impeccable visual pass.** With an isolated profile and deterministic provider fixture, inspect painted screenshots at 60x18, 80x24, 100x30, and 140x42 in one batch. Include live displayable, collapsed historical, expanded proprietary, long-line wrapping, and Settings/Context states. Check readable status text, stable one-row headers, focus outline, no color-only state, no clipped unavailable label, no scroll jump, and no full body mounted while collapsed. Fix all observed defects in one batch, regenerate CSS, then perform one confirmation batch.

- [ ] **Step 6: Run narrow/painted tests and CSS/static gates.**

```bash
PYTHONPATH=. .venv/bin/python -m pytest Tests/UI/test_console_thinking_disclosures.py Tests/UI/test_console_narrow_layout.py Tests/UI/test_console_transcript_window_reconcile.py -q
.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
.venv/bin/python -m ruff format --check tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_turn_grouping.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Widgets/Console/console_assistant_turn.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/console_context_controls.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/UI/Screens/settings_context_memory.py tldw_chatbook/UI/Console_Modules/session.py Tests/Chat/test_console_thinking_presentation.py Tests/UI/test_console_thinking_disclosures.py
.venv/bin/python -m ruff check tldw_chatbook/Chat/console_chat_models.py tldw_chatbook/Chat/console_turn_grouping.py tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Widgets/Console/console_assistant_turn.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/console_context_controls.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/UI/Screens/settings_context_memory.py tldw_chatbook/UI/Console_Modules/session.py Tests/Chat/test_console_thinking_presentation.py Tests/UI/test_console_thinking_disclosures.py
git diff --check
```

- [ ] **Step 7: Commit Planning/QA and close TASK-18932.3.**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_activity_presentation.py Tests/Chat/test_console_agent_bridge.py Tests/UI/test_console_thinking_disclosures.py Tests/UI/test_console_narrow_layout.py Tests/UI/test_console_transcript_window_reconcile.py
git commit -m "fix: distinguish Console planning from model thinking"
```

Update TASK-18932.3 ACs, add Implementation Notes with the visual matrix and exact test/CSS evidence, and set it `Done` only after all checks pass.
