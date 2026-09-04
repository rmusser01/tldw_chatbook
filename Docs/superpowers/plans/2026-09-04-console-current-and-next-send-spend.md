# Console Current and Next-Send Spend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Console status chip show context fullness, current conversation spend, and an honest incremental request/input estimate for the next send.

**Spec:** `Docs/superpowers/specs/2026-09-04-console-current-and-next-send-spend-design.md`

**Architecture:** Keep display formatting in the existing immutable Console cost/context state builders. The screen owns live inputs: it separates sent transcript rows from staged next-send inputs, adds the mounted composer draft to the shared context estimate, derives an independent selected-model input forecast, and coalesces idle draft refreshes. The status-strip widget remains a pure renderer and gains a resize path that reapplies full versus compact copy.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, existing pricing catalog and Console state builders.

**Global constraints:** Preserve all unrelated dirty-worktree changes. In particular, do not edit, import, stage, or delete the unrelated untracked `tldw_chatbook/UI/Console_Modules/send_price.py` or `Tests/UI/test_console_send_price.py`; their output-inclusive upper-bound contract conflicts with this approved input-only baseline. Do not run the full test suite without user opt-in.

---

### Task 1: Reopen and extend the tracked backlog task

**Files:**
- Modify: `backlog/tasks/task-31382 - Make-Console-context-status-and-automatic-compaction-truthful.md`

- [ ] **Step 1: Move Task 31382 back to In Progress**

Run: `backlog task edit 31382 --status "In Progress" --plain`

- [ ] **Step 2: Add acceptance criteria for explicit spend timing**

Add criteria covering exact current/next-send labels, draft and staged-input ownership, media/unknown fallbacks, and responsive resize behavior. Preserve the existing completed criteria and link `Docs/superpowers/specs/2026-09-04-console-current-and-next-send-spend-design.md` from the plan/notes.

- [ ] **Step 3: Record the ADR check**

ADR required: no  
ADR paths: `backlog/decisions/052-console-conversation-memory-and-compaction-policy.md`, `backlog/decisions/095-conversation-owned-console-generation-settings.md`  
Reason: this is display and estimation behavior inside existing ownership boundaries.

### Task 2: Define the pure current/next-send presentation contract

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_context_controls.py:157`
- Test: `Tests/UI/test_console_context_controls.py:350`

- [ ] **Step 1: Write failing formatter tests**

Add tests expecting:

```python
assert state.label == (
    "Context 45% · Current $0.48 ● · On next send ~+$0.13"
)
assert state.compact_label == (
    "Ctx 45% · Now $0.48 ● · Next ~+$0.13"
)
assert "On next send: ~+$0.13 uncached input baseline" in state.tooltip
assert "Response/output spend is added after completion." in state.tooltip
assert "Cache reads may lower it; cache writes may raise it." in state.tooltip
```

Cover unavailable, no-sendable-input (`—`), independent token-only current spend, and explicit local `$0.00` forecast states without parsing already-formatted labels.

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest -q Tests/UI/test_console_context_controls.py -k "context_cost_state"`

Expected: failures because the formatter has no next-send input and still renders the old single spend value.

- [ ] **Step 3: Add the minimum immutable forecast state**

Add a focused value object beside `ConsoleContextControlState`:

```python
@dataclass(frozen=True, slots=True)
class ConsoleNextSendSpendState:
    label: str
    tooltip: str
```

Add a pure builder with explicit orthogonal inputs:

```python
def build_console_next_send_spend_state(
    *,
    request_tokens: int | None,
    input_per_mtok: float | None,
    sendable_text: bool,
    has_media: bool,
) -> ConsoleNextSendSpendState:
    ...
```

Extend `build_console_context_cost_state` to accept this state, use `cost.compact_label` for the current value so the old cache-delta suffix cannot masquerade as the full next-send charge, and emit the approved wide/narrow labels. Keep the cost tooltip and inspector action intact. Apply fallback precedence exactly in this order:

1. `has_media=True` produces `unavailable`, including attachment-only sends.
2. Missing request tokens or missing selected-model pricing produces `unavailable`.
3. Only the clean no-media, known-token, known-pricing, empty-draft state produces `—`.
4. Otherwise render `~+$<request input estimate>`, including a real `$0.00` for a known zero rate.

Add combination tests for media plus empty draft, media plus unknown pricing, clean empty draft, and sendable text plus known zero pricing so a later refactor cannot reorder these predicates.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `.venv/bin/pytest -q Tests/UI/test_console_context_controls.py -k "context_cost_state"`

Expected: PASS.

### Task 3: Separate current spend from the next-send forecast

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:5744-5910`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:10109-10351`
- Test: `Tests/UI/test_console_cost_chip_screen.py`
- Test: `Tests/Chat/test_console_cost_tracker.py`

- [ ] **Step 1: Write failing screen/state tests**

Cover these externally visible behaviors:

1. Staged evidence no longer increases `Current`; it increases `On next send`.
2. A mounted composer draft contributes to the context percentage and next-send estimate.
3. A draft edit while idle updates the chip after one coalesced delay.
4. Current pricing and selected-model forecast pricing fail independently.
5. Pending or historical media makes the next-send total unavailable while preserving text-token detail in the tooltip.
6. A known zero-rate local model renders `~+$0.00`; an empty unsendable conversation renders `On next send —`.
7. An unpriced current transcript containing locally estimated historical rows renders `~12.3k tok` and explains `Includes locally estimated transcript rows.`; it never calls those sent rows “unsent.”

Use concrete arrangements such as:

```python
# Empty transcript, known selected-model pricing.
state = console._build_console_cost_state()
assert "Current $0.00" in state.label
assert "On next send —" in state.label

# Live idle draft: the forecast must update without starting a send.
composer.load_draft("hello forecast")
await pilot.pause(0.25)
assert console._last_console_cost_state != initial_state
assert "On next send ~+$" in console._last_console_cost_state.label

# Staged evidence belongs only to the forecast.
assert current_amount_after_staging == current_amount_before_staging
assert next_send_amount_after_staging > next_send_amount_before_staging
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest -q Tests/UI/test_console_cost_chip_screen.py -k "current or next_send or staged or draft or media"`

Expected: failures showing staged evidence is still folded into current spend, draft text is absent, and no independent forecast exists.

- [ ] **Step 3: Include the live draft in the shared context estimate**

In `_console_settings_context_estimate_for_session`, read `self._console_composer_or_none()` and append one synthetic user message only for the active mounted composer when `draft_text().strip()` is non-empty. Keep inactive-session settings previews unchanged.

- [ ] **Step 4: Keep staged inputs out of current spend**

Build `ConsoleCostSnapshot` from sent transcript rows only. Do not append the staged-evidence pseudo-row to the current snapshot. Preserve locally estimated historical transcript rows and unpriced fleet-token disclosure.

- [ ] **Step 5: Build the next-send input estimate independently**

Resolve selected-model pricing independently from `snapshot.pricing_known`. Define `sendable_text` exactly as `bool(active_mounted_composer.draft_text().strip())`. Define media as any row in `store.pending_attachments(session_id)`, any historical `message.attachments`, or legacy `message.image_data is not None`. When there is sendable text, known `request_tokens`, no media, and selected-model pricing, compute:

```python
next_send_usd = round(
    context_state.request_tokens * pricing.input_per_mtok / 1_000_000,
    6,
)
```

Format it with `format_cost_amount`. Otherwise build the precise `—` or `unavailable` state and reason from the approved state matrix. Treat `input_per_mtok == 0.0` as known, not missing.

For an empty transcript with known selected-model pricing, explicitly rebase the current snapshot before formatting:

```python
if snapshot.row_count == 0 and pricing is not None:
    snapshot = replace(snapshot, total_usd=0.0, pricing_known=True)
```

This is the minimum branch needed for the exact `Current $0.00 · On next send —` contract and does not change non-empty unknown historical pricing.

In `build_cost_state`, keep the existing tokens-only fallback but prefix its label with `~` when `snapshot.has_estimated_entries` and replace `Includes estimated (unsent) rows.` with `Includes locally estimated transcript rows.`. Add a focused `Tests/Chat/test_console_cost_tracker.py` regression before this production edit and verify it fails on both the missing prefix and stale ownership copy.

- [ ] **Step 6: Coalesce idle draft refreshes**

Add one screen-owned timer slot. On the existing hidden-input `Input.Changed` path, first inspect the active session's `controller.run_state_for(session_id).status`. Active runs already own the transcript sync timer, so schedule nothing for statuses in `CONSOLE_ACTIVE_RUN_STATUSES`. While idle, stop any prior pending timer and schedule a 0.2-second callback that calls `_sync_console_settings_summary()` followed by `_sync_console_cost_chip()`. Stop the timer on unmount. Do not run full native-session reconciliation per keystroke.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `.venv/bin/pytest -q Tests/UI/test_console_cost_chip_screen.py Tests/Chat/test_console_cost_tracker.py -k "current or next_send or staged or draft or media or cost_state"`

Expected: PASS.

### Task 4: Make compact/full labels respond to live resizing

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_status_chips.py:800-860`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:3207-3265`
- Regenerate carefully: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/Chat/test_console_status_chips_cost.py:240-330`

- [ ] **Step 1: Write a failing live-resize test**

First, cold-mount at 80 columns with the combined state already supplied and assert compact copy appears without calling `sync_cost_state` with a different state. Separately mount once at a wide width, assert `Context … Current … On next send …`, resize below 120 columns, assert `Ctx … Now … Next …`, resize wide again, and assert the full label returns without changing the cost state.

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/pytest -q Tests/Chat/test_console_status_chips_cost.py -k "resize or context_cost_label"`

Expected: the label stays in its initial form because equal cost state short-circuits `sync_cost_state`.

- [ ] **Step 3: Centralize label selection and handle resize**

Extract a private renderer that selects `label` versus `compact_label` from the current width and applies it to the mounted chip. Call it from `sync_cost_state` and Textual's resize event; keep state equality guarding expensive non-layout updates. Add a cost-chip-only width rule after `.console-control-chip`:

```tcss
#console-cost-chip {
    max-width: 100%;
}
```

Run `.venv/bin/python tldw_chatbook/css/build_css.py` so the generated bundle matches the source module. Before and after regeneration, inspect the diff for both CSS files and preserve all pre-existing roleplay/console CSS hunks. In mounted tests, assert the rendered label contains no ellipsis and the chip content width is at least the rendered cell width for `Context 100%+` and a multi-digit current/forecast amount.

- [ ] **Step 4: Run mounted tests and verify GREEN**

Run: `.venv/bin/pytest -q Tests/Chat/test_console_status_chips_cost.py -k "resize or context_cost_label or keyboard_activation"`

Expected: PASS at initial narrow mount and across both resize directions.

### Task 5: Verify and close the task

**Files:**
- Modify: `backlog/tasks/task-31382 - Make-Console-context-status-and-automatic-compaction-truthful.md`

- [ ] **Step 1: Run focused behavior verification**

Run the exact new formatter, live-screen, resize, keyboard, and automatic-compaction tests. Do not run the full suite without user opt-in.

- [ ] **Step 2: Run static checks**

Run scoped `.venv/bin/ruff check`, `.venv/bin/python -m py_compile`, `.venv/bin/python tldw_chatbook/css/check_bundle_sync.py`, and `git diff --check` over the touched production/test files. Preserve unrelated dirty-worktree changes.

- [ ] **Step 3: Review the diff against the spec**

Confirm current spend excludes staged/draft inputs, next-send excludes output/media, cache alerts remain explained, and neither unrelated untracked file (`tldw_chatbook/UI/Console_Modules/send_price.py`, `Tests/UI/test_console_send_price.py`) was edited, imported, staged, or deleted.

- [ ] **Step 4: Complete Task 31382**

Check the added acceptance criteria, replace Implementation Notes with the combined first and second implementation phases plus exact verification evidence, and set the task to Done.
