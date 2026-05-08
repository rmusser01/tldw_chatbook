# Console Large Paste Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Console composer feature that collapses large pasted chunks into compact reversible tokens while preserving the exact send payload.

**Architecture:** `ConsoleComposerBar` becomes the source of truth for segmented draft state: normal text segments plus optional collapsed paste segments. `ChatScreen` routes paste events through an explicit paste insertion API, while normal typing and draft restore remain literal. Settings exposes a default-enabled global toggle backed by config.

**Tech Stack:** Python 3.12, Textual widgets/events, Rich `Text`, existing TOML config helpers, pytest mounted UI tests, actual Textual/browser screenshot QA.

---

## References

- Spec: `Docs/superpowers/specs/2026-05-08-console-large-paste-collapse-design.md`
- Existing composer: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Existing Console screen routing: `tldw_chatbook/UI/Screens/chat_screen.py`
- Existing Settings shell: `tldw_chatbook/UI/Screens/settings_screen.py`
- Existing config helpers/defaults: `tldw_chatbook/config.py`
- Existing Console tests: `Tests/UI/test_console_internals_decomposition.py`
- Existing destination shell tests: `Tests/UI/test_destination_shells.py`

Use:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
```

Hard constraints:

- Do not replace the Console composer with the legacy chat input.
- Do not collapse normal typing, `load_draft()`, or restored/session drafts.
- Do not send collapsed labels such as `Pasted Text: X Characters` or `Unfurl?`.
- Do not approve visual state without actual rendered screenshots and user approval.
- Do not commit `.playwright-cli/`; copy approved screenshot artifacts into `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/`.

---

## File Structure

Modify:

- `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  - Add `ConsoleDraftSegment`.
  - Add segment state, `insert_pasted_text()`, collapsed display labels, two-step unfurl state, atomic Backspace, and click hit handling.
  - Keep `draft_text()`, `load_draft()`, `insert_text()`, `delete_left()`, `clear_draft()`, hidden input sync, and current composer sizing behavior stable.

- `tldw_chatbook/UI/Screens/chat_screen.py`
  - Route `on_paste()` to `insert_pasted_text()`.
  - Keep printable key handling on `insert_text()`.
  - Reset pending unfurl state on click-away if composer-level handling is insufficient.
  - Configure composer from `app_config["console"]`.

- `tldw_chatbook/UI/Screens/settings_screen.py`
  - Add a global Console behavior toggle for `Collapse large pasted text`.
  - Persist to `console.collapse_large_pastes`.

- `tldw_chatbook/config.py`
  - Add default `[console] collapse_large_pastes = true`.
  - Add default `[console] large_paste_collapse_threshold = 50`.

- `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add token styling only if Rich text styling is insufficient.

- `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Regenerate with `tldw_chatbook/css/build_css.py` only if TCSS source changes.

Test:

- `Tests/UI/test_console_internals_decomposition.py`
  - Add focused mounted tests for collapse, threshold, literal typing, paste routing, send payload, atomic Backspace, unfurl, reset, multiple chunks, disabled config, and draft restore.

- `Tests/UI/test_destination_shells.py`
  - Add or extend Settings mounted coverage for the toggle.

---

## Task 1: Add Segmented Draft Model And Core Paste Collapse

**Files:**

- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`

- [ ] **Step 1: Add failing tests for core segment behavior**

Add these tests near the existing Console native composer tests, using the same `_build_test_app()`, `ConsoleHarness`, `_wait_for_selector()`, `ConsoleComposerBar`, and `Static` patterns already in the file:

```python
@pytest.mark.asyncio
async def test_console_large_paste_collapses_but_preserves_payload():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted = "x" * 51

        composer.insert_pasted_text(pasted)
        await pilot.pause(0.1)

        assert composer.draft_text() == pasted
        assert visible_draft.renderable.plain == "Pasted Text: 51 Characters"
        assert pasted not in visible_draft.renderable.plain
```

```python
@pytest.mark.asyncio
async def test_console_paste_at_threshold_remains_literal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted = "x" * 50

        composer.insert_pasted_text(pasted)
        await pilot.pause(0.1)

        assert composer.draft_text() == pasted
        assert visible_draft.renderable.plain == pasted
```

```python
@pytest.mark.asyncio
async def test_console_paste_under_threshold_remains_literal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted = "short pasted text"

        composer.insert_pasted_text(pasted)
        await pilot.pause(0.1)

        assert composer.draft_text() == pasted
        assert visible_draft.renderable.plain == pasted
```

```python
@pytest.mark.asyncio
async def test_console_clear_draft_keeps_canonical_payload_empty():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        hidden_input = composer.query_one("#console-command-input", Input)

        composer.load_draft("stale text")
        composer.clear_draft()
        await pilot.pause(0.1)

        assert composer.draft_text() == ""
        assert hidden_input.value == ""
```

```python
@pytest.mark.asyncio
async def test_console_typing_past_threshold_remains_literal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        typed = "t" * 60

        composer.insert_text(typed)
        await pilot.pause(0.1)

        assert composer.draft_text() == typed
        assert "Pasted Text:" not in visible_draft.renderable.plain
```

- [ ] **Step 2: Run tests and verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_large_paste_collapses_but_preserves_payload Tests/UI/test_console_internals_decomposition.py::test_console_paste_at_threshold_remains_literal Tests/UI/test_console_internals_decomposition.py::test_console_paste_under_threshold_remains_literal Tests/UI/test_console_internals_decomposition.py::test_console_clear_draft_keeps_canonical_payload_empty Tests/UI/test_console_internals_decomposition.py::test_console_typing_past_threshold_remains_literal --tb=short
```

Expected: fail because `ConsoleComposerBar.insert_pasted_text()` is missing.

- [ ] **Step 3: Add `ConsoleDraftSegment`**

In `console_composer_bar.py`, add:

```python
from dataclasses import dataclass, field
from typing import Literal
import uuid
```

Add near the top:

```python
@dataclass
class ConsoleDraftSegment:
    kind: Literal["text", "paste"]
    text: str
    collapsed: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
```

Add constants to `ConsoleComposerBar`:

```python
PASTE_COLLAPSE_THRESHOLD = 50
COLLAPSE_LARGE_PASTES_DEFAULT = True
```

Add state in `__init__()`:

```python
self._segments: list[ConsoleDraftSegment] = []
self._segments_initialized = False
self._pending_unfurl_segment_id: str | None = None
self._collapse_large_pastes = self.COLLAPSE_LARGE_PASTES_DEFAULT
self._large_paste_threshold = self.PASTE_COLLAPSE_THRESHOLD
```

- [ ] **Step 4: Make segments the source of truth**

Add helper methods. Empty `_segments` is a valid canonical empty draft after `clear_draft()` or deleting the final character, so hidden-input fallback must not reintroduce stale text.

```python
def _raw_draft_text(self) -> str:
    return "".join(segment.text for segment in self._segments)

def _merge_adjacent_text_segments(self, segments: list[ConsoleDraftSegment]) -> list[ConsoleDraftSegment]:
    merged: list[ConsoleDraftSegment] = []
    for segment in segments:
        if not segment.text:
            continue
        if merged and merged[-1].kind == "text" and segment.kind == "text":
            merged[-1].text += segment.text
            continue
        merged.append(segment)
    return merged

def _sync_hidden_input(self) -> None:
    try:
        self.query_one("#console-command-input", Input).value = self._raw_draft_text()
    except NoMatches:
        return

def _set_segments(self, segments: list[ConsoleDraftSegment]) -> None:
    self._segments_initialized = True
    self._segments = self._merge_adjacent_text_segments(segments)
    self._sync_hidden_input()
    self._refresh_visible_draft()
```

Update `draft_text()`:

```python
def draft_text(self) -> str:
    if self._segments_initialized:
        return self._raw_draft_text()
    try:
        return self.query_one("#console-command-input", Input).value
    except NoMatches:
        return ""
```

Update `load_draft(text)` to reset to one normal text segment and never collapse:

```python
self._pending_unfurl_segment_id = None
self._set_segments([ConsoleDraftSegment(kind="text", text=text)])
```

Update `insert_text(text)` to append literal text:

```python
self._pending_unfurl_segment_id = None
self._set_segments([*self._segments, ConsoleDraftSegment(kind="text", text=text)])
```

Add `insert_pasted_text(text)`:

```python
def insert_pasted_text(self, text: str) -> None:
    if not text:
        return
    should_collapse = (
        self._collapse_large_pastes
        and len(text) > self._large_paste_threshold
    )
    self._pending_unfurl_segment_id = None
    self._set_segments([
        *self._segments,
        ConsoleDraftSegment(kind="paste", text=text, collapsed=should_collapse),
    ])
```

Update `clear_draft()` to clear segments through `load_draft("")`.

- [ ] **Step 5: Render collapsed labels**

Add:

```python
def _segment_display_text(self) -> str:
    parts: list[str] = []
    for segment in self._segments:
        if segment.kind == "paste" and segment.collapsed:
            if self._pending_unfurl_segment_id == segment.id:
                parts.append("Unfurl?")
            else:
                parts.append(f"Pasted Text: {len(segment.text)} Characters")
        else:
            parts.append(segment.text)
    return "".join(parts)
```

Update `_refresh_visible_draft()` so the visible renderer wraps `_segment_display_text()` while hidden input sync remains based on `draft_text()`.

- [ ] **Step 6: Run green tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_large_paste_collapses_but_preserves_payload Tests/UI/test_console_internals_decomposition.py::test_console_paste_at_threshold_remains_literal Tests/UI/test_console_internals_decomposition.py::test_console_paste_under_threshold_remains_literal Tests/UI/test_console_internals_decomposition.py::test_console_clear_draft_keeps_canonical_payload_empty Tests/UI/test_console_internals_decomposition.py::test_console_typing_past_threshold_remains_literal Tests/UI/test_console_internals_decomposition.py::test_console_native_composer_receives_typing_on_open --tb=short
```

Expected: pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add Tests/UI/test_console_internals_decomposition.py tldw_chatbook/Widgets/Console/console_composer_bar.py
git commit -m "Add Console segmented draft paste collapse"
```

---

## Task 2: Route Paste Events And Preserve Send/Restore Boundaries

**Files:**

- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`

- [ ] **Step 1: Add failing tests for paste routing and boundaries**

Add:

```python
@pytest.mark.asyncio
async def test_console_native_composer_paste_collapses_large_draft_and_preserves_payload():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted_text = "pasted composer qa " * 80

        console.on_paste(Paste(pasted_text))
        await pilot.pause(0.2)

        assert composer.draft_text() == pasted_text
        assert visible_draft.renderable.plain == f"Pasted Text: {len(pasted_text)} Characters"
```

Add:

```python
@pytest.mark.asyncio
async def test_console_load_draft_keeps_large_text_literal():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        restored = "restored draft " * 30

        composer.load_draft(restored)
        await pilot.pause(0.1)

        assert composer.draft_text() == restored
        assert "Pasted Text:" not in visible_draft.renderable.plain
```

Add blocked-send payload preservation:

```python
@pytest.mark.asyncio
async def test_console_send_uses_full_paste_payload_not_collapsed_label(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config.setdefault("api_settings", {}).setdefault("openai", {})["api_key"] = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(212, 64)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        send_button = console.query_one("#console-send-message", Button)
        pasted = "payload " * 20

        composer.insert_pasted_text(pasted)
        await console.handle_console_send_message(Button.Pressed(send_button))
        await pilot.pause(0.2)

        assert composer.draft_text() == pasted
        assert "Pasted Text:" not in composer.draft_text()
        assert "Unfurl?" not in composer.draft_text()
        assert "Console send blocked" in _visible_text(console)
```

- [ ] **Step 2: Run tests and verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_native_composer_paste_collapses_large_draft_and_preserves_payload Tests/UI/test_console_internals_decomposition.py::test_console_load_draft_keeps_large_text_literal Tests/UI/test_console_internals_decomposition.py::test_console_send_uses_full_paste_payload_not_collapsed_label --tb=short
```

Expected: paste routing fails until `ChatScreen.on_paste()` calls `insert_pasted_text()`.

- [ ] **Step 3: Update paste routing**

In `ChatScreen.on_paste()`, change the Console path from `composer.insert_text(event.text)` to:

```python
composer.insert_pasted_text(event.text)
```

Do not change printable key handling; it must keep calling `insert_text()`.

- [ ] **Step 4: Run green tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_native_composer_paste_collapses_large_draft_and_preserves_payload Tests/UI/test_console_internals_decomposition.py::test_console_load_draft_keeps_large_text_literal Tests/UI/test_console_internals_decomposition.py::test_console_send_uses_full_paste_payload_not_collapsed_label Tests/UI/test_console_internals_decomposition.py::test_console_enter_sends_native_composer_draft --tb=short
```

Expected: pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add Tests/UI/test_console_internals_decomposition.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py
git commit -m "Route Console paste through collapse-aware draft API"
```

---

## Task 3: Add Atomic Backspace And Two-Step Unfurl

**Files:**

- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] **Step 1: Add failing behavior tests**

Add tests with the same mounted setup as Task 1:

```python
@pytest.mark.asyncio
async def test_console_backspace_removes_collapsed_paste_atomically():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        composer.insert_text("before ")
        composer.insert_pasted_text("x" * 80)
        composer.delete_left()
        await pilot.pause(0.1)

        assert composer.draft_text() == "before "
        assert "Pasted Text:" not in visible_draft.renderable.plain
```

```python
@pytest.mark.asyncio
async def test_console_collapsed_paste_two_step_unfurls_to_normal_text():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted = "unfurl payload " * 8

        composer.insert_pasted_text(pasted)
        segment_id = composer.collapsed_paste_segment_ids()[0]
        composer.activate_segment(segment_id)
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == "Unfurl?"

        composer.activate_segment(segment_id)
        await pilot.pause(0.1)
        assert visible_draft.renderable.plain == pasted
```

```python
@pytest.mark.asyncio
async def test_console_collapsed_paste_real_click_enters_unfurl_prompt():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted = "click payload " * 8

        composer.insert_pasted_text(pasted)
        await pilot.pause(0.1)
        await pilot.click("#console-command-visible-text")
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == "Unfurl?"
        assert composer.draft_text() == pasted
```

```python
@pytest.mark.asyncio
async def test_console_collapsed_paste_unfurl_prompt_resets_on_click_away():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted = "reset prompt " * 8

        composer.insert_pasted_text(pasted)
        segment_id = composer.collapsed_paste_segment_ids()[0]
        composer.activate_segment(segment_id)
        composer.reset_pending_unfurl()
        await pilot.pause(0.1)

        assert visible_draft.renderable.plain == f"Pasted Text: {len(pasted)} Characters"
```

```python
@pytest.mark.asyncio
async def test_console_multiple_collapsed_pastes_are_independent():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)

        composer.insert_pasted_text("a" * 60)
        composer.insert_text(" middle ")
        composer.insert_pasted_text("b" * 70)
        ids = composer.collapsed_paste_segment_ids()
        assert len(ids) == 2

        composer.activate_segment(ids[0])
        composer.activate_segment(ids[0])
        await pilot.pause(0.1)

        assert "a" * 60 in visible_draft.renderable.plain
        assert "Pasted Text: 70 Characters" in visible_draft.renderable.plain
```

- [ ] **Step 2: Run tests and verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_backspace_removes_collapsed_paste_atomically Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_two_step_unfurls_to_normal_text Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_real_click_enters_unfurl_prompt Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_unfurl_prompt_resets_on_click_away Tests/UI/test_console_internals_decomposition.py::test_console_multiple_collapsed_pastes_are_independent --tb=short
```

Expected: fail because activation/reset helpers and atomic delete do not exist yet.

- [ ] **Step 3: Implement atomic Backspace**

Update `delete_left()` so a final collapsed paste segment is removed whole:

```python
if not self._segments:
    return
last = self._segments[-1]
if last.kind == "paste" and last.collapsed:
    self._pending_unfurl_segment_id = None
    self._set_segments(self._segments[:-1])
    return
last.text = last.text[:-1]
self._pending_unfurl_segment_id = None
self._set_segments(self._segments)
```

- [ ] **Step 4: Implement activation/reset helpers**

Add:

```python
def collapsed_paste_segment_ids(self) -> list[str]:
    return [
        segment.id
        for segment in self._segments
        if segment.kind == "paste" and segment.collapsed
    ]

def activate_segment(self, segment_id: str) -> None:
    for segment in self._segments:
        if segment.id != segment_id:
            continue
        if segment.kind != "paste" or not segment.collapsed:
            self.reset_pending_unfurl()
            return
        if self._pending_unfurl_segment_id == segment_id:
            segment.collapsed = False
            self._pending_unfurl_segment_id = None
            self._set_segments(self._segments)
            return
        self._pending_unfurl_segment_id = segment_id
        self._refresh_visible_draft()
        return
    self.reset_pending_unfurl()

def reset_pending_unfurl(self) -> None:
    if self._pending_unfurl_segment_id is None:
        return
    self._pending_unfurl_segment_id = None
    self._refresh_visible_draft()
```

- [ ] **Step 5: Add actual click handling**

Use the existing `#console-command-visible-text` as the click target for the first implementation. Track token labels when `_segment_display_text()` renders collapsed segments:

```python
self._visible_token_ranges: list[tuple[str, int, int]] = []
```

Implement `_segment_id_from_click(event)` using visible draft-relative coordinates and token ranges. The first required path is the common case where the collapsed token is the only visible draft content; `await pilot.click("#console-command-visible-text")` must enter the `Unfurl?` state.

Add `on_click()` to `ConsoleComposerBar`:

```python
from textual.events import Click

def on_click(self, event: Click) -> None:
    target_id = getattr(event.widget, "id", None)
    if target_id != "console-command-visible-text":
        self.reset_pending_unfurl()
        return
    segment_id = self._segment_id_from_click(event)
    if segment_id:
        self.activate_segment(segment_id)
        event.stop()
        return
    self.reset_pending_unfurl()
```

If coordinate hit-testing cannot be made reliable, replace the visible collapsed label with a small `ConsolePasteToken` widget and update the real-click test to click that widget selector. Do not rely only on `activate_segment()` helper tests; at least one mounted click test must pass.

- [ ] **Step 6: Add click-away reset**

Prefer composer-level click-away reset. If clicks outside the composer do not reach it, add a conservative `ChatScreen` helper:

```python
def _reset_console_composer_unfurl(self) -> None:
    try:
        self.query_one("#console-native-composer", ConsoleComposerBar).reset_pending_unfurl()
    except NoMatches:
        return
```

Do not reset pending unfurl before processing a token activation. Any `ChatScreen` click-away handling must ignore events already stopped by the composer token click path.

- [ ] **Step 7: Run green tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_backspace_removes_collapsed_paste_atomically Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_two_step_unfurls_to_normal_text Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_real_click_enters_unfurl_prompt Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_unfurl_prompt_resets_on_click_away Tests/UI/test_console_internals_decomposition.py::test_console_multiple_collapsed_pastes_are_independent --tb=short
```

Expected: pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add Tests/UI/test_console_internals_decomposition.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "Add Console paste token unfurl interactions"
```

---

## Task 4: Add Config Defaults And Settings Toggle

**Files:**

- Modify: `Tests/UI/test_console_internals_decomposition.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`

- [ ] **Step 1: Add failing disabled-setting test**

Add:

```python
@pytest.mark.asyncio
async def test_console_large_paste_collapse_can_be_disabled_from_app_config():
    app = _build_test_app()
    app.app_config.setdefault("console", {})["collapse_large_pastes"] = False
    host = ConsoleHarness(app)

    async with host.run_test(size=(140, 42)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        visible_draft = composer.query_one("#console-command-visible-text", Static)
        pasted = "disabled setting " * 10

        composer.insert_pasted_text(pasted)
        await pilot.pause(0.1)

        assert composer.draft_text() == pasted
        assert "Pasted Text:" not in visible_draft.renderable.plain
```

- [ ] **Step 2: Add failing Settings UI test**

In `Tests/UI/test_destination_shells.py`, add or extend the Settings destination test so it mounts Settings and asserts:

```python
await _wait_for_selector(settings, pilot, "#settings-collapse-large-pastes")
assert "Collapse large pasted text" in _visible_text(settings)
assert settings.query_one("#settings-collapse-large-pastes", Switch).value is True
```

Import `Switch` if needed.

- [ ] **Step 3: Run tests and verify red state**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_large_paste_collapse_can_be_disabled_from_app_config Tests/UI/test_destination_shells.py --tb=short
```

Expected: fail until config read and Settings switch exist.

- [ ] **Step 4: Add config defaults**

In `config.py` default TOML, add:

```toml
[console]
collapse_large_pastes = true
large_paste_collapse_threshold = 50
```

- [ ] **Step 5: Configure composer from app config**

Add to `ConsoleComposerBar`:

```python
def configure_paste_collapse(self, *, enabled: bool = True, threshold: int = 50) -> None:
    self._collapse_large_pastes = bool(enabled)
    self._large_paste_threshold = max(1, int(threshold or self.PASTE_COLLAPSE_THRESHOLD))
```

In `ChatScreen`, after the composer is available, call:

```python
config = getattr(self.app_instance, "app_config", {}) or {}
console_config = config.get("console", {}) if isinstance(config, dict) else {}
composer.configure_paste_collapse(
    enabled=console_config.get("collapse_large_pastes", True),
    threshold=console_config.get("large_paste_collapse_threshold", 50),
)
```

- [ ] **Step 6: Add Settings toggle**

In `settings_screen.py`, import `Switch` and add the visible toggle under the App-level behavior / detail pane:

```python
console_config = getattr(self.app_instance, "app_config", {}).get("console", {})
collapse_large_pastes = bool(console_config.get("collapse_large_pastes", True))
yield Static("Console Behavior", classes="destination-section")
yield Static("Collapse large pasted text")
yield Switch(
    value=collapse_large_pastes,
    id="settings-collapse-large-pastes",
    tooltip="Pastes over 50 characters appear as compact tokens until unfurled.",
)
yield Static(
    "Pastes over 50 characters appear as compact tokens until unfurled.",
    id="settings-collapse-large-pastes-help",
)
```

Add:

```python
@on(Switch.Changed, "#settings-collapse-large-pastes")
def save_console_large_paste_collapse(self, event: Switch.Changed) -> None:
    from ...config import save_setting_to_cli_config

    self.app_instance.app_config.setdefault("console", {})["collapse_large_pastes"] = event.value
    if save_setting_to_cli_config("console", "collapse_large_pastes", event.value):
        self.app.notify("Console paste collapse setting saved.", severity="information")
    else:
        self.app.notify("Failed to save Console paste collapse setting.", severity="error")
```

- [ ] **Step 7: Run green tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py::test_console_large_paste_collapse_can_be_disabled_from_app_config Tests/UI/test_destination_shells.py --tb=short
```

Expected: pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add Tests/UI/test_console_internals_decomposition.py Tests/UI/test_destination_shells.py tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_composer_bar.py
git commit -m "Add Console paste collapse setting"
```

---

## Task 5: Visual Styling And Actual Screenshot QA

**Files:**

- Modify if needed: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify if needed: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create/update: `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/`

- [ ] **Step 1: Add token styling only if needed**

If collapsed token display needs a clearer visual affordance, add focused styles to `_agentic_terminal.tcss`:

```css
.console-paste-token {
    background: $ds-surface-raised;
    color: $ds-text-primary;
    text-style: bold;
}

.console-paste-token-pending {
    background: $ds-accent-muted;
    color: $ds-text-primary;
    text-style: bold;
}
```

Skip this if Rich text styling is sufficient.

- [ ] **Step 2: Regenerate generated CSS if TCSS changed**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: exits 0. Do not edit `tldw_chatbook/css/tldw_cli_modular.tcss` directly.

- [ ] **Step 3: Run actual app through textual-serve**

Start or reuse textual-serve:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-visual-ui-correction
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/textual-serve "cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-visual-ui-correction && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app" --host 127.0.0.1 --port 8765
```

If macOS/browser automation requires escalation, request it.

- [ ] **Step 4: Capture collapsed token screenshot**

Use browser automation against `http://127.0.0.1:8765`:

- Open Console.
- Paste a string over 50 characters into the composer.
- Verify `Pasted Text: X Characters` is visible.
- Save to `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/console-large-paste-collapsed-2026-05-08.png`.

- [ ] **Step 5: Capture `Unfurl?` screenshot**

- Click the collapsed token once.
- Verify `Unfurl?` is visible.
- Save to `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/console-large-paste-unfurl-prompt-2026-05-08.png`.

- [ ] **Step 6: Capture unfurled text screenshot**

- Click `Unfurl?` again.
- Verify normal text is visible in the bounded composer.
- Save to `Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures/console-large-paste-unfurled-2026-05-08.png`.

- [ ] **Step 7: Ask user for screenshot approval**

Show actual screenshot paths and wait for user approval. Do not claim approval before the user approves screenshots.

- [ ] **Step 8: Commit Task 5**

Only after screenshot files are saved:

```bash
git add Docs/superpowers/qa/product-maturity/phase-3/actual-visual-captures
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "Capture Console large paste collapse QA"
```

If no CSS changed, omit the CSS files.

---

## Task 6: Final Verification And PR Readiness

**Files:**

- Review all files touched by Tasks 1-5.

- [ ] **Step 1: Run focused Console tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_console_internals_decomposition.py --tb=short
```

Expected: all tests pass.

- [ ] **Step 2: Run destination shell/settings tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py --tb=short
```

Expected: all tests pass.

- [ ] **Step 3: Run Console shell contract smoke**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_gate1_core_loop_screen_adaptation.py::test_console_core_loop_exposes_agentic_shell_regions --tb=short
```

Expected: pass.

- [ ] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output and exit 0.

- [ ] **Step 5: Inspect status**

Run:

```bash
git status --short
```

Expected:

- Only intentional source/test/spec/plan/QA files are modified or committed.
- `.playwright-cli/` is not staged.
- Existing unrelated dirty files from the broader visual correction branch are not reverted.

- [ ] **Step 6: Summarize implementation**

Final summary must include:

- Behavior implemented.
- Settings default and toggle.
- Test commands and results.
- Screenshot paths and approval status.
- Any residual risks.

Do not mark the screen approved unless the user approved the actual screenshots.
