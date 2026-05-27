# Console Setup-Aware Empty State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Console empty transcript and composer accurately reflect setup-blocked states, especially missing provider/model configuration, without changing the broader Console layout.

**Architecture:** Reuse the existing Console state flow instead of adding a new state owner. `ChatScreen` already knows the send-blocked reason, `ConsoleSessionSurface` already mediates transcript guidance, `ConsoleTranscript` owns the empty transcript rendering, and `ConsoleComposerBar.sync_action_state()` already controls send availability.

**Tech Stack:** Python 3.12, Textual widgets, existing Console widget tests under `Tests/UI/`, focused pytest verification.

---

## Scope

This slice fixes the misleading first-run/setup-blocked state seen in the current Console screen:

- The transcript must not say "Ready" when sending is blocked by missing setup.
- The composer must explain why Send is disabled when the blocker is provider/model setup.
- The existing Console Settings `Configure` path remains the setup action. Do not add a second modal or redesign the side rail in this slice.
- Keep layout structure unchanged: left rail open, center transcript dominant, right inspector collapsible.

Do not address the later review items in this plan:

- Top status row density.
- Left rail hierarchy beyond setup copy.
- Left rail scrolling/clipping.
- Message action row redesign.

## Files

- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
  - Owns empty transcript copy and renders `.console-transcript-empty-state`.
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
  - Forwards setup-aware empty copy from `ChatScreen` to `ConsoleTranscript`.
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
  - Shows a setup-blocked placeholder and send tooltip when send is blocked.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  - Computes a concise setup blocker copy and passes it to transcript/composer sync.
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Adds a small visual state for setup-blocked composer copy.
  - Aligns stale right rail handle width declarations with the current inline width.
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Run `python tldw_chatbook/css/build_css.py` after TCSS edits.
- Test: `Tests/UI/test_console_native_transcript.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_console_session_settings.py`
- Test: `Tests/UI/test_console_persistent_rails.py`

## Acceptance Criteria

- [ ] When provider/model setup blocks sending and the transcript has no messages, the empty transcript text explains the setup blocker instead of saying "Ready".
- [ ] When setup blocks sending and the composer is empty, the visible composer draft area explains the blocker instead of showing the generic placeholder.
- [ ] When the user types a draft while setup remains blocked, their draft remains visible and Send remains disabled with a specific tooltip.
- [ ] Active-run blocking remains distinct from setup blocking: the composer keeps the existing active-run tooltip and does not show setup copy while a run is merely in progress.
- [ ] When setup becomes ready, the transcript and composer return to the normal empty copy and Send behavior.
- [ ] Provider/API key details are not duplicated across the provider recovery strip, transcript empty state, and composer placeholder. Exact provider details such as "missing API key" remain in the recovery strip or action tooltip only.
- [ ] Existing message rendering, paste collapse, tab rename, rail persistence, and right-rail setup badge behavior remain intact.
- [ ] CSS source and generated modular CSS agree for the right collapsed inspector rail widths.

---

## Copy Contract

Use these strings unless a test proves they need a minor grammar adjustment:

- Missing model transcript: `Choose a model in Console Settings to start chatting.`
- Missing model composer placeholder: `Setup required: choose a model in Console Settings.`
- Provider/API key transcript: `Finish provider setup to start chatting.`
- Provider/API key composer placeholder: `Setup required: finish provider setup.`
- Active run tooltip: keep the existing `Wait for the active Console run to finish before sending.`

Do not repeat exact provider recovery details such as `missing API key` in the transcript or composer. The provider recovery strip already owns those details and existing tests assert the text is not duplicated.

### Task 1: Make ConsoleTranscript Empty Copy Stateful

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Test: `Tests/UI/test_console_native_transcript.py`

- [ ] **Step 1: Write failing test for custom empty copy**

Add a focused widget test. Do not use `TranscriptHarness` if it mounts default messages; define a small empty harness or clear the messages before asserting:

```python
async def test_console_transcript_empty_state_accepts_setup_copy():
    app = EmptyTranscriptHarness()
    async with app.run_test() as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)

        transcript.sync_empty_state("Choose a model in Console Settings to start chatting.")
        await pilot.pause()

        empty_state = transcript.query_one(".console-transcript-empty-state", Static)
        assert empty_state.renderable.plain == (
            "Choose a model in Console Settings to start chatting."
        )
```

Keep `to_plain_text()` unchanged unless there is a separate product reason to include empty-state copy in transcript export. It currently represents messages, not the empty placeholder.

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py::test_console_transcript_empty_state_accepts_setup_copy -q
```

Expected: FAIL because `ConsoleTranscript` has no `sync_empty_state()` method or still renders `EMPTY_TRANSCRIPT_COPY`.

- [ ] **Step 3: Implement minimal transcript state**

In `ConsoleTranscript.__init__()`:

```python
self.empty_state_copy = EMPTY_TRANSCRIPT_COPY
```

Add:

```python
def sync_empty_state(self, copy: str = "") -> None:
    next_copy = copy.strip() or EMPTY_TRANSCRIPT_COPY
    if self.empty_state_copy == next_copy:
        return
    self.empty_state_copy = next_copy
    if self.is_mounted and not self._messages:
        self.call_later(self.refresh_messages)
```

In `_message_widgets()`, replace `EMPTY_TRANSCRIPT_COPY` with `self.empty_state_copy`.

Update `to_plain_text()` only if the existing empty state tests expect plain-text export to include the empty copy.

- [ ] **Step 4: Run transcript tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py -q
```

Expected: PASS.

---

### Task 2: Route Setup Copy Through ConsoleSessionSurface and ChatScreen

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_session_surface.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Write failing setup-empty-state UI test**

Add a test near the existing empty transcript assertions:

```python
async def test_console_empty_transcript_names_missing_model_setup_blocker(monkeypatch):
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": ""},
        "console": {
            "rail_state": {
                "console_rail_state:global:global": {
                    "left_open": True,
                    "right_open": False,
                }
            }
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = ""
    monkeypatch.setattr(
        ChatScreen,
        "_effective_console_provider_model",
        lambda self: ("llama_cpp", ""),
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        transcript = await _wait_for_selector(
            console,
            pilot,
            "#console-native-transcript",
        )

        text = _visible_text(transcript)
        assert "Choose a model in Console Settings to start chatting." in text
        assert "Ready. Ask a question" not in text
```

Use the existing helper names in the file. If `_visible_text()` does not exist in that test module, use the local text extractor already used by nearby tests.

Also update nearby existing expectations deliberately:

- Keep ready-state tests such as `test_console_native_transcript_is_visible_transcript_surface` and `test_console_empty_transcript_uses_compact_ready_state` asserting the generic ready copy for non-blocked sessions.
- Update the missing-model test that currently asserts no `Choose model` text so it instead verifies the provider recovery strip stays hidden while the transcript/composer show setup guidance.
- Update the API-key provider blocker test to replace generic ready/composer-placeholder expectations with setup-aware generic copy while preserving the single occurrence of the exact `missing api key` provider detail.

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_internals_decomposition.py::test_console_empty_transcript_names_missing_model_setup_blocker -q
```

Expected: FAIL because the transcript still renders the generic ready copy.

- [ ] **Step 3: Add surface forwarding**

In `ConsoleSessionSurface.sync_inline_guidance()`, keep the title stable, then forward copy to the transcript:

```python
def sync_inline_guidance(self, *, visible: bool, copy: str = "") -> None:
    try:
        title = self.query_one("#console-transcript-title", Static)
    except Exception:
        return
    title.update(CONSOLE_TRANSCRIPT_TITLE)

    try:
        transcript = self.query_one("#console-native-transcript", ConsoleTranscript)
    except Exception:
        return
    transcript.sync_empty_state(copy if visible else "")
```

- [ ] **Step 4: Add ChatScreen setup copy helper**

Add a helper close to `_console_send_blocked_reason()`:

```python
def _console_empty_transcript_copy(self) -> str:
    blocked_reason = self._console_send_blocked_reason()
    if not blocked_reason:
        return ""
    reason = blocked_reason.removeprefix("Console send blocked: ").strip()
    if "Select a model" in reason:
        return "Choose a model in Console Settings to start chatting."
    if "provider" in reason.lower() or "api key" in reason.lower():
        return "Finish provider setup to start chatting."
    return ""
```

Keep the copy concise. Do not include keyboard shortcuts, long instructions, or exact provider recovery details already visible in the provider recovery strip.

- [ ] **Step 5: Feed helper into existing guidance sync**

In `_sync_console_transcript_guidance()`, change the surface call from:

```python
surface.sync_inline_guidance(
    visible=guidance_visible,
    copy=CONSOLE_INLINE_GUIDANCE_COPY,
)
```

to:

```python
empty_copy = self._console_empty_transcript_copy()
surface.sync_inline_guidance(
    visible=bool(empty_copy),
    copy=empty_copy,
)
```

Leave the static `#console-start-here` and `#console-action-hints` behavior unchanged unless tests show they are still visible when they should not be.

- [ ] **Step 6: Run UI tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_internals_decomposition.py::test_console_empty_transcript_names_missing_model_setup_blocker Tests/UI/test_console_session_settings.py -q
```

Expected: PASS.

---

### Task 3: Make the Composer Explain Send-Blocked Setup State

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_internals_decomposition.py`
- Test: `Tests/UI/test_console_session_settings.py`

- [ ] **Step 1: Write failing composer unit/widget test**

Add or extend a composer action-state test:

```python
async def test_console_composer_empty_setup_blocked_state_shows_reason():
    host = ConsoleHarness(_build_test_app())
    async with host.run_test(size=(180, 48)) as pilot:
        composer = await _wait_for_selector(
            host.screen_stack[-1],
            pilot,
            "#console-native-composer",
        )

        composer.sync_action_state(
            has_draft=False,
            run_active=False,
            can_save_chatbook=False,
            send_blocked=True,
            setup_blocked_reason="Choose a model in Console Settings before sending.",
        )
        await pilot.pause()

        visible_draft = composer.query_one("#console-command-visible-text", Static)
        send_button = composer.query_one("#console-send-message", Button)

        assert "Setup required: choose a model in Console Settings." in visible_draft.renderable.plain
        assert send_button.disabled is True
        assert send_button.tooltip == "Choose a model in Console Settings before sending."
```

Add a companion assertion or second focused test for the non-setup blocked path:

```python
composer.sync_action_state(
    has_draft=False,
    run_active=True,
    can_save_chatbook=False,
    send_blocked=True,
    setup_blocked_reason="",
)
await pilot.pause()

visible_draft = composer.query_one("#console-command-visible-text", Static)
send_button = composer.query_one("#console-send-message", Button)

assert "Setup required" not in visible_draft.renderable.plain
assert send_button.tooltip == "Wait for the active Console run to finish before sending."
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_internals_decomposition.py::test_console_composer_empty_setup_blocked_state_shows_reason -q
```

Expected: FAIL because `sync_action_state()` does not accept `setup_blocked_reason`.

- [ ] **Step 3: Extend composer state minimally**

In `ConsoleComposerBar.__init__()`:

```python
self._setup_blocked_reason = ""
```

Extend `_sync_current_action_state()`:

```python
setup_blocked_reason=self._setup_blocked_reason,
```

Extend `sync_action_state()` signature:

```python
setup_blocked_reason: str = "",
```

Normalize it:

```python
reason = setup_blocked_reason.strip()
self._setup_blocked_reason = reason
```

Use it for the Send tooltip:

```python
if send_blocked and reason:
    send_button.tooltip = reason
elif send_blocked:
    send_button.tooltip = "Wait for the active Console run to finish before sending."
```

Do not infer setup state from `send_blocked=True`; active runs also block sending and must retain the existing active-run behavior.

- [ ] **Step 4: Show setup copy in the visible draft only when empty**

Add:

```python
def _placeholder_renderable(self) -> Text:
    if self._send_blocked and self._setup_blocked_reason:
        if "model" in self._setup_blocked_reason.lower():
            return Text("Setup required: choose a model in Console Settings.", style="bold yellow")
        return Text("Setup required: finish provider setup.", style="bold yellow")
    return self._draft_renderable(self.DRAFT_PLACEHOLDER, has_focus=self.has_focus)
```

Keep `_draft_renderable()` as a `@classmethod`. Existing tests call `ConsoleComposerBar._draft_renderable(...)` directly, so do not convert it into an instance method. Add an instance wrapper such as `_placeholder_renderable()` for setup-specific empty copy.

When the draft is empty, render:

```python
renderable = self._placeholder_renderable()
```

For non-empty drafts, continue using the existing classmethod path:

```python
renderable = self._draft_renderable(draft_text, has_focus=self.has_focus)
```

Do not replace typed draft text. Users must still see what they typed even when setup remains blocked.

- [ ] **Step 5: Toggle a CSS class for setup-blocked state**

In `sync_action_state()`:

```python
self.set_class(send_blocked and bool(self._setup_blocked_reason), "console-composer-setup-blocked")
```

In `_agentic_terminal.tcss`:

```css
#console-native-composer.console-composer-setup-blocked #console-command-visible-text {
    color: $ds-status-warning;
}
```

Avoid adding a large new warning panel in this slice.

- [ ] **Step 6: Pass reason from ChatScreen**

In `_sync_console_composer_action_state()`, compute:

```python
setup_blocked_reason = self._console_setup_blocked_reason()
```

Pass it to the composer:

```python
composer.sync_action_state(
    has_draft=bool(composer.draft_text().strip()),
    run_active=run_active,
    can_save_chatbook=can_save_chatbook,
    send_blocked=send_blocked,
    setup_blocked_reason=setup_blocked_reason,
)
```

Implement `_console_setup_blocked_reason()` so it returns a non-empty value only for provider/model setup blockers:

```python
def _console_setup_blocked_reason(self) -> str:
    blocked_reason = self._console_send_blocked_reason()
    reason = blocked_reason.removeprefix("Console send blocked: ").strip()
    if "Select a model" in reason:
        return "Choose a model in Console Settings before sending."
    if "provider" in reason.lower() or "api key" in reason.lower():
        return "Finish provider setup before sending."
    return ""
```

Active-run blocking should pass `setup_blocked_reason=""`.

- [ ] **Step 7: Regenerate CSS**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` updates if the build script rewrites it.

- [ ] **Step 8: Run composer tests and verify GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_session_settings.py -q
```

Expected: PASS.

---

### Task 4: Align Stale Right Rail CSS Widths

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_persistent_rails.py`

- [ ] **Step 1: Write or tighten CSS contract test**

Extend the existing persistent rail CSS test to assert:

```python
def _css_block(css: str, selector: str) -> str:
    start = css.index(selector)
    end = css.index("}", start)
    return css[start:end]

right_handle = _css_block(css, ".console-rail-handle-right")
right_button = _css_block(css, ".console-rail-handle-button-right")

assert ".console-rail-handle-right" in css
assert "width: 11;" in right_handle
assert "min-width: 11;" in right_handle
assert "max-width: 11;" in right_handle
assert ".console-rail-handle-button-right" in css
assert "width: 9;" in right_button
assert "max-width: 9;" in right_button
```

Keep this assertion scoped to selector blocks. A plain `assert "width: 11;" in css` can pass because of unrelated selectors.

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_persistent_rails.py::test_console_rail_css_keeps_handle_geometry_stable -q
```

Expected: FAIL if the test targets the stale source CSS currently declaring right handle/button widths as `9` and `7`.

- [ ] **Step 3: Update TCSS**

In `_agentic_terminal.tcss`, align right handle CSS with the current runtime behavior:

```css
.console-rail-handle-right {
    width: 11;
    min-width: 11;
    max-width: 11;
}

.console-rail-handle-button-right {
    width: 9;
    max-width: 9;
}
```

- [ ] **Step 4: Regenerate CSS and run rail tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_persistent_rails.py -q
```

Expected: PASS.

---

### Task 5: Integration, Screenshot, and Commit

**Files:**
- Verify all changed files.
- Optional local artifact: `/private/tmp/console-main-current.svg.png`

- [ ] **Step 1: Run focused regression suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_persistent_rails.py -q
```

Expected: PASS.

- [ ] **Step 2: Run whitespace check**

Run:

```bash
git diff --check
```

Expected: no output, exit code 0.

- [ ] **Step 3: Regenerate visual QA screenshot**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/console_main_visual_qa.py
qlmanage -t -s 1920 -o /private/tmp /private/tmp/console-main-current.svg
```

Expected screenshot:

- Transcript empty state no longer says "Ready" when setup is blocked.
- Composer empty state names the model/setup blocker.
- Right collapsed inspector still reads `Inspect` and `setup`.
- No new clipping in left rail or composer button row.

- [ ] **Step 4: Review diff**

Run:

```bash
git diff --stat
git diff -- tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_native_transcript.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_persistent_rails.py
```

Expected: only the setup-aware empty/composer state plus CSS-width alignment changed.

- [ ] **Step 5: Commit**

Run:

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/Widgets/Console/console_composer_bar.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_native_transcript.py Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_persistent_rails.py
git commit -m "Clarify console setup-blocked empty state"
```

Expected: one focused commit.

---

## Manual Review Gate

The writing-plans skill normally calls for a plan-document-reviewer subagent. The currently available subagent tool may only be used when the user explicitly asks for subagents or delegation, so this plan uses a manual review gate instead:

- Review this plan for scope creep before implementation.
- Confirm the copy strings are acceptable before coding.
- If subagent review is desired, explicitly approve using a plan-review subagent before implementation.

## Execution Recommendation

Use inline execution for this plan. The change is narrow, touches shared Console widgets, and benefits from one engineer keeping the state flow coherent across transcript, composer, and `ChatScreen`.
