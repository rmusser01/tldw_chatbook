# Console Visual Hierarchy (Phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the Console's visual-hierarchy pass: exactly one frame per region, dim/bright counter chips, composer-as-primary (Stop only while streaming), the consistent `▾/▸/●/✓/✕` glyph language, dimmed transcript role labels with an accent-bordered selection, and deletion of the legacy empty-panel shim promised in PR #577.

**Architecture:** A tiny pure glyph-constants module feeds every glyph site. Frame de-doubling removes the CSS-side border so `_frame_console_region` becomes the single border source, with inner trays always quiet. Counter emphasis is pure state (`ConsoleControlState` gains three booleans) rendered as chip classes. All CSS edits go in `_agentic_terminal.tcss` + `./build_css.sh`. Test realignments preserve intent, including the long-stale left-rail priority test which is corrected to the user-approved section order.

**Tech Stack:** Python 3.11+, Textual, Rich Text, pytest + pilot, existing harnesses.

**Spec:** `Docs/superpowers/specs/2026-07-02-console-dual-audience-ux-design.md` §4. Phases 1–3 merged (PRs #576/#577/#579).

## Already delivered by prior phases (do NOT rebuild)

- "One header line": the legacy `#console-title`/`#console-status-row` rows are already `display:none` compat seams; the chip row is one line. Phase 4's header deliverable is ONLY the zero-counter dimming (Task 4).
- Setup-modal steps already use `✓/●/○`; rail sections/prefs/keyboard layer all exist.

## Global Constraints

- Run tests: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q <target> --tb=short`. The `timeout` shell command is unavailable.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is GENERATED: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, run `./build_css.sh`, commit both.
- Glyph language (spec §4 + directional extension): `▾` expanded/expandable-open, `▸` collapsed/expandable-closed AND active/selected marker, `●` in-progress, `✓` done, `✕` close, `◂`/`▸` directional collapse for left/right rails. No `-`/`+`/`>`/`x`/`<` affordances remain in the Console.
- Exactly ONE border per region (left rail, transcript region, composer, right rail): `_frame_console_region` inline border is the single source; `.console-region` contributes NO border; inner trays are always quiet.
- Counter chips: `Sources`/`Tools`/`Approvals` render class `console-chip-dim` when their count is 0 and `console-chip-alert` when > 0. Other chips unchanged.
- Composer: Send is the ONLY `variant="primary"` button; Stop is `display:none` unless a run is active; when send-blocked, Send carries `console-send-blocked` AND `#console-send-disabled-reason` is visible beside it (existing behavior — keep).
- Transcript: role labels render dim, message body full contrast; selected message gets a thick accent left border (keep existing background); the tab strip gets one blank line of separation above the transcript content.
- The stale test `test_console_left_rail_prioritizes_attach_and_active_conversation` (fails on dev since Phase 1: asserts Attach before Conversations) is REALIGNED to the user-approved Session→Context order — the layout does not change. After this plan there is NO expected-failure baseline at all.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`, `git add .`, or `git commit -a`.** Never touch `.claude/settings.local.json`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- Console screen changes require live screenshot QA + explicit user approval before merge (Task 8).
- Branch: `claude/console-visual-phase4` from current `dev`.

## File Structure

- Create `tldw_chatbook/Chat/console_glyphs.py` — pure glyph constants.
- Modify `tldw_chatbook/Widgets/Console/console_rail_section.py`, `console_workspace_context.py`, `console_session_surface.py`, `console_transcript.py`, `console_control_bar.py`, `console_composer_bar.py`.
- Modify `tldw_chatbook/Chat/console_rail_state.py` (handle label constants), `console_display_state.py` (control-state booleans).
- Modify `tldw_chatbook/UI/Screens/chat_screen.py` (collapse buttons, staged-tray variant).
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerate).
- Tests: `Tests/Chat/test_console_display_state.py`, `Tests/UI/test_console_rail_sections.py`, `test_console_internals_decomposition.py`, `test_console_workbench_contract.py`, `test_console_persistent_rails.py`, `test_console_native_chat_flow.py`, `test_console_session_settings.py`.

---

### Task 1: Glyph constants + rail/section toggle glyphs

**Files:**
- Create: `tldw_chatbook/Chat/console_glyphs.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rail_section.py` (`_toggle_label`, line ~76), `tldw_chatbook/Widgets/Console/console_workspace_context.py` (browser section/group toggle labels — grep `"+" if` / `"-"`), `tldw_chatbook/Chat/console_rail_state.py:14-15` (handle labels), `tldw_chatbook/UI/Screens/chat_screen.py` (rail collapse buttons `"<"` ~4831 and `">"` ~5081)
- Test: `Tests/UI/test_console_rail_sections.py`, plus updates to any test asserting `-`/`+`/`<`/`>` labels (grep)

**Interfaces:**
- Produces: module `tldw_chatbook/Chat/console_glyphs.py`:
  ```python
  """Shared Console glyph language (spec §4)."""

  GLYPH_EXPANDED = "▾"
  GLYPH_COLLAPSED = "▸"
  GLYPH_ACTIVE = "▸"
  GLYPH_IN_PROGRESS = "●"
  GLYPH_DONE = "✓"
  GLYPH_CLOSE = "✕"
  GLYPH_COLLAPSE_LEFT = "◂"
  GLYPH_COLLAPSE_RIGHT = "▸"
  ```
  Every later task imports from here — no literal glyphs at call sites.

- [ ] **Step 1: Write the failing tests**

In `Tests/UI/test_console_rail_sections.py`, update the two section-header tests: expected toggle label `"+"` → `"▸"` (collapsed) and `"-"` → `"▾"` (open) — `test_rail_section_header_renders_title_and_toggle` and `test_rail_section_header_sync_open_flips_toggle` (tooltips unchanged). Add:

```python
def test_console_glyph_constants():
    from tldw_chatbook.Chat.console_glyphs import (
        GLYPH_ACTIVE, GLYPH_CLOSE, GLYPH_COLLAPSED, GLYPH_COLLAPSE_LEFT,
        GLYPH_DONE, GLYPH_EXPANDED, GLYPH_IN_PROGRESS,
    )
    assert (GLYPH_EXPANDED, GLYPH_COLLAPSED) == ("▾", "▸")
    assert (GLYPH_ACTIVE, GLYPH_IN_PROGRESS, GLYPH_DONE) == ("▸", "●", "✓")
    assert (GLYPH_CLOSE, GLYPH_COLLAPSE_LEFT) == ("✕", "◂")
```

- [ ] **Step 2: Run to verify failures** — `-k "glyph_constants or rail_section_header"`; expect ImportError + label mismatches.

- [ ] **Step 3: Implement**

Create the module (code above). In `console_rail_section.py`: `_toggle_label` returns `GLYPH_EXPANDED if self.open else GLYPH_COLLAPSED` (import from the new module). In `console_workspace_context.py`: the conversation-browser section toggle (`"+" if section.collapsed else "-"`) and group toggle (`"+" if group.collapsed else "-"`) become `GLYPH_COLLAPSED if … else GLYPH_EXPANDED`. In `console_rail_state.py`: `CONSOLE_RAIL_CONTEXT_LABEL = "Context ▸"` → use `f"Context {GLYPH_COLLAPSED}"`… constants module import is fine there (pure→pure); `CONSOLE_RAIL_INSPECTOR_LABEL = f"{GLYPH_COLLAPSE_LEFT} Inspector"`. In `chat_screen.py`: left collapse button label `"<"` → `GLYPH_COLLAPSE_LEFT`, right collapse `">"` → `GLYPH_COLLAPSE_RIGHT`.

- [ ] **Step 4: Run** the updated tests + `Tests/UI/test_console_persistent_rails.py` in full (handle-label and width assertions live there — update any `"Context >"`/`"< Inspector"`/`"<"` expectations to the new glyphs, intent preserved; handle-width assertions must still hold since glyph strings are same length).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_glyphs.py tldw_chatbook/Widgets/Console/console_rail_section.py tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/Chat/console_rail_state.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_persistent_rails.py
git commit -m "feat(console): shared glyph language for toggles, handles, collapse"
```

---

### Task 2: Active markers and tab close glyphs

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py` (row marker `"> "` at both render sites — grep `marker = `), `tldw_chatbook/Widgets/Console/console_session_surface.py:132` (close button `"x"`), `tldw_chatbook/Widgets/Console/console_session_switcher_modal.py` (if any `>` markers — grep; the active-result class is styling-only, likely none)
- Test: `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_workbench_contract.py` (grep `"> "` / `"x"` expectations)

**Interfaces:**
- Consumes: Task 1's `GLYPH_ACTIVE`, `GLYPH_CLOSE`.

- [ ] **Step 1: Failing tests** — grep the two test files for assertions on `"> "` row markers (e.g. rail-row text checks like `row_text.startswith("> ")`) and the close button label `"x"`; update expectations to `"▸ "` and `"✕"`. Run to see them fail against current code… they will PASS against current code and FAIL after the change — so flip the order: make the production change first is NOT the TDD house style; instead write one NEW test first:

```python
def test_console_active_row_marker_and_close_glyphs():
    from tldw_chatbook.Chat.console_glyphs import GLYPH_ACTIVE, GLYPH_CLOSE
    from tldw_chatbook.Widgets.Console import console_workspace_context, console_session_surface
    import inspect
    assert '"> "' not in inspect.getsource(console_workspace_context)
    assert '"x"' not in inspect.getsource(console_session_surface)
```

(Source-scan test: crude but deterministic RED first; the real behavior checks are the updated pilot expectations.) Place in `Tests/UI/test_console_rail_sections.py`.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — `marker = f"{GLYPH_ACTIVE} " if row.selected else "  "` at both `console_workspace_context.py` marker sites; close button label `GLYPH_CLOSE` in `console_session_surface.py` (`_build_close_tab_button`). Update every test expectation asserting the old glyphs (grep `'"> '` and `'"x"'`/`str(...) == "x"` across Tests/UI — includes the phase-2 auto-title rail-row tests asserting `row_text.startswith("> ")`).

- [ ] **Step 4: Run** the touched test files fully (`test_console_native_chat_flow.py`, `test_console_workbench_contract.py`, `test_console_rail_sections.py`).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/Widgets/Console/console_session_surface.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_workbench_contract.py
git commit -m "feat(console): active-row and tab-close glyphs"
```

---

### Task 3: Single frame per region + stale rail-test realignment

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (`.console-region` border rule, lines ~372-411) + regenerate `tldw_cli_modular.tcss`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — `_staged_context_frame_variant` (line ~4531) returns `"quiet"` unconditionally (rail frame is the single border; delete the now-unused solid branch if nothing else uses it — grep `_staged_context_frame_variant` first)
- Test: `Tests/UI/test_console_persistent_rails.py` (stylesheet test), `Tests/UI/test_console_workbench_contract.py` (realign the stale test)

**Interfaces:**
- Consumes: nothing new. Produces: no border on `.console-region` in either CSS file.

- [ ] **Step 1: Failing stylesheet test** — extend the generated-stylesheet test family in `Tests/UI/test_console_persistent_rails.py`:

```python
def test_generated_console_stylesheet_single_frame_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    for css_path in (
        root / "components" / "_agentic_terminal.tcss",
        root / "tldw_cli_modular.tcss",
    ):
        css = css_path.read_text()
        block = _css_block(css, ".console-region")
        assert "border:" not in block, css_path.name
```

(`_css_block` helper already exists in this file.)

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — remove the `border: solid $ds-grid-line;` line from `.console-region` in `_agentic_terminal.tcss` (keep the rest of the block); `./build_css.sh`. Change `_staged_context_frame_variant` to return `"quiet"` always, updating its docstring ("the rail frame is the single border source"). Realign the stale test `test_console_left_rail_prioritizes_attach_and_active_conversation` (test_console_workbench_contract.py:275-292): rename to `test_console_left_rail_orders_session_then_staged_context`, flip the index assertion to `visible_text.index("Conversations") < visible_text.index("Attach")` (the user-approved Session→Context order), keep the `"No sources attached."` and `"Chat 1"` assertions unchanged.

- [ ] **Step 4: Run** `test_console_persistent_rails.py` + `test_console_workbench_contract.py` in full — with the realignment, expect ZERO failures (this removes the last documented baseline failure). Also run `test_console_internals_decomposition.py -k "width or frame or rail_section"` for geometry fallout; fix any test asserting the old double-border geometry (widths may shift by the removed border cells — preserve each test's intent).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_persistent_rails.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): single frame per region, realign stale rail-priority test"
```

---

### Task 4: Dim/bright counter chips

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (`ConsoleControlState`, lines 248-283), `tldw_chatbook/Widgets/Console/console_control_bar.py` (`_chip` + chip row, lines ~160-250), `_agentic_terminal.tcss` (+ regenerate)
- Test: `Tests/Chat/test_console_display_state.py`, `Tests/UI/test_console_workbench_contract.py`

**Interfaces:**
- Produces: `ConsoleControlState` gains `sources_active: bool = False`, `tools_active: bool = False`, `approvals_active: bool = False` (set in `from_values` from `staged_source_count > 0` etc.). `ConsoleControlBar._chip` gains keyword `emphasis: bool | None = None` — `None` renders as today; `False` adds class `console-chip-dim`; `True` adds `console-chip-alert`. The three counter chips pass their boolean; others pass `None`.

- [ ] **Step 1: Failing tests**

`Tests/Chat/test_console_display_state.py`:

```python
def test_console_control_state_counter_activity_flags():
    from tldw_chatbook.Chat.console_display_state import ConsoleControlState
    idle = ConsoleControlState.from_values()
    assert (idle.sources_active, idle.tools_active, idle.approvals_active) == (False, False, False)
    busy = ConsoleControlState.from_values(staged_source_count=2, tool_count=1, approval_count=3)
    assert (busy.sources_active, busy.tools_active, busy.approvals_active) == (True, True, True)
```

`Tests/UI/test_console_workbench_contract.py` (pilot; reuse the file's harness):

```python
@pytest.mark.asyncio
async def test_console_counter_chips_dim_when_zero():
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-sources-chip")
        for chip_id in ("#console-sources-chip", "#console-tools-chip", "#console-approvals-chip"):
            chip = console.query_one(chip_id)
            assert chip.has_class("console-chip-dim"), chip_id
            assert not chip.has_class("console-chip-alert"), chip_id
        assert not console.query_one("#console-provider-chip").has_class("console-chip-dim")
```

- [ ] **Step 2: Run to verify failures.**

- [ ] **Step 3: Implement** — dataclass fields + `from_values` booleans; `_chip(…, emphasis=…)` applying the classes; `sync_state` (or the bar's refresh path — read how chips update on state change and apply classes there too, not only at compose); CSS in `_agentic_terminal.tcss` next to `.console-control-chip`:

```css
.console-chip-dim {
    color: $ds-text-muted;
}

.console-chip-alert {
    color: $ds-text-primary;
    text-style: bold;
}
```

`./build_css.sh`. Add both selectors to the generated-stylesheet presence test family.

- [ ] **Step 4: Run** both test files fully.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_display_state.py tldw_chatbook/Widgets/Console/console_control_bar.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/Chat/test_console_display_state.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_persistent_rails.py
git commit -m "feat(console): counter chips dim at zero, brighten when active"
```

---

### Task 5: Stop only while streaming

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py` (`sync_send_state` lines ~221-293, compose ~1192)
- Test: `Tests/UI/test_console_internals_decomposition.py`

**Interfaces:**
- Consumes: existing `sync_send_state(has_draft, run_active, send_blocked, setup_blocked_reason)` and classes `console-stop-active`/`console-stop-idle`.
- Produces: `#console-stop-generation` has `display: none` when `run_active` is False and is shown when True (set `styles.display` in `sync_send_state`; compose starts hidden). Class handling otherwise unchanged. Send stays the only primary; the blocked-reason Static behavior is untouched.

- [ ] **Step 1: Failing pilot test**

```python
@pytest.mark.asyncio
async def test_console_stop_button_hidden_unless_streaming():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        stop = console.query_one("#console-stop-generation")
        assert stop.styles.display == "none"
        composer = console.query_one("#console-native-composer")
        composer.sync_send_state(has_draft=True, run_active=True)
        await pilot.pause()
        assert stop.styles.display != "none"
        composer.sync_send_state(has_draft=False, run_active=False)
        await pilot.pause()
        assert stop.styles.display == "none"
```

(If `#console-native-composer` isn't the ConsoleComposerBar itself, query the bar type — mirror neighboring composer tests' access pattern.)

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — in compose, after creating the Stop button: `stop.styles.display = "none"`. In `sync_send_state`, alongside the existing stop class toggling: `stop_button.styles.display = "block" if run_active else "none"`.

- [ ] **Step 4: Run** the new test + `test_console_internals_decomposition.py -k "composer or stop or send"`; update any test asserting Stop is visible while idle (preserve intent: tests about Stop behavior during runs still assert visibility under `run_active=True`).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py Tests/UI/test_console_internals_decomposition.py
git commit -m "feat(console): stop button only visible while a run is active"
```

---

### Task 6: Transcript role dimming, selection accent, tab-strip separation

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (message rendering, lines ~62-120), `_agentic_terminal.tcss` (+ regenerate), `tldw_chatbook/Widgets/Console/console_session_surface.py` (tab strip spacing)
- Test: `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_persistent_rails.py` (stylesheet)

**Interfaces:**
- Consumes: existing `_message_role_label` and message row rendering (plain strings today).
- Produces: message rows render as Rich `Text` with the role label styled `"dim"` and the body unstyled (full contrast); plain-text content (`.plain`) is UNCHANGED so every existing text assertion keeps passing. Selected message CSS gains `border-left: thick $ds-accent;` (keep existing background; check the tokens file for the accent token name — grep `accent` in `_agentic_terminal.tcss` and use the file's convention, e.g. `$ds-accent` or the nearest equivalent). Tab strip container gets `margin-bottom: 1` (one blank line between the strip and transcript content) via CSS class rule (`#console-native-tab-strip`), not inline.

- [ ] **Step 1: Failing tests**

In `Tests/UI/test_console_native_chat_flow.py` (reuse the direct store-append arrange from the sibling message tests):

```python
@pytest.mark.asyncio
async def test_transcript_role_label_renders_dim_body_full_contrast():
    # arrange: same store.append_message + sync pattern as
    # test_console_selected_message_copy_action_uses_app_clipboard
    ...
    row = console.query_one(f"#console-transcript-message-{message.id}", Static)
    rendered = row.renderable
    # Rich Text with spans: the role prefix span carries "dim"
    assert rendered.plain.startswith("Assistant")
    assert any("dim" in str(span.style) for span in rendered.spans), rendered.spans
```

(Replace `...` with the named test's arrange verbatim; adjust the row-id query to the actual message-row id scheme — grep `console-transcript-message` for the exact format.) Stylesheet additions test: extend the presence family with `.console-transcript-message-selected` containing `border-left:` and `#console-native-tab-strip` containing `margin-bottom: 1`.

- [ ] **Step 2: Run to verify failures.**

- [ ] **Step 3: Implement** — in `console_transcript.py`, build the row renderable as `Text`: role portion `Text(role_label, style="dim")` + separator + body appended unstyled (both the single-line `f"{role_label}  {body}"` and multi-line `f"{role_label}\n{body}"` branches — construct with `Text.assemble((role_label, "dim"), "  ", body)` / `Text.assemble((role_label, "dim"), "\n", body)`); ensure `markup=False`-equivalent safety (Text is literal — good). CSS: add `border-left: thick $ds-accent;` (or the file's accent token) inside `.console-transcript-message-selected`; add `#console-native-tab-strip { margin-bottom: 1; }` if no such rule exists (or extend the existing one). `./build_css.sh`.

- [ ] **Step 4: Run** the new tests + `test_console_native_chat_flow.py` in full (plain-text parity means expectations keep passing) + the stylesheet test.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/Widgets/Console/console_session_surface.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_persistent_rails.py
git commit -m "feat(console): dim role labels, accent selection border, tab-strip breathing room"
```

---

### Task 7: Delete the legacy empty-panel shim

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (lines ~260-305: remove `#console-empty-title`, `#console-empty-action-row` + the three `ConsoleTranscriptEmptyAction` children `#console-empty-choose-model`/`-attach-context`/`-run-library-rag`; **KEEP `#console-empty-body`** — it is the live quiet/ready line; delete `ConsoleTranscriptEmptyAction` itself if nothing else uses it — grep first)
- Test re-anchoring: `Tests/UI/test_console_workbench_contract.py` (~115-146), `Tests/UI/test_console_rail_sections.py`, `Tests/UI/test_console_session_settings.py`, `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_internals_decomposition.py` — every reference to the deleted ids (grep `console-empty-` across Tests/)

**Interfaces:**
- Consumes: the live equivalents — the setup modal's `#console-setup-modal-action` (recovery routing) and the control-bar actions (attach-context / run-library-rag).
- Produces: no `console-empty-title`/`console-empty-action-row`/`console-empty-choose-model`/`console-empty-attach-context`/`console-empty-run-library-rag` anywhere in production code; `#console-empty-body` unchanged.

- [ ] **Step 1: Failing test** — add to `Tests/UI/test_console_rail_sections.py`:

```python
@pytest.mark.asyncio
async def test_empty_panel_has_no_legacy_shim_widgets():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        assert not list(app.query("#console-empty-title"))
        assert not list(app.query("#console-empty-action-row"))
        assert not list(app.query("#console-empty-choose-model"))
        assert list(app.query("#console-empty-body"))
```

- [ ] **Step 2: Run to verify failure** (title/action-row currently mounted hidden).

- [ ] **Step 3: Implement** — delete the hidden yields and the `ConsoleTranscriptEmptyAction` class if orphaned. Then re-anchor every test that used the deleted ids, preserving each test's concern:
  - Recovery-action assertions (`#console-empty-choose-model` label/tooltip/click-navigation in workbench_contract, session_settings, native_chat_flow, internals_decomposition) → `#console-setup-modal-action` (same adaptive label source; navigation tests press the modal action in the blocked state).
  - Hidden-state assertions (tests asserting the shim is hidden) → assert ABSENT.
  - Any `_wait_for_selector(..., "#console-empty-choose-model")` used as a mount signal → wait for `#console-setup-modal` (blocked flows) or `#console-transcript-empty-state` (ready flows) per each test's scenario.

- [ ] **Step 4: Run** all five touched test files in full. Expect zero failures (the last baseline failure died in Task 3).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py Tests/UI/test_console_rail_sections.py Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py
git commit -m "refactor(console): delete legacy empty-panel shim, re-anchor tests on live controls"
```

---

### Task 8: Verification, screenshot QA, approval gate

**Files:**
- Create: screenshots under `Docs/superpowers/qa/console-visual-2026-07/`

- [ ] **Step 1: Full affected run**

```bash
env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share .venv/bin/python -m pytest -q \
  Tests/Chat/test_console_display_state.py Tests/UI/test_console_rail_sections.py \
  Tests/UI/test_console_internals_decomposition.py Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_persistent_rails.py \
  Tests/UI/test_console_session_settings.py --tb=short
```

Expected: ALL PASS — zero failures, no baseline exceptions anymore.

- [ ] **Step 2: Live screenshot QA** — proven recipe (bundled chromium, `.intro-dialog` wait, route-abort external, fresh + ready-seeded HOMEs, kill stale app processes; driver `/private/tmp/tldw-console-rail-ia-cdp-20260702/cap.py`, copy if missing). Capture:
  1. Ready Console overview: single frames (no doubled rail rules), glyphed section toggles (`▾`/`▸`), `▸` active row marker, `✕` tab close, dimmed zero-counter chips, one-line breathing room under the tab strip.
  2. Same view with a staged source / pending approval seeded if cheaply possible (bright `console-chip-alert` chips); otherwise document chips-bright as pilot-covered.
  3. Transcript after a real send: dim role labels vs full-contrast body; a selected message showing the accent border; Stop ABSENT while idle.
  4. During a streaming run (send a long prompt to llama.cpp if live and capture mid-stream): Stop visible.

- [ ] **Step 3: User approval gate** — present captures; no merge without explicit approval.

- [ ] **Step 4: Commit QA artifacts**

```bash
git add Docs/superpowers/qa/console-visual-2026-07/
git commit -m "docs(console): visual hierarchy phase 4 QA evidence"
```
