# Roleplay P0 — reframe + honest mode strip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-13-roleplay-p0-reframe-northstar-design.md`. Branch `claude/personas-redesign` off dev `ea88ff95`. Line numbers exact at branch point; grep symbols if they drift.

**Goal:** Reframe the Personas screen's *display* to "Roleplay" with a self-explaining, honest 4-mode strip — no behavior change to the working Characters/Personas modes, and no touching of any file the parallel Library-Prompts branch owns.

**Architecture:** All edits are display-only, inside `tldw_chatbook/UI/Screens/personas_screen.py` (+ its existing UI tests). Task 1 reframes the identity (title base string) and turns the static purpose line into a per-mode descriptor that updates on mode switch. Task 2 makes the mode strip honest (self-explaining chip tooltips reusing Task 1's descriptor copy, a "· soon" marker + per-mode "coming soon" body for the not-yet-built Dictionaries/Lore modes). No CSS changes; no new files.

**Tech Stack:** Python, Textual, pytest + `app.run_test()` (the existing `Tests/UI/test_personas_workbench.py` harness).

## Global Constraints

- **Display-only, Personas-owned.** Edit ONLY `personas_screen.py` (+ its tests). Do NOT touch `screen_registry.py`, `shell_destinations.py`, `route_inventory.py`, the route id `personas`, or `MODE_LABELS`/`MODE_CHIP_ORDER`'s membership.
- **Leave "prompts" alone.** `MODE_CHIP_ORDER` still contains `"prompts"` on this branch; removing it is the parallel branch's Task 7. Do NOT remove it. New copy must be correct whether or not "prompts" is present (use a generic fallback for any mode without an explicit descriptor).
- **Name = "Roleplay"** (never "Studio" — collides with the existing `Prompt_Studio_Interop` "Prompt Studio").
- **Keep the dynamic title suffixes.** `_title_text()` appends ` | New {character|persona}` / ` | Editing {name}` / ` | Ready` and a ` - unsaved` suffix — change ONLY the `base` string.
- **DRY:** the four mode meanings live in ONE dict (`_MODE_DESCRIPTORS`, Task 1), reused by both the purpose descriptor and the chip tooltips.
- **No behavior regression:** Characters/Personas modes still compose, switch, and edit exactly as before.
- Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <target> \
    -q -p no:cacheprovider -o addopts="" --timeout=180 --timeout-method=thread
  ```

---

### Task 1: Identity reframe + per-mode descriptor

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (`_title_text` base `:887`; add `_MODE_DESCRIPTORS` + `_mode_descriptor_text`; `#personas-purpose` compose value `:355`; `_apply_mode` update `~:863`)
- Modify: `Tests/UI/test_personas_workbench.py` (update TWO existing assertions that pin the old copy + add reframe tests)
- Modify: `Tests/UI/test_unified_shell_phase6_first_time_replay.py` (update the personas nav-row expected text fragments — old title/purpose copy)

**Interfaces:**
- Produces: `_MODE_DESCRIPTORS: dict[str, str]` (module-level) and `PersonasScreen._mode_descriptor_text(self, mode: str) -> str` — the one-line meaning of a mode (falls back to `MODE_LABELS.get(mode, mode)` for modes without an explicit entry, e.g. "prompts"). Task 2 reuses these for chip tooltips.

- [ ] **Step 1: Write/adjust the tests (RED)**

First update the existing assertions that pin the old copy — the reframe *will* break them, so they're part of this change:

(a) In `Tests/UI/test_personas_workbench.py`, `TestWorkbenchShell.test_route_renders_destination_workbench` (~`:148`):
```python
            assert "Personas" in str(title.renderable)      # OLD
```
→
```python
            assert "Roleplay" in str(title.renderable)      # NEW
```

(b) In `Tests/UI/test_personas_workbench.py` (~`:4054-4057`), the exact-title assertion:
```python
            assert (
                str(title.renderable)
                == "Personas | Behavior profiles for chat and agents | Ready"      # OLD
            )
```
→
```python
            assert (
                str(title.renderable)
                == "Roleplay | Author the pieces that shape a chat | Ready"         # NEW
            )
```

(c) In `Tests/UI/test_unified_shell_phase6_first_time_replay.py` (~`:157-163`), the `"nav-personas"` row's expected-fragments tuple (which pins the old title + old purpose copy) — replace the fragment tuple:
```python
                    (
                        "Personas",
                        "Behavior profiles for chat and agents",
                        "characters, personas, prompts, dictionaries, and lore",
                        "Attach to Console",
                    ),
```
→ fragments that are actually rendered after the reframe (title tagline + a chip that's always present):
```python
                    (
                        "Roleplay",
                        "Author the pieces that shape a chat",
                        "Characters",
                        "Lore",
                    ),
```

Then add these methods to `class TestWorkbenchShell:` (reusing its `mock_app_instance`/`stub_characters` fixtures + `_mounted`/`PersonasTestApp`):
```python
    async def test_title_reframed_to_roleplay_keeps_state_suffix(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            assert str(screen.query_one("#personas-title", Static).renderable).startswith("Roleplay")
            # dynamic suffix still appends in create mode
            screen._edit_mode = "create"
            screen._update_title()
            await pilot.pause()
            title = str(screen.query_one("#personas-title", Static).renderable)
            assert title.startswith("Roleplay") and "New character" in title

    async def test_purpose_shows_active_mode_descriptor_and_updates_on_switch(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            purpose = screen.query_one("#personas-purpose", Static)
            assert "Who the AI plays" in str(purpose.renderable)   # characters is the default mode
            await screen._apply_mode("personas")
            await pilot.pause()
            assert "Who you are" in str(screen.query_one("#personas-purpose", Static).renderable)
```

- [ ] **Step 2: Run to verify RED**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  "Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_title_reframed_to_roleplay_keeps_state_suffix" \
  "Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_purpose_shows_active_mode_descriptor_and_updates_on_switch" \
  -q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — title still starts with "Personas"; `#personas-purpose` holds the old static "Create and manage…" copy.

- [ ] **Step 3: Reframe the title base**

In `personas_screen.py:887`, change:
```python
        base = "Personas | Behavior profiles for chat and agents"
```
to:
```python
        base = "Roleplay | Author the pieces that shape a chat"
```
(Leave the rest of `_title_text` — the `suffix`/`create`/`edit`/`Ready` branches — untouched.)

- [ ] **Step 4: Add the mode-descriptor source of truth**

Near the other module-level constants (by `MODE_CHIP_ORDER`/`PLACEHOLDER_COPY`, ~`:85`), add:
```python
#: One-line "what this mode is" copy, shown under the title and as chip tooltips.
_MODE_DESCRIPTORS: dict[str, str] = {
    "characters": "Characters — who the AI plays.",
    "personas": "Personas — who you are.",
    "dictionaries": "Dictionaries — text find/replace rules.",
    "lore": "Lore — world facts injected on keywords.",
}
```
And add this method to `PersonasScreen` (next to `_title_text`):
```python
    def _mode_descriptor_text(self, mode: str) -> str:
        """The visible one-line meaning of a mode (falls back for un-described modes)."""
        return _MODE_DESCRIPTORS.get(mode, MODE_LABELS.get(mode, mode))
```

- [ ] **Step 5: Make `#personas-purpose` the per-mode descriptor**

In `compose_content` (`:355-359`), replace the static purpose Static's text so it starts on the active mode's descriptor:
```python
            yield Static(
                self._mode_descriptor_text(self.state.active_mode),
                id="personas-purpose",
                classes="destination-purpose",
            )
```
Then in `_apply_mode`, right after the existing status-row refresh (`self.query_one("#personas-status-row", Static).update(self._status_row_text())`, ~`:863`), add:
```python
        self.query_one("#personas-purpose", Static).update(self._mode_descriptor_text(mode))
```

- [ ] **Step 6: Run to verify GREEN (incl. all updated assertions)**

Run the full personas workbench suite (catches the exact-title assertion at ~:4056 and `test_route_renders_destination_workbench`) plus the phase6 replay test (whose fragments you updated):
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_personas_workbench.py \
  "Tests/UI/test_unified_shell_phase6_first_time_replay.py" \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: all pass (2 new reframe tests + the 3 updated assertions + all pre-existing personas + phase6 tests). If the phase6 test fails on a fragment, the reframed screen simply doesn't render that literal — adjust the fragment to one that is present (e.g. a chip label) rather than reverting the reframe.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat(roleplay): reframe title to Roleplay + per-mode descriptor line (P0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Honest mode strip — self-explaining chips + coming-soon

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (chip loop tooltip + label `:367-379`; `PLACEHOLDER_COPY`→per-mode `_MODE_COMING_SOON` `:87`; `_apply_mode` placeholder text `~:866`)
- Modify: `Tests/UI/test_personas_workbench.py` (add tests)

**Interfaces:**
- Consumes: `_MODE_DESCRIPTORS` / `_mode_descriptor_text` from Task 1 (chip tooltips reuse them).
- Produces: `_MODE_COMING_SOON: dict[str, str]` (module-level) and `PersonasScreen._coming_soon_text(self, mode: str) -> str`.

- [ ] **Step 1: Write/adjust the tests (RED)**

First update the existing placeholder assertion the new copy breaks — in `Tests/UI/test_personas_workbench.py` (~`:362`, the test that clicks a placeholder mode and checks the copy):
```python
            assert "not available yet" in str(placeholder.renderable)      # OLD
```
→
```python
            assert "coming soon" in str(placeholder.renderable).lower()    # NEW
```
(That test currently clicks `#personas-mode-prompts`; leave the mode it clicks alone — `_coming_soon_text("prompts")` returns the generic "This mode is coming soon." fallback, which satisfies the updated assertion. Removing the prompts chip is the parallel branch's job, not P0's.)

Then add to `class TestWorkbenchShell:`:
```python
    async def test_mode_chips_are_self_explaining_and_mark_coming_soon(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            dict_chip = screen.query_one("#personas-mode-dictionaries", Button)
            assert dict_chip.tooltip == "Dictionaries — text find/replace rules."   # meaning, not "switch to…"
            assert "soon" in str(dict_chip.label).lower()                            # planned marker
            char_chip = screen.query_one("#personas-mode-characters", Button)
            assert "soon" not in str(char_chip.label).lower()                        # built modes unmarked

    async def test_coming_soon_mode_shows_inviting_copy(self, mock_app_instance, stub_characters):
        app = PersonasTestApp(mock_app_instance)
        async with app.run_test() as pilot:
            screen = await _mounted(pilot)
            await screen._apply_mode("dictionaries")
            await pilot.pause()
            body = str(screen.query_one("#personas-mode-placeholder", Static).renderable)
            assert "coming soon" in body.lower()
            assert "not available yet" not in body.lower()
```

- [ ] **Step 2: Run to verify RED**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  "Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_mode_chips_are_self_explaining_and_mark_coming_soon" \
  "Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_coming_soon_mode_shows_inviting_copy" \
  -q -p no:cacheprovider -o addopts="" --timeout=120 --timeout-method=thread
```
Expected: FAIL — chip tooltip is the generic "Switch the workbench to …", labels lack "soon", and the placeholder still reads "not available yet".

- [ ] **Step 3: Add the coming-soon source of truth**

Replace the `PLACEHOLDER_COPY` constant (`:87`) with a per-mode map + keep a generic fallback:
```python
#: Inviting "coming soon" body per not-yet-built mode; generic fallback for others.
_MODE_COMING_SOON: dict[str, str] = {
    "dictionaries": "Dictionaries — author text find/replace rules for your chats. Coming soon.",
    "lore": "Lore — build world facts that get injected when keywords appear. Coming soon.",
}
_COMING_SOON_FALLBACK = "This mode is coming soon."
```
And add to `PersonasScreen`:
```python
    def _coming_soon_text(self, mode: str) -> str:
        return _MODE_COMING_SOON.get(mode, _COMING_SOON_FALLBACK)
```
Update the initial placeholder Static in `compose_content` (currently `yield Static(PLACEHOLDER_COPY, id="personas-mode-placeholder")`, `:418`) to a neutral default (it is hidden until a coming-soon mode is selected):
```python
                    yield Static(self._coming_soon_text("dictionaries"), id="personas-mode-placeholder")
```

- [ ] **Step 4: Self-explaining chips + planned marker**

In the mode-chip loop in `compose_content` (`:368-379`), change the `Button` so the label marks planned modes and the tooltip states the meaning:
```python
                for mode in MODE_CHIP_ORDER:
                    classes = "personas-mode-chip"
                    if mode == self.state.active_mode:
                        classes = f"{classes} is-active"
                    label = MODE_LABELS[mode]
                    if mode in _MODE_COMING_SOON:
                        label = f"{label} · soon"
                    yield Button(
                        label,
                        id=f"personas-mode-{mode}",
                        classes=classes,
                        tooltip=self._mode_descriptor_text(mode),
                    )
```

- [ ] **Step 5: Show the per-mode coming-soon body on switch**

In `_apply_mode`, at the branch that shows the placeholder for not-yet-built modes (currently `self._show_center("#personas-mode-placeholder")`, ~`:866`), set its text first:
```python
            self.query_one("#personas-mode-placeholder", Static).update(self._coming_soon_text(mode))
            self._show_center("#personas-mode-placeholder")
```

- [ ] **Step 6: Run to verify GREEN + no regression**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_personas_workbench.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: the whole `test_personas_workbench.py` suite passes (new tests + the reframed one + all pre-existing workbench tests — no regression to the working modes).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_workbench.py
git commit -m "feat(roleplay): self-explaining mode chips + honest per-mode coming-soon (P0)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 2)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_personas_workbench.py Tests/UI/test_personas_workbench_foundation.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Plus `python -c "import tldw_chatbook.app"`. Then the whole-branch review and finishing-a-development-branch. (This is a display-only reframe; a served-TUI visual glance to confirm the header reads "Roleplay" and the coming-soon chips read as planned is worthwhile but optional.)
