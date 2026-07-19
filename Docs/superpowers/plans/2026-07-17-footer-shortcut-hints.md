# Footer Shortcut Hints on Feature Screens — Implementation Plan (task-264)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Registered workbench shortcut hints (Console F6 set, MCP mode set) visibly render on every feature screen, with the footer status system (shortcut strip + word/token/DB stats) alive per-screen.

**Architecture / Decision (AC #3, recorded):** **Per-screen mounting of the existing `AppFooterStatus`, fed by the existing registration API.** The alternative — repairing the single App-level strip — is architecturally impossible under `push_screen` navigation (Textual screens are full-viewport; the App-default-screen widget can never composite under a pushed screen; `App.query_one` even resolves against the default screen by design, which is why every registration "succeeds" invisibly). Making one global strip real would require abandoning screen-stack navigation — the app-owned chrome migration the design contract explicitly defers ("Do not create a second global navigation system"). The design mockups show a per-screen bottom shortcut row, and the CSS uses a type selector (`AppFooterStatus { dock: bottom; height: 1 }`, Constants.py ~1612) so per-screen instances style themselves. `BaseAppScreen`'s current `Footer(show_command_palette=False)` is replaced by the `AppFooterStatus` instance — on every screen except Console that Footer renders nothing today (zero `show=True` bindings), and Console's four visible bindings (Ctrl+K/Alt+M/Alt+V/Ctrl+T) migrate into its registered shortcut set (the bindings stay functional; only the rendering channel changes).

**Tech Stack:** Python 3.11, Textual, pytest with `app.run_test()` harnesses (harnesses lack the app stylesheet — assert on state/text, not geometry).

## Global Constraints

- Backlog ACs (task-264): (1) registered workbench shortcut hints visibly render on every `BaseAppScreen` that registers them; (2) MCP and Console hint sets verified LIVE (coordinator capture task); (3) the mounting-vs-per-screen decision recorded (this plan + the task notes).
- The registration API surface is unchanged: `set_workbench_shortcuts(source=, shortcuts=)` / `clear_shortcut_context(source=)` keep their signatures and ShortcutContext race-guard semantics (Tests/UI/test_app_footer_shortcut_context.py must pass unchanged).
- The App-level default-screen `AppFooterStatus` mount (app.py:4387) stays — it covers the pre-push startup frame; per-screen instances get a DIFFERENT id (`screen-footer-status`) so `#app-footer-status` id-references (if any) stay unambiguous.
- No behavior change for screens that never register shortcuts: they show the widget's default text (`"Ctrl+Q quit | Ctrl+P palette"`) — a strict improvement over today's empty Footer.
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/agent-runtime`, branch `claude/footer-hints-264` (off origin/dev 1bf8c0d3). Tests FOREGROUND via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`; `timeout` unavailable; UI tests may need `python3 tldw_chatbook/css/build_css.py` once if stylesheet errors appear.

---

### Task 1: Per-screen AppFooterStatus + screen-aware callers

**Files:**
- Modify: `tldw_chatbook/UI/Navigation/base_app_screen.py` (compose, line ~100)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_register_console_footer_shortcuts`/`_clear_console_footer_shortcuts` ~984-996; `CONSOLE_WORKBENCH_SHORTCUTS` ~367)
- Modify: `tldw_chatbook/UI/Screens/mcp_screen.py` (~220, ~230)
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (~3498, ~3507)
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py` (~101, ~208)
- Modify: `tldw_chatbook/app.py` (DB-size updater: `_db_size_status_widget` cache at ~5937 + the update path ~5875; second site ~6461)
- Test: `Tests/UI/test_screen_footer_hints.py` (create) + adjust any structure pins in `Tests/UI/test_destination_shells.py` etc. that assert the Textual `Footer` on screens

**Steps:**

- [ ] **Step 1: Write the failing tests** — `Tests/UI/test_screen_footer_hints.py`, following the file-local harness patterns of existing `Tests/UI` screen tests (`app.run_test()`; the repo's AppTest equivalent is unavailable — see other tests in the directory for the working idiom):

```python
async def test_base_app_screen_composes_footer_status():
    """Every BaseAppScreen carries its own AppFooterStatus instance."""
    # mount any lightweight BaseAppScreen subclass (or a minimal test
    # subclass) in a harness app; assert screen.query_one(AppFooterStatus)
    # exists with id "screen-footer-status" and shortcut_text ==
    # AppFooterStatus.DEFAULT_SHORTCUT_TEXT.


async def test_console_registration_updates_the_screens_own_footer():
    """chat_screen's registration must land on ITS instance, not the app's
    default-screen one; text contains 'F6' and 'Ctrl+K'."""


async def test_mcp_registration_updates_the_screens_own_footer():
    """mcp_screen registration -> its footer text contains 'mode' and
    'a add server'."""
```

Write them fully against the real screens where feasible; where a full screen is too heavy for the harness, a minimal `BaseAppScreen` subclass exercising the same `set_workbench_shortcuts` path through the screen-local instance is acceptable for the first test, but the Console/MCP tests must drive the REAL registration methods (`_register_console_footer_shortcuts` / mcp's registration) with the screen mounted.

- [ ] **Step 2: RED** — run the new file; the compose test fails (no AppFooterStatus on screens).

- [ ] **Step 3: Implement**

`base_app_screen.py` compose (~line 100): replace `yield Footer(show_command_palette=False)` with:

```python
        yield AppFooterStatus(id="screen-footer-status")
```

(import it; remove the `Footer` import if now unused).

Caller redirections — each site changes ONLY the lookup expression, keeping the try/except and getattr guards:
- `chat_screen.py` ~984/994 and `mcp_screen.py` ~220/230: `self.app.query_one(AppFooterStatus)` → `self.query_one(AppFooterStatus)` (the screen's own instance).
- `personas_screen.py` ~3498/3507: `self.app.query_one("AppFooterStatus")` → `self.query_one("AppFooterStatus")`.
- `chat_token_events.py` ~101/208: `app.query_one("AppFooterStatus")` → `app.screen.query_one("AppFooterStatus")` (active screen; keep the except-guard so the default screen pre-push doesn't crash).
- `app.py` DB-size path: the cached `_db_size_status_widget` (acquired once from the default screen at ~5937) must not be the only target — change the updater that renders the size string to resolve the ACTIVE screen's instance per tick:

```python
    def _active_footer_status(self) -> Optional[AppFooterStatus]:
        """The visible screen's footer, falling back to the default-screen one."""
        try:
            return self.screen.query_one(AppFooterStatus)
        except QueryError:
            return self._db_size_status_widget
```

and call `update_db_sizes_display` on that (keep the startup acquisition as the fallback cache). Apply the same active-screen resolution at ~6461.

`CONSOLE_WORKBENCH_SHORTCUTS` (~367) gains the two most-used bindings the removed Footer used to render:

```python
CONSOLE_WORKBENCH_SHORTCUTS = (
    ("F6", "next pane"),
    ("Shift+F6", "previous pane"),
    ("F1", "help"),
    ("Enter", "send"),
    ("Ctrl+K", "switch session"),
    ("Ctrl+T", "new tab"),
    ("Ctrl+P", "palette"),
)
```

(Alt+M/Alt+V stay functional but unrendered — alt-key hints were unverifiable over xterm.js anyway per the QA recipe notes.)

- [ ] **Step 4: GREEN + regressions** — run: the new file; `Tests/UI/test_app_footer_shortcut_context.py` (must pass unchanged); `Tests/UI/test_destination_shells.py`, `test_console_workbench_contract.py`, `test_workbench_pane_focus.py`, `test_personas_dictionaries.py` (fix any Footer-structure pins to expect AppFooterStatus — flag in the report exactly which pins changed and why); then `Tests/UI/ -q` broadly and `Tests/Chat/test_console_agent_bridge.py -q`.

- [ ] **Step 5: Commit**

```bash
git add -u tldw_chatbook/ Tests/UI/
git add Tests/UI/test_screen_footer_hints.py
git commit -m "feat(ui): footer shortcut hints render per-screen — AppFooterStatus mounts on every BaseAppScreen (task-264)"
```

### Task 2 (coordinator): live verification (AC #2) + close-out

textual-serve + playwright at 2050×1240 (established recipe): capture Console (F6 hint set incl. Ctrl+K/Ctrl+T) and MCP (mode/add-server/test-tool/permission hints) footers; spot-check Library/Personas default footer text. Evidence dir `Docs/superpowers/qa/footer-hints-2026-07/`. Then backlog close-out (ACs, decision note, Done), review, PR.

## Self-Review

- AC #1 → Task 1 (per-screen instance + screen-aware registration, pinned by three tests). AC #2 → Task 2 live captures. AC #3 → the Decision paragraph above + task notes.
- Registration API unchanged; context race-guard untouched (existing test file passes unchanged).
- Console's visible-binding loss handled by folding Ctrl+K/Ctrl+T into the registered set; every other screen's Footer rendered nothing (verified: zero show=True bindings outside chat_screen).
