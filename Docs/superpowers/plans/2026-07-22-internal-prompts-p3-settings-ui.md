# Internal Prompts P3 — Settings Editor UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the "Internal Prompts" Settings page — browse the 29 registry-backed internal prompts grouped by subsystem, see customized / default-changed badges, edit any prompt in a modal with a live render preview, and reset to the shipped default.

**Architecture:** A pure `Internal_Prompts/authoring.py` API (group/hash/state/save/reset over `CATALOG` + config helpers, no Textual) is consumed by a self-contained `InternalPromptsPanel(Vertical)` widget (search + subsystem-grouped rows + badges + targeted-filter + worker-backed persistence) which pushes an `InternalPromptEditorModal(ModalScreen)` (contract callout, placeholder chips, TextArea, live preview, Save/Reset/Cancel). The panel is wired into the `settings_screen.py` monolith as a new `INTERNAL_PROMPTS` category in the Expert nav group, following the Theme-editor self-contained pattern.

**Tech Stack:** Python ≥3.11, Textual, pytest; registry API from P1/P2 (`CATALOG`, `PromptSpec`, `safe_substitute`, `get_internal_prompt`); config helpers `get_cli_setting` / `save_settings_to_cli_config` / `delete_settings_from_cli_config`.

Spec: `Docs/superpowers/specs/2026-07-22-internal-prompts-p3-settings-ui-design.md` (P3), implementing §4 of `Docs/superpowers/specs/2026-07-21-internal-prompts-settings-page-design.md`.

## Global Constraints

- **Worktree/branch:** work in an isolated worktree on branch `feat/internal-prompts-p3-settings-ui` off `origin/dev` (currently `68992ef3e`, has P1+P2 merged). venv-only pytest: `.venv/bin/python -m pytest` from the worktree root. No `timeout` command. Never broad-kill pytest.
- **`.superpowers/sdd/progress.md` is a TRACKED file** — NEVER stage/commit it; before any rebase, copy-aside → `git checkout --` → restore.
- **Never raise for user-caused problems** carries into the UI: authoring `save_override`/`reset_override` return `bool` (never raise); the panel's persistence runs in `@work(thread=True, exclusive=True, group="internal-prompt-save")` whose body wraps the call in `try/except Exception` and marshals results via `self.app.call_from_thread(...)` — Textual `@work` defaults `exit_on_error=True`, so an uncaught worker exception crashes the app.
- **No recompose / no list rebuild in the interaction path** (search keystrokes, post-save badge refresh) — targeted `display` toggles and single-row updates only (the task-284 perf class).
- **Preview uses `safe_substitute` (never `render_internal_prompt`)** — can't raise, leaves JSON/`{{ .Prompt }}` braces literal, never trips the render-time required-placeholder warning. Preview shown ONLY when `spec.required_placeholders` is non-empty.
- **Reset deletes the override table AND a customized legacy key** (its value differs from the shipped default) — never delete an uncustomized shipped-value legacy key (e.g. first-run `[prompts.document_generation.*].prompt`).
- **CSS** goes in `tldw_chatbook/css/components/_agentic_terminal.tcss` (where the destination-native `settings-*` classes live — NOT `_workbench.tcss`/`_tools-settings.tcss`). NEVER hand-edit the generated bundle `css/tldw_cli_modular.tcss`; the app rebuilds at start when a partial is newer.
- **`authoring.py` imports config lazily** (call-time, like `resolver.py`) so `Tests/Internal_Prompts/test_import_hygiene.py` stays green.
- **Merge gate:** the live-TUI screenshot QA (Task 7) must be approved by the user before any merge.
- Commit after every task; trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; stage only files the task names.

---

### Task 1: Authoring API (`Internal_Prompts/authoring.py`)

**Files:**
- Create: `tldw_chatbook/Internal_Prompts/authoring.py`
- Test: `Tests/Internal_Prompts/test_authoring.py`

**Interfaces:**
- Consumes: `CATALOG`, `PromptSpec` (`.catalog`); `get_internal_prompt` (`.resolver`); lazy `tldw_chatbook.config.{get_cli_setting, save_settings_to_cli_config, delete_settings_from_cli_config, DEFAULT_CONFIG_FROM_TOML}`.
- Produces (Tasks 3-4 consume these exact signatures):
  - `@dataclass(frozen=True) class OverrideState: customized: bool; default_changed: bool; has_override_table: bool; active_text: str`
  - `iter_specs_by_subsystem() -> list[tuple[str, list[PromptSpec]]]`
  - `baseline_hash(text: str) -> str`
  - `override_state(prompt_id: str) -> OverrideState`
  - `save_override(prompt_id: str, text: str) -> bool`
  - `reset_override(prompt_id: str) -> bool`
  - `customized_count() -> int`

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Internal_Prompts/test_authoring.py
"""Authoring API: grouping, baseline hashing, override state, save/reset.
Real config round-trip via the scratch_config fixture (Tests/Internal_Prompts/
conftest.py) — no mocks."""

import pytest

from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Internal_Prompts.catalog import CATALOG


def test_iter_groups_cover_all_specs_stable_order():
    groups = authoring.iter_specs_by_subsystem()
    subsystems = [name for name, _ in groups]
    # every subsystem appears once, first-seen (registration) order
    assert subsystems == list(dict.fromkeys(s.subsystem for s in CATALOG.values()))
    flat = [spec.id for _, specs in groups for spec in specs]
    assert sorted(flat) == sorted(CATALOG.keys())
    # specs within a group sorted by title
    for _, specs in groups:
        assert [s.title for s in specs] == sorted(s.title for s in specs)


def test_baseline_hash_stable_and_short():
    h = authoring.baseline_hash("hello world")
    assert h == authoring.baseline_hash("hello world")
    assert h != authoring.baseline_hash("hello worlD")
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)


def test_state_default_when_no_override(scratch_config):
    scratch_config("")
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is False
    assert st.default_changed is False
    assert st.has_override_table is False
    assert st.active_text == CATALOG["agents.subagent_system"].default


def test_save_makes_customized_with_fresh_baseline(scratch_config):
    scratch_config("")
    assert authoring.save_override("agents.subagent_system", "CUSTOM") is True
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is True
    assert st.has_override_table is True
    assert st.default_changed is False  # baseline written == current default hash
    assert st.active_text == "CUSTOM"


def test_default_changed_when_baseline_stale(scratch_config):
    scratch_config(
        "[internal_prompts.agents]\n"
        'subagent_system = { text = "CUSTOM", baseline = "deadbeef0000" }\n'
    )
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is True
    assert st.default_changed is True  # stored baseline != hash(current default)


def test_reset_removes_override_returns_to_default(scratch_config):
    scratch_config("")
    authoring.save_override("agents.subagent_system", "CUSTOM")
    assert authoring.reset_override("agents.subagent_system") is True
    st = authoring.override_state("agents.subagent_system")
    assert st.customized is False
    assert st.active_text == CATALOG["agents.subagent_system"].default


def test_reset_deletes_customized_legacy_key(scratch_config):
    # rolling_summarize_system has legacy_config_path chunking_config.summarize_system_prompt
    scratch_config(
        "[chunking_config]\n"
        'summarize_system_prompt = "MY CUSTOM ROLLING PROMPT {none}"\n'
    )
    pid = "summarization.rolling_summarize_system"
    assert authoring.override_state(pid).customized is True
    assert authoring.reset_override(pid) is True
    from tldw_chatbook.Internal_Prompts import get_internal_prompt
    assert get_internal_prompt(pid) == CATALOG[pid].default


def test_reset_leaves_uncustomized_shipped_legacy_key(scratch_config, monkeypatch):
    # A doc-gen user prompt whose legacy [prompts.document_generation.*].prompt
    # equals the shipped default must NOT have that key deleted on reset.
    from tldw_chatbook import config as config_mod
    pid = "document_generation.timeline_user"
    shipped = config_mod.DEFAULT_CONFIG_FROM_TOML["prompts"]["document_generation"]["timeline"]["prompt"]
    scratch_config(
        "[prompts.document_generation.timeline]\n"
        f'prompt = {shipped!r}\ntemperature = 0.3\n'
    )
    assert authoring.override_state(pid).customized is False  # equals shipped -> not customized
    assert authoring.reset_override(pid) is True
    # the legacy key survives (temperature sibling proves the table is intact)
    tbl = config_mod.get_cli_setting("prompts.document_generation", "timeline", None)
    assert tbl is not None and tbl.get("temperature") == 0.3
    assert tbl.get("prompt") == shipped


def test_customized_count(scratch_config):
    scratch_config("")
    assert authoring.customized_count() == 0
    authoring.save_override("agents.subagent_system", "A")
    authoring.save_override("agents.console_agent_operating", "B")
    assert authoring.customized_count() == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Internal_Prompts/test_authoring.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'authoring'`.

- [ ] **Step 3: Write the implementation**

```python
# tldw_chatbook/Internal_Prompts/authoring.py
"""Read/write authoring helpers for the Settings "Internal Prompts" page.

Pure functions over CATALOG + config helpers. No Textual imports. Config
helpers are imported lazily (call-time) so the package stays off the
cold-start import chain (import-hygiene test).

Override storage: sparse ``[internal_prompts.<subsystem>.<key>]`` tables of
``{text, baseline}``. ``baseline`` is a fingerprint of the shipped default at
save time; the resolver ignores it, the UI uses it to flag "default changed".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .catalog import CATALOG, PromptSpec
from .resolver import get_internal_prompt


@dataclass(frozen=True)
class OverrideState:
    customized: bool          # resolved text != shipped default (override OR legacy)
    default_changed: bool     # override table exists AND its baseline != current default hash
    has_override_table: bool  # a [internal_prompts.<sub>.<key>] table is present
    active_text: str          # currently resolved text (editor prefill)


def baseline_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def iter_specs_by_subsystem() -> list[tuple[str, list[PromptSpec]]]:
    order: list[str] = list(dict.fromkeys(s.subsystem for s in CATALOG.values()))
    grouped: dict[str, list[PromptSpec]] = {name: [] for name in order}
    for spec in CATALOG.values():
        grouped[spec.subsystem].append(spec)
    return [
        (name, sorted(grouped[name], key=lambda s: s.title)) for name in order
    ]


def _split(prompt_id: str) -> tuple[str, str]:
    subsystem, _, key = prompt_id.partition(".")
    return subsystem, key


def _override_table(prompt_id: str) -> dict | None:
    from tldw_chatbook.config import get_cli_setting  # lazy
    subsystem, key = _split(prompt_id)
    raw = get_cli_setting("internal_prompts." + subsystem, key, None)
    return raw if isinstance(raw, dict) else None


def override_state(prompt_id: str) -> OverrideState:
    spec = CATALOG[prompt_id]
    active = get_internal_prompt(prompt_id)
    customized = active != spec.default
    table = _override_table(prompt_id)
    has_table = table is not None
    default_changed = bool(
        has_table and table.get("baseline") != baseline_hash(spec.default)
    )
    return OverrideState(
        customized=customized,
        default_changed=default_changed,
        has_override_table=has_table,
        active_text=active,
    )


def save_override(prompt_id: str, text: str) -> bool:
    from tldw_chatbook.config import save_settings_to_cli_config  # lazy
    spec = CATALOG[prompt_id]
    subsystem, key = _split(prompt_id)
    return save_settings_to_cli_config(
        {
            "internal_prompts." + subsystem: {
                key: {"text": text, "baseline": baseline_hash(spec.default)}
            }
        }
    )


def _legacy_differs_from_shipped(spec: PromptSpec) -> tuple[str, str] | None:
    """Return (section, key) of the legacy config path IF the user's current
    value there differs from the shipped default (i.e. a real customization).
    None when no legacy path, or the value is absent/equal-to-shipped."""
    if not spec.legacy_config_path:
        return None
    from tldw_chatbook.config import (  # lazy
        get_cli_setting,
        DEFAULT_CONFIG_FROM_TOML,
    )
    section, _, key = spec.legacy_config_path.rpartition(".")
    if not section:
        return None
    current = get_cli_setting(section, key, None)
    if current is None:
        return None
    node: object = DEFAULT_CONFIG_FROM_TOML
    for part in spec.legacy_config_path.split("."):
        if not isinstance(node, dict) or part not in node:
            node = None
            break
        node = node[part]
    shipped = node
    if current == shipped:
        return None
    return section, key


def reset_override(prompt_id: str) -> bool:
    from tldw_chatbook.config import delete_settings_from_cli_config  # lazy
    spec = CATALOG[prompt_id]
    subsystem, key = _split(prompt_id)
    ok = delete_settings_from_cli_config("internal_prompts." + subsystem, [key])
    legacy = _legacy_differs_from_shipped(spec)
    if legacy is not None:
        section, legacy_key = legacy
        ok = delete_settings_from_cli_config(section, [legacy_key]) and ok
    return ok


def customized_count() -> int:
    return sum(1 for pid in CATALOG if override_state(pid).customized)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest Tests/Internal_Prompts/test_authoring.py -v`
Expected: all pass. If `save_override`'s write shape doesn't resolve back, re-read the design doc's "Verification performed at design time" section — the round-trip was proven, so a failure means a transcription error in the section/key split.

- [ ] **Step 5: Confirm import hygiene still holds**

Run: `.venv/bin/python -m pytest Tests/Internal_Prompts/test_import_hygiene.py -q`
Expected: pass (authoring imports config lazily; nothing new at module import).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Internal_Prompts/authoring.py Tests/Internal_Prompts/test_authoring.py
git commit -m "feat(internal-prompts): P3 authoring API (group/hash/state/save/reset)"
```

---

### Task 2: Editor modal (`InternalPromptEditorModal`)

**Files:**
- Create: `tldw_chatbook/Widgets/settings_internal_prompts_editor_modal.py`
- Test: `Tests/UI/test_internal_prompts_editor_modal.py`

**Interfaces:**
- Consumes: `PromptSpec` (`Internal_Prompts.catalog`), `safe_substitute` (`Internal_Prompts`).
- Produces: `InternalPromptEditorModal(ModalScreen[Optional[dict]])` — constructed `InternalPromptEditorModal(spec=<PromptSpec>, active_text=<str>)`; dismisses with `{"action": "save", "text": str}`, `{"action": "reset"}`, or `None` (cancel). Task 3 pushes it.

Pattern to copy: `tldw_chatbook/Widgets/Console/console_system_prompt_modal.py` (`ModalScreen[Optional[str]]`, `compose()` under a `Vertical`, `TextArea`, `@on(Button.Pressed, "#id")` handlers, `dismiss(value)`, `on_mount` focus, escape binding).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_internal_prompts_editor_modal.py
"""Editor modal: dismiss values, Save validation, preview presence."""

import pytest
from textual.app import App

from tldw_chatbook.Internal_Prompts.catalog import CATALOG
from tldw_chatbook.Widgets.settings_internal_prompts_editor_modal import (
    InternalPromptEditorModal,
)


class _Host(App):
    def __init__(self, spec, active):
        super().__init__()
        self._spec, self._active = spec, active
        self.result = "UNSET"

    def on_mount(self):
        def cb(value):
            self.result = value
        self.push_screen(
            InternalPromptEditorModal(spec=self._spec, active_text=self._active),
            cb,
        )


@pytest.mark.asyncio
async def test_cancel_returns_none():
    spec = CATALOG["agents.subagent_system"]
    app = _Host(spec, spec.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
    assert app.result is None


@pytest.mark.asyncio
async def test_save_returns_action_and_text():
    spec = CATALOG["agents.subagent_system"]
    app = _Host(spec, spec.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#internal-prompt-editor-text").text = "EDITED TEXT"
        await modal._save_from_test()  # helper invokes the same path as the Save button
        await pilot.pause()
    assert app.result == {"action": "save", "text": "EDITED TEXT"}


@pytest.mark.asyncio
async def test_save_blocks_on_missing_required_placeholder():
    spec = CATALOG["rag_reranker.pointwise_template"]  # has required placeholders
    app = _Host(spec, spec.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        modal = app.screen
        modal.query_one("#internal-prompt-editor-text").text = "no tokens here"
        await modal._save_from_test()
        await pilot.pause()
        assert app.result == "UNSET"  # did NOT dismiss
        assert modal.query_one("#internal-prompt-editor-error").renderable != ""


@pytest.mark.asyncio
async def test_preview_present_for_templated_absent_for_plain():
    templated = CATALOG["rag_reranker.pointwise_template"]
    app = _Host(templated, templated.default)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.screen.query("#internal-prompt-editor-preview")
    plain = CATALOG["agents.subagent_system"]
    app2 = _Host(plain, plain.default)
    async with app2.run_test() as pilot:
        await pilot.pause()
        assert not app2.screen.query("#internal-prompt-editor-preview")
```

(If `Tests/RAG/` async tests use a different marker convention than `@pytest.mark.asyncio`, match the existing `Tests/UI/` convention — check `pyproject.toml` `asyncio_mode`.)

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_internal_prompts_editor_modal.py -v`
Expected: FAIL — `ModuleNotFoundError` for the modal.

- [ ] **Step 3: Implement the modal**

```python
# tldw_chatbook/Widgets/settings_internal_prompts_editor_modal.py
"""Modal editor for one internal prompt: contract callout, placeholder chips,
TextArea, live render preview (templated prompts only), and Save/Reset/Cancel.

Pure UI — performs no config IO. Dismisses with an action dict the panel acts
on: {"action":"save","text":str} | {"action":"reset"} | None (cancel)."""

from __future__ import annotations

from typing import Optional

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Static, TextArea

from tldw_chatbook.Internal_Prompts import safe_substitute
from tldw_chatbook.Internal_Prompts.catalog import PromptSpec

# Realistic sample values for the live preview; visible ‹token› fallback for
# any declared token not mapped here.
_SAMPLE_VALUES = {
    "query": "What is quantum computing?",
    "original_query": "What is quantum computing?",
    "original_question": "What is quantum computing?",
    "question": "What is quantum computing?",
    "content": "‹document text›",
    "content_summary": "‹summary of collected sources›",
    "concatenated_texts": "1. ‹source one›\n2. ‹source two›",
    "current_date": "2026-07-22",
    "title": "Example Result Title",
    "title1": "Result One",
    "title2": "Result Two",
    "content1": "‹content one›",
    "content2": "‹content two›",
    "url": "https://example.com/article",
    "published": "2026-07-01",
    "results_list": "0. Title: A\n   Content: ‹...›",
    "tool_list": '{\n  "name": "demo",\n  "description": "…",\n  "parameters": {}\n}',
    "fence_open": "```tool_call",
    "fence_close": "```",
    "change_percentage": "12.5",
    "type": "rss",
    "name": "Example Subscription",
    "reasoning": "",
    "results_list_placeholder": "",
}


def _sample_for(token: str) -> str:
    return _SAMPLE_VALUES.get(token, f"‹{token}›")


class InternalPromptEditorModal(ModalScreen[Optional[dict]]):
    """Edit / reset a single internal prompt."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(self, *, spec: PromptSpec, active_text: str) -> None:
        super().__init__()
        self._spec = spec
        self._active_text = active_text

    def compose(self) -> ComposeResult:
        spec = self._spec
        with Vertical(id="internal-prompt-editor-modal"):
            yield Static(spec.title, classes="console-modal-header")
            yield Static(spec.description, classes="internal-prompt-editor-desc", markup=False)
            if spec.contract_note:
                yield Static(
                    "⚠ " + spec.contract_note,
                    id="internal-prompt-editor-contract",
                    classes="internal-prompt-editor-contract",
                    markup=False,
                )
            if spec.required_placeholders:
                chips = "  ".join("{" + p + "}" for p in spec.required_placeholders)
                yield Static(
                    "Required placeholders: " + chips,
                    classes="internal-prompt-editor-chips",
                    markup=False,
                )
            if spec.applies and spec.applies != "live":
                yield Static(
                    "Applies: " + spec.applies,
                    classes="internal-prompt-editor-applies",
                    markup=False,
                )
            yield TextArea(self._active_text, id="internal-prompt-editor-text")
            if spec.required_placeholders:
                yield Static("Preview", classes="internal-prompt-editor-section")
                yield Static(
                    self._render_preview(self._active_text),
                    id="internal-prompt-editor-preview",
                    classes="internal-prompt-editor-preview",
                    markup=False,
                )
            with Collapsible(title="Shipped default", collapsed=True):
                yield Static(spec.default, markup=False)
            yield Static("", id="internal-prompt-editor-error",
                         classes="internal-prompt-editor-error", markup=False)
            with Horizontal(classes="internal-prompt-editor-actions"):
                yield Button("Reset to default", id="internal-prompt-editor-reset")
                yield Button("Cancel", id="internal-prompt-editor-cancel")
                yield Button("Save", id="internal-prompt-editor-save", variant="primary")

    def on_mount(self) -> None:
        try:
            self.query_one("#internal-prompt-editor-text", TextArea).focus()
        except (NoMatches, QueryError):
            pass

    def _render_preview(self, text: str) -> str:
        values = {p: _sample_for(p) for p in self._spec.required_placeholders}
        return safe_substitute(text, **values)

    @on(TextArea.Changed, "#internal-prompt-editor-text")
    def _on_text_changed(self, event: TextArea.Changed) -> None:
        if not self._spec.required_placeholders:
            return
        try:
            self.query_one("#internal-prompt-editor-preview", Static).update(
                self._render_preview(event.text_area.text)
            )
        except (NoMatches, QueryError):
            pass

    def _validate(self, text: str) -> Optional[str]:
        if not text.strip():
            return "Prompt text cannot be empty."
        missing = [
            p for p in self._spec.required_placeholders if ("{" + p + "}") not in text
        ]
        if missing:
            return "Missing required placeholder(s): " + ", ".join(
                "{" + p + "}" for p in missing
            )
        return None

    def _do_save(self) -> None:
        text = self.query_one("#internal-prompt-editor-text", TextArea).text
        err = self._validate(text)
        if err:
            self.query_one("#internal-prompt-editor-error", Static).update(err)
            return
        self.dismiss({"action": "save", "text": text})

    async def _save_from_test(self) -> None:  # test seam; same path as the button
        self._do_save()

    @on(Button.Pressed, "#internal-prompt-editor-save")
    def _save(self, event: Button.Pressed) -> None:
        event.stop()
        self._do_save()

    @on(Button.Pressed, "#internal-prompt-editor-reset")
    def _reset(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss({"action": "reset"})

    @on(Button.Pressed, "#internal-prompt-editor-cancel")
    def _cancel_button(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest Tests/UI/test_internal_prompts_editor_modal.py -v`
Expected: 4 passed. If `Collapsible` isn't importable in this Textual version, check `python -c "from textual.widgets import Collapsible"`; if absent, replace the Collapsible block with a plain `Static("Shipped default:")` + `Static(spec.default)` and drop the collapse (note it in the report).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/settings_internal_prompts_editor_modal.py Tests/UI/test_internal_prompts_editor_modal.py
git commit -m "feat(internal-prompts): P3 editor modal with live preview + validation"
```

---

### Task 3: Panel — browse (list / group / badges / filter, read-only)

**Files:**
- Create: `tldw_chatbook/Widgets/settings_internal_prompts_panel.py`
- Test: `Tests/UI/test_internal_prompts_panel.py`

**Interfaces:**
- Consumes: Task 1 `authoring` (`iter_specs_by_subsystem`, `override_state`, `customized_count`).
- Produces: `InternalPromptsPanel(Vertical)` — composes a search `Input#internal-prompts-search` + a `VerticalScroll#internal-prompts-list` of per-subsystem `Static` headers and one `Button.internal-prompt-row` per prompt (id `prompt-row-<prompt_id>`, badges as child `Static.badge`). `class Modified(Message)`. Task 4 adds editing. Row activation is stubbed here (Task 4 wires it).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/UI/test_internal_prompts_panel.py
"""Panel browse: renders grouped rows, badges reflect config, search filters
without a rebuild."""

import pytest
from textual.app import App

from tldw_chatbook.Internal_Prompts.catalog import CATALOG
from tldw_chatbook.Widgets.settings_internal_prompts_panel import InternalPromptsPanel


class _Host(App):
    def compose(self):
        yield InternalPromptsPanel(id="p")


@pytest.mark.asyncio
async def test_renders_one_row_per_prompt(scratch_config):
    scratch_config("")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = app.query(".internal-prompt-row")
        assert len(rows) == len(CATALOG)


@pytest.mark.asyncio
async def test_customized_badge_reflects_override(scratch_config):
    scratch_config(
        '[internal_prompts.agents]\nsubagent_system = "X"\n'
    )
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        row = app.query_one("#prompt-row-agents__subagent_system")
        assert "row-customized" in row.classes


@pytest.mark.asyncio
async def test_search_hides_nonmatching_rows_without_rebuild(scratch_config):
    scratch_config("")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows_before = {r.id for r in app.query(".internal-prompt-row")}
        search = app.query_one("#internal-prompts-search")
        search.value = "answer synthesis"
        await pilot.pause()
        rows_after = {r.id for r in app.query(".internal-prompt-row")}
        assert rows_before == rows_after  # same widgets, not rebuilt
        visible = [r for r in app.query(".internal-prompt-row") if r.display]
        assert all("answer" in r.tooltip.lower() or "answer" in str(r.label).lower()
                   for r in visible)
        assert len(visible) < len(rows_before)
```

Note: the row `id` uses `__` for the dotted prompt id (Textual ids can't contain `.`) — `prompt-row-<subsystem>__<key>`. Match this in the implementation. The visible-row assertion uses whatever text carrier the row exposes (`label`/`tooltip`); align the assertion with the real row after Step 3.

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_internal_prompts_panel.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement the browse panel**

```python
# tldw_chatbook/Widgets/settings_internal_prompts_panel.py
"""Settings "Internal Prompts" panel: browse the registry prompts grouped by
subsystem with customized / default-changed badges, filter by search, and
(Task 4) open the editor modal to save/reset overrides.

Self-contained editor pattern (mirrors SettingsThemeEditor): owns its state,
posts a Modified message the screen watches for the sidebar dirty-marker."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches, QueryError
from textual.message import Message
from textual.widgets import Button, Input, Static

from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Internal_Prompts.catalog import CATALOG


def _row_id(prompt_id: str) -> str:
    return "prompt-row-" + prompt_id.replace(".", "__")


class InternalPromptsPanel(Vertical):
    """Browse + edit internal prompts. Title is rendered by the screen."""

    class Modified(Message):
        def __init__(self, customized_count: int) -> None:
            self.customized_count = customized_count
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search prompts…", id="internal-prompts-search")
        with VerticalScroll(id="internal-prompts-list"):
            for subsystem, specs in authoring.iter_specs_by_subsystem():
                yield Static(
                    f"{subsystem}  ({len(specs)})",
                    classes="internal-prompts-group-header",
                    id="group-header-" + subsystem,
                )
                for spec in specs:
                    yield self._make_row(spec)

    def _make_row(self, spec) -> Button:
        st = authoring.override_state(spec.id)
        label = spec.title
        badges = []
        if st.customized:
            badges.append("● customized")
        if st.default_changed:
            badges.append("⟳ default changed")
        if badges:
            label = f"{spec.title}   [{'  '.join(badges)}]"
        row = Button(label, id=_row_id(spec.id), classes="internal-prompt-row")
        row.tooltip = spec.title + " — " + spec.description
        if st.customized:
            row.add_class("row-customized")
        if st.default_changed:
            row.add_class("row-default-changed")
        # carry the prompt id for Task 4's activation handler
        row.prompt_id = spec.id  # type: ignore[attr-defined]
        return row

    @on(Input.Changed, "#internal-prompts-search")
    def _on_search(self, event: Input.Changed) -> None:
        needle = event.value.strip().lower()
        for subsystem, specs in authoring.iter_specs_by_subsystem():
            any_visible = False
            for spec in specs:
                match = (not needle) or needle in spec.title.lower() \
                    or needle in spec.description.lower() or needle in spec.id.lower()
                try:
                    self.query_one("#" + _row_id(spec.id), Button).display = match
                except (NoMatches, QueryError):
                    continue
                any_visible = any_visible or match
            try:
                self.query_one("#group-header-" + subsystem, Static).display = any_visible
            except (NoMatches, QueryError):
                pass

    def _refresh_row(self, prompt_id: str) -> None:
        """Targeted in-place badge refresh for one row (no recompose)."""
        try:
            row = self.query_one("#" + _row_id(prompt_id), Button)
        except (NoMatches, QueryError):
            return
        st = authoring.override_state(prompt_id)
        spec = CATALOG[prompt_id]
        badges = []
        if st.customized:
            badges.append("● customized")
        if st.default_changed:
            badges.append("⟳ default changed")
        row.label = spec.title + (f"   [{'  '.join(badges)}]" if badges else "")
        row.set_class(st.customized, "row-customized")
        row.set_class(st.default_changed, "row-default-changed")
```

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest Tests/UI/test_internal_prompts_panel.py -v`
Expected: 3 passed. Align the search test's visible-row text assertion with the real `row.label`/`row.tooltip` if it mismatches — fix the test's carrier, not the behavior.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/settings_internal_prompts_panel.py Tests/UI/test_internal_prompts_panel.py
git commit -m "feat(internal-prompts): P3 browse panel (grouped rows, badges, search filter)"
```

---

### Task 4: Panel — editing (modal push + worker save/reset + targeted refresh)

**Files:**
- Modify: `tldw_chatbook/Widgets/settings_internal_prompts_panel.py`
- Test: `Tests/UI/test_internal_prompts_panel_editing.py`

**Interfaces:**
- Consumes: Task 1 `authoring.{save_override, reset_override, override_state, customized_count}`; Task 2 `InternalPromptEditorModal`; Task 3 `_row_id`, `_refresh_row`, `Modified`.
- Produces: row activation opens the modal; save/reset persist in a thread worker; the affected row refreshes in place; `Modified` is posted.

- [ ] **Step 1: Write the failing test**

```python
# Tests/UI/test_internal_prompts_panel_editing.py
"""Row activation -> modal -> save/reset persists and refreshes the row."""

import pytest
from textual.app import App

from tldw_chatbook.Internal_Prompts import authoring
from tldw_chatbook.Widgets.settings_internal_prompts_panel import InternalPromptsPanel


class _Host(App):
    def compose(self):
        yield InternalPromptsPanel(id="p")


@pytest.mark.asyncio
async def test_save_via_panel_persists_and_marks_row(scratch_config):
    scratch_config("")
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(InternalPromptsPanel)
        # drive the panel's save path directly (modal UI covered in Task 2)
        await panel._apply_editor_result(
            "agents.subagent_system", {"action": "save", "text": "PANEL EDIT"}
        )
        await pilot.pause()
        assert authoring.override_state("agents.subagent_system").active_text == "PANEL EDIT"
        row = app.query_one("#prompt-row-agents__subagent_system")
        assert "row-customized" in row.classes


@pytest.mark.asyncio
async def test_reset_via_panel_clears_override(scratch_config):
    scratch_config('[internal_prompts.agents]\nsubagent_system = "X"\n')
    app = _Host()
    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(InternalPromptsPanel)
        await panel._apply_editor_result(
            "agents.subagent_system", {"action": "reset"}
        )
        await pilot.pause()
        assert authoring.override_state("agents.subagent_system").customized is False
        row = app.query_one("#prompt-row-agents__subagent_system")
        assert "row-customized" not in row.classes
```

- [ ] **Step 2: Run to verify fail**

Run: `.venv/bin/python -m pytest Tests/UI/test_internal_prompts_panel_editing.py -v`
Expected: FAIL — `AttributeError: ... has no attribute '_apply_editor_result'`.

- [ ] **Step 3: Add editing to the panel**

Add these imports at the top of `settings_internal_prompts_panel.py`:

```python
from textual import work
from tldw_chatbook.Widgets.settings_internal_prompts_editor_modal import (
    InternalPromptEditorModal,
)
```

Add these methods to `InternalPromptsPanel`:

```python
    @on(Button.Pressed, ".internal-prompt-row")
    def _open_editor(self, event: Button.Pressed) -> None:
        event.stop()
        prompt_id = getattr(event.button, "prompt_id", None)
        if prompt_id is None:
            return
        spec = CATALOG[prompt_id]
        st = authoring.override_state(prompt_id)
        self.app.push_screen(
            InternalPromptEditorModal(spec=spec, active_text=st.active_text),
            lambda result, pid=prompt_id: self._on_editor_closed(pid, result),
        )

    def _on_editor_closed(self, prompt_id: str, result) -> None:
        if result is None:
            return
        # schedule the async apply (worker + refresh)
        self.run_worker(self._apply_editor_result(prompt_id, result), exclusive=False)

    async def _apply_editor_result(self, prompt_id: str, result: dict) -> None:
        action = result.get("action")
        if action == "save":
            ok = await self._persist(prompt_id, result.get("text", ""), reset=False)
        elif action == "reset":
            ok = await self._persist(prompt_id, "", reset=True)
        else:
            return
        if ok:
            self._refresh_row(prompt_id)
            self.post_message(self.Modified(authoring.customized_count()))
        else:
            self.app.notify("Could not save the prompt override.", severity="error")

    async def _persist(self, prompt_id: str, text: str, reset: bool) -> bool:
        import asyncio
        def _io() -> bool:
            try:
                return (
                    authoring.reset_override(prompt_id)
                    if reset
                    else authoring.save_override(prompt_id, text)
                )
            except Exception:  # never let the worker crash the app
                return False
        return await asyncio.to_thread(_io)
```

Rationale for `asyncio.to_thread` over `@work(thread=True)`: `_apply_editor_result` is already an async task (scheduled via `run_worker`); offloading just the blocking config IO to a thread keeps the UI loop responsive and the exception contained, matching the constraint without a second worker decorator. (This mirrors the P1/P2 pattern of offloading sync config IO with `to_thread`.)

- [ ] **Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest Tests/UI/test_internal_prompts_panel_editing.py Tests/UI/test_internal_prompts_panel.py -v`
Expected: all pass (browse tests still green).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/settings_internal_prompts_panel.py Tests/UI/test_internal_prompts_panel_editing.py
git commit -m "feat(internal-prompts): P3 panel editing (modal push, threaded persist, targeted refresh)"
```

---

### Task 5: Settings-screen wiring + CSS + invariant tests

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py` (enum member)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` (5 spots + Modified watcher + import)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (styles)
- Modify: `Tests/UI/test_settings_configuration_hub.py` (invariants)

**Interfaces:**
- Consumes: `InternalPromptsPanel` (Task 3/4), `authoring.customized_count` (Task 1).
- Produces: the live "Internal Prompts" category in the Expert group rendering the panel.

- [ ] **Step 1: Add the enum member**

In `settings_config_models.py`, add to `SettingsCategoryId` (after `ADVANCED_CONFIG` or in logical order):

```python
    INTERNAL_PROMPTS = "internal-prompts"
```

- [ ] **Step 2: Wire the 5 settings_screen spots**

Re-locate each anchor by its function name (line numbers drift). Import at top of `settings_screen.py` with the other widget imports:

```python
from tldw_chatbook.Widgets.settings_internal_prompts_panel import InternalPromptsPanel
from tldw_chatbook.Internal_Prompts import authoring as internal_prompts_authoring
```

**(a) `_category_summaries()`** — add an entry:

```python
        SettingsCategorySummary(
            SettingsCategoryId.INTERNAL_PROMPTS,
            "Internal Prompts",
            "View and edit the system prompts tldw_chatbook uses internally "
            "(RAG, web search, agents, summarization, more).",
            self._internal_prompts_status(),
        ),
```

Add the helper method on the screen:

```python
    def _internal_prompts_status(self) -> str:
        try:
            n = internal_prompts_authoring.customized_count()
        except Exception:
            return ""
        return f"{n} customized" if n else "Defaults"
```

**(b) `_category_groups()`** — add `SettingsCategoryId.INTERNAL_PROMPTS` to the **Expert** tuple (the one currently `(SettingsCategoryId.ADVANCED_CONFIG,)`):

```python
        ("Expert", (SettingsCategoryId.INTERNAL_PROMPTS, SettingsCategoryId.ADVANCED_CONFIG)),
```

**(c) `_render_detail_pane()`** — add a branch (after the SPLASH_SCREEN branch, before the terminal `else`):

```python
        elif category is SettingsCategoryId.INTERNAL_PROMPTS:
            yield Static("Internal Prompts", classes="destination-section settings-column-title")
            yield InternalPromptsPanel(id="settings-internal-prompts-panel")
```

**(d) `_render_impact_pane()`** — extend the self-contained-editor guard (currently `if summary.category not in (THEME, SPLASH_SCREEN):`) to include INTERNAL_PROMPTS:

```python
        if summary.category not in (
            SettingsCategoryId.THEME,
            SettingsCategoryId.SPLASH_SCREEN,
            SettingsCategoryId.INTERNAL_PROMPTS,
        ):
```

and add a branch mirroring the THEME one:

```python
        elif summary.category is SettingsCategoryId.INTERNAL_PROMPTS:
            yield Static("Edit the prompts used by internal tooling.", classes="destination-section")
            yield self._detail_row("Save target", "~/.config/tldw_cli/config.toml  [internal_prompts]")
            yield self._detail_row("Note", "Use each prompt's own Save / Reset buttons.")
            try:
                n = internal_prompts_authoring.customized_count()
            except Exception:
                n = 0
            yield self._detail_row("Customized prompts", str(n))
```

**(e) Modified watcher** — add a reactive + handler mirroring `theme_editor_modified` (near `settings_screen.py:746` / `8043`):

```python
    internal_prompts_dirty = reactive(0, recompose=True)
```

```python
    @on(InternalPromptsPanel.Modified)
    def _on_internal_prompts_modified(self, event: InternalPromptsPanel.Modified) -> None:
        self.internal_prompts_dirty = event.customized_count
```

Leave `INTERNAL_PROMPTS` OUT of `GUIDED_SETTINGS_MUTATION_CATEGORIES` (no change to that frozenset).

- [ ] **Step 3: Add CSS to the partial**

Append to `tldw_chatbook/css/components/_agentic_terminal.tcss` (match the file's existing class conventions; these are minimal and safe):

```css
/* Internal Prompts settings page */
#internal-prompts-search { width: 1fr; margin: 0 0 1 0; }
#internal-prompts-list { height: 1fr; }
.internal-prompts-group-header { text-style: bold; padding: 1 0 0 0; color: $text-muted; }
.internal-prompt-row { width: 1fr; height: auto; }
.internal-prompt-row.row-customized { border-left: thick $accent; }
.internal-prompt-row.row-default-changed { border-left: thick $warning; }
#internal-prompt-editor-modal { width: 90%; max-width: 120; height: auto; max-height: 90%; padding: 1 2; background: $panel; border: thick $primary; }
.internal-prompt-editor-contract { color: $warning; padding: 1 0; }
.internal-prompt-editor-chips { color: $text-muted; }
.internal-prompt-editor-applies { color: $text-muted; text-style: italic; }
.internal-prompt-editor-preview { background: $boost; padding: 1; height: auto; max-height: 12; }
.internal-prompt-editor-error { color: $error; }
.internal-prompt-editor-actions { align-horizontal: right; height: auto; padding: 1 0 0 0; }
```

Do NOT edit `css/tldw_cli_modular.tcss`. Verify the app rebuilds it: `.venv/bin/python -c "from tldw_chatbook.css import build_css; build_css.build()"` (or the module's actual entry — check `build_css.py`), then confirm the bundle contains `internal-prompt-row`.

- [ ] **Step 4: Update the hub invariant tests**

In `Tests/UI/test_settings_configuration_hub.py`: bump the total-category-count assertion by 1; add `INTERNAL_PROMPTS` to the expected Expert-group membership; confirm the existing "THEME/SPLASH not in guided set" test still passes (INTERNAL_PROMPTS should also be absent from the guided set — add that assertion if the test enumerates non-guided self-contained pages). Read the current assertions first and adjust exact expected values.

- [ ] **Step 5: Run the settings + panel suites**

Run: `.venv/bin/python -m pytest Tests/UI/test_settings_configuration_hub.py Tests/UI/test_internal_prompts_panel.py Tests/UI/test_internal_prompts_panel_editing.py Tests/UI/test_internal_prompts_editor_modal.py -q`
Expected: all pass. Also import-smoke the screen: `.venv/bin/python -c "import tldw_chatbook.UI.Screens.settings_screen"`.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_config_models.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss Tests/UI/test_settings_configuration_hub.py
git commit -m "feat(internal-prompts): P3 wire Internal Prompts page into Settings (Expert group) + CSS"
```

---

### Task 6: Verification sweep

**Files:** none unless a regression forces a fix.

- [ ] **Step 1: Registry + settings + UI suites**

Run: `.venv/bin/python -m pytest Tests/Internal_Prompts/ Tests/UI/test_settings_configuration_hub.py Tests/UI/test_internal_prompts_panel.py Tests/UI/test_internal_prompts_panel_editing.py Tests/UI/test_internal_prompts_editor_modal.py -q`
Expected: all pass.

- [ ] **Step 2: Cold-import guard**

Run: `.venv/bin/python -m pytest Tests/Internal_Prompts/test_import_hygiene.py -q`
Expected: pass — `authoring.py` must not have pulled config into package import.

- [ ] **Step 3: Wider settings smoke**

Run: `.venv/bin/python -m pytest Tests/UI/ -k "settings" -q`
Expected: same results as origin/dev baseline; note any pre-existing failures rather than fixing.

- [ ] **Step 4: Commit only if a fix was required; otherwise proceed to Task 7.**

---

### Task 7: Live-TUI QA + user approval gate (controller-run — MERGE GATE)

**Files:** QA captures under a scratch dir; no source changes unless QA finds a defect.

This task is executed by the controller (not a fresh subagent) using the project's TUI-run/verify recipe (the `verify` skill / textual-serve capture). It is the **hard merge gate** per project policy: every new Settings screen needs explicit user screenshot approval before merge.

- [ ] **Step 1:** Launch the app against a scratch `TLDW_CONFIG_PATH` profile; navigate Settings → Expert → Internal Prompts.
- [ ] **Step 2:** Capture: (a) the page with grouped list + a customized badge, (b) the search filtering the list, (c) the editor modal for a templated prompt showing the contract callout + placeholder chips + live preview, (d) a Save landing a customized badge, (e) a Reset clearing it, (f) the impact pane showing the customized count.
- [ ] **Step 3:** Present the captures to the user for sign-off. Fix any defect found, re-capture. Do NOT merge before explicit approval.
- [ ] **Step 4:** On approval, hand off via superpowers:finishing-a-development-branch (PR to `dev`; merge only on the user's word).

---

## Self-review (performed at plan-writing time)

- **Spec coverage:** authoring API (§ Component A) → Task 1; editor modal incl. preview/contract/chips/applies/validation (§ Component C) → Task 2; panel browse + badges + filter (§ Component B) → Task 3; panel persistence + targeted refresh + Modified (§ Component B, Deltas 1-2) → Task 4; the 6 wiring edits + CSS-location correction (§ verified wiring, Delta 2) + hub invariants → Task 5; error model (§5) is realized in authoring (bool, never raise) + worker try/except (Tasks 1,4); testing (§6) distributed across each task; merge gate → Task 7.
- **Placeholder scan:** the two deliberate "align the test's text carrier / adjust exact expected values" notes (Tasks 3,5) are verification-against-real-widgets instructions with the invariant fixed, not TBDs; the Collapsible fallback (Task 2) is a named contingency. No unspecified work.
- **Type consistency:** `OverrideState` fields, `authoring.*` signatures, `_row_id` (`__` for dots), the modal's dismiss dict shape (`{"action","text"}` / `{"action":"reset"}` / `None`), and `InternalPromptsPanel.Modified(customized_count)` are identical across Tasks 1-5.
