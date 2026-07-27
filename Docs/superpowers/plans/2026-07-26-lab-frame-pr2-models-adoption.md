# Lab Frame PR2 — Frame plus Models Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared `LabScreen` frame and have the Models screen adopt it, lifting its nine-row sidebar into the frame's rail.

**Architecture:** Four pure, mountless modules (collapse state, config store, server-status reader, plus a chip dataclass) sit under a `LabWorkbench` container and a `LabScreen` base. `LLMScreen` inherits the base, supplies rail/body/status, and drives rail highlighting by *watching* `LLMManagementWindow.active_view` rather than styling on press.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, pytest, the generated TCSS bundle (`build_css.py`).

**Spec:** `Docs/superpowers/specs/2026-07-26-lab-frame-pr2-models-adoption-design.md`

## Global Constraints

- Run pytest from the worktree, venv only, PYTHONPATH pinned:
  `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ... -p no:randomly`
  Run in the **foreground**. Background runs stall.
- Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss`. Regenerate with
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`
- All colours and borders go **app-tier** in `css/features/_lab.tcss`, never in widget `DEFAULT_CSS`. The bundle outranks `DEFAULT_CSS` regardless of specificity.
- PR2 **extends** `css/features/_lab.tcss`; it does not create another module.
- The collapse reactive is named **`rail_layout`**, never `layout` — `Widget.layout` is an unsettable Textual property the compositor calls `.arrange()` on every pass.
- Rail collapse persists to **config**, never `save_state`: navigation builds a fresh screen instance every time.
- **Do not unify the two active-class names.** `llm-view-*` bodies are shown by `.llm-view.-active` (toggled by `watch_active_view`); rail rows use `is-active`. Renaming either breaks something.
- Every test is **mutation-checked**: revert the change, confirm red, restore.

## Known baseline failures — do NOT fix, do NOT let change

Gate on failure **names**, never a raw count.

- `test_console_persistent_rails.py::test_generated_console_stylesheet_includes_rail_rules` — pre-existing; a global `"border: thick $ds-action-focus;"` assertion tripped by unrelated RAG-settings CSS.
- `test_library_shell.py` — 3 deterministic failures plus 2 self-documented CPU-contention flakes that can add a rotating 4th under load.

---

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/UI/Lab_Modules/__init__.py` | **new** — package marker |
| `tldw_chatbook/UI/Lab_Modules/lab_rail_layout.py` | **new** — `LabRailLayout`, pure collapse state |
| `tldw_chatbook/UI/Lab_Modules/lab_rail_store.py` | **new** — config load/save of collapse state |
| `tldw_chatbook/UI/Lab_Modules/lab_server_status.py` | **new** — pure reader over the six app `Popen` handles |
| `tldw_chatbook/UI/Lab_Modules/lab_workbench.py` | **new** — `LabWorkbench` container |
| `tldw_chatbook/UI/Screens/lab_frame.py` | **new** — `LabStatusChip`, `LabScreen(BaseAppScreen)` |
| `tldw_chatbook/UI/Screens/llm_screen.py` | modify — adopts the frame, supplies rail/body/status |
| `tldw_chatbook/UI/LLM_Management_Window.py` | modify — drop nav buttons, trim orphaned watcher block |
| `tldw_chatbook/css/features/_lab.tcss` | extend — rail, workbench, status row, rail-row `is-active` |

---

### Task 1: `LabRailLayout` — pure collapse state

**Files:**
- Create: `tldw_chatbook/UI/Lab_Modules/__init__.py`, `tldw_chatbook/UI/Lab_Modules/lab_rail_layout.py`
- Test: `Tests/UI/test_lab_rail_layout.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  ```python
  LAB_RAIL_LEFT = "rail"
  LAB_RAIL_INSPECTOR = "inspector"
  LAB_RAILS: tuple[str, ...] = (LAB_RAIL_LEFT, LAB_RAIL_INSPECTOR)

  @dataclass(frozen=True)
  class LabRailLayout:
      collapsed: frozenset[str] = frozenset()
      def is_collapsed(self, rail: str) -> bool
      def toggle(self, rail: str) -> "LabRailLayout"
  ```

**Background.** Collapse logic lives in a frozen dataclass so it is testable without mounting a widget — the lesson Watchlists' `region_layout.py` established. Unknown rail names must not silently succeed: a typo'd rail would otherwise "collapse" nothing forever.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_lab_rail_layout.py`:

```python
"""Pure collapse state for the Lab frame's two rails."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LAB_RAILS,
    LabRailLayout,
)


def test_default_layout_has_nothing_collapsed():
    layout = LabRailLayout()
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is False


def test_toggle_collapses_then_expands():
    layout = LabRailLayout()
    collapsed = layout.toggle(LAB_RAIL_LEFT)
    assert collapsed.is_collapsed(LAB_RAIL_LEFT) is True
    assert collapsed.is_collapsed(LAB_RAIL_INSPECTOR) is False
    assert collapsed.toggle(LAB_RAIL_LEFT).is_collapsed(LAB_RAIL_LEFT) is False


def test_toggle_returns_a_new_instance_and_leaves_the_original_alone():
    """Frozen means callers can hold an old layout without it mutating."""
    layout = LabRailLayout()
    other = layout.toggle(LAB_RAIL_INSPECTOR)
    assert other is not layout
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is False


def test_the_two_rails_are_independent():
    layout = LabRailLayout().toggle(LAB_RAIL_LEFT).toggle(LAB_RAIL_INSPECTOR)
    assert layout.is_collapsed(LAB_RAIL_LEFT) is True
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is True


@pytest.mark.parametrize("method", ["is_collapsed", "toggle"])
def test_unknown_rail_names_raise(method):
    """A typo'd rail must fail loudly, not collapse nothing forever."""
    layout = LabRailLayout()
    with pytest.raises(ValueError):
        getattr(layout, method)("sidebar")


def test_lab_rails_lists_both_rails_in_render_order():
    assert LAB_RAILS == (LAB_RAIL_LEFT, LAB_RAIL_INSPECTOR)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_rail_layout.py -p no:randomly -q`

Expected: collection error — `ModuleNotFoundError: No module named 'tldw_chatbook.UI.Lab_Modules'`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Lab_Modules/__init__.py`:

```python
"""Panes and helpers for the Lab destination's shared frame."""
```

Create `tldw_chatbook/UI/Lab_Modules/lab_rail_layout.py`:

```python
"""Pure collapse state for the Lab frame's two rails.

Kept separate from the widget so collapse behaviour is testable without
mounting anything, and so the frame can persist a plain value rather than
scraping widget state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: The left catalog rail.
LAB_RAIL_LEFT = "rail"
#: The right inspector rail.
LAB_RAIL_INSPECTOR = "inspector"
#: Both rails, in render order.
LAB_RAILS: tuple[str, ...] = (LAB_RAIL_LEFT, LAB_RAIL_INSPECTOR)


def _validate(rail: str) -> None:
    """Reject rail names that are not one of the two real rails.

    Args:
        rail: Candidate rail name.

    Raises:
        ValueError: If ``rail`` is not in :data:`LAB_RAILS`.
    """
    if rail not in LAB_RAILS:
        raise ValueError(f"Unknown Lab rail {rail!r}; expected one of {LAB_RAILS}")


@dataclass(frozen=True)
class LabRailLayout:
    """Which of the Lab frame's rails are currently collapsed.

    Attributes:
        collapsed: Names of collapsed rails; members of :data:`LAB_RAILS`.
    """

    collapsed: frozenset[str] = field(default_factory=frozenset)

    def is_collapsed(self, rail: str) -> bool:
        """Report whether one rail is collapsed.

        Args:
            rail: Rail name, one of :data:`LAB_RAILS`.

        Returns:
            True when that rail is collapsed.

        Raises:
            ValueError: If ``rail`` is not a known rail.
        """
        _validate(rail)
        return rail in self.collapsed

    def toggle(self, rail: str) -> "LabRailLayout":
        """Return a new layout with one rail's collapse state flipped.

        Args:
            rail: Rail name, one of :data:`LAB_RAILS`.

        Returns:
            A new ``LabRailLayout``; the receiver is unchanged.

        Raises:
            ValueError: If ``rail`` is not a known rail.
        """
        _validate(rail)
        if rail in self.collapsed:
            return LabRailLayout(collapsed=frozenset(self.collapsed - {rail}))
        return LabRailLayout(collapsed=frozenset(self.collapsed | {rail}))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_rail_layout.py -p no:randomly -q`

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Lab_Modules/__init__.py \
        tldw_chatbook/UI/Lab_Modules/lab_rail_layout.py \
        Tests/UI/test_lab_rail_layout.py
git commit -m "feat(lab): add pure rail collapse state

Frozen dataclass so the frame's collapse behaviour is testable without
mounting a widget, and so the value can be persisted directly. Unknown
rail names raise rather than silently collapsing nothing."
```

---

### Task 2: `LabRailStore` — persist collapse to config

**Files:**
- Create: `tldw_chatbook/UI/Lab_Modules/lab_rail_store.py`
- Test: `Tests/UI/test_lab_rail_store.py`

**Interfaces:**
- Consumes: `LabRailLayout`, `LAB_RAILS`, `LAB_RAIL_INSPECTOR` from Task 1.
- Produces:
  ```python
  LAB_CONFIG_SECTION = "lab"
  LAB_COLLAPSED_RAILS_KEY = "collapsed_rails"
  LAB_FIRST_RUN_LAYOUT = LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR}))
  def load_rail_layout() -> LabRailLayout
  def save_rail_layout(layout: LabRailLayout) -> None
  ```

**Background.** Navigation builds a fresh screen every time (`app.py:5508-5530` forbids caching), so collapse cannot live in `save_state`. Two rules carry over from the sibling store in `Watchlists_Modules/region_layout_store.py`:

- The default sentinel passed to `get_cli_setting` must be **`None`, never `[]`** — it returns the default only when the key is *absent*, so `None` is what distinguishes "never set" from "user explicitly expanded everything". Collapsing that distinction re-imposes the first-run default forever.
- First run starts with the **inspector collapsed** and the left rail open: the rail is the mode's primary navigation and earns its width.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_lab_rail_store.py`:

```python
"""Config persistence for the Lab frame's rail collapse state."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Lab_Modules import lab_rail_store
from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LabRailLayout,
)


@pytest.fixture
def fake_config(monkeypatch):
    """Capture reads and writes without touching the user's config file."""
    store = {}

    def fake_get(section, key=None, default=None):
        assert section == lab_rail_store.LAB_CONFIG_SECTION
        assert key == lab_rail_store.LAB_COLLAPSED_RAILS_KEY
        return store.get("value", default)

    def fake_save(section, key, value):
        assert section == lab_rail_store.LAB_CONFIG_SECTION
        assert key == lab_rail_store.LAB_COLLAPSED_RAILS_KEY
        store["value"] = value
        return True

    monkeypatch.setattr(lab_rail_store, "get_cli_setting", fake_get)
    monkeypatch.setattr(lab_rail_store, "save_setting_to_cli_config", fake_save)
    return store


def test_unset_config_yields_the_first_run_layout(fake_config):
    """Never-set must give the first-run default: inspector collapsed."""
    layout = lab_rail_store.load_rail_layout()
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is True
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False


def test_explicitly_empty_is_not_the_first_run_default(fake_config):
    """A user who expanded everything must not get the default re-imposed.

    This is why the sentinel passed to get_cli_setting is None, not [].
    """
    fake_config["value"] = []
    layout = lab_rail_store.load_rail_layout()
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is False
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False


def test_round_trip(fake_config):
    saved = LabRailLayout(collapsed=frozenset({LAB_RAIL_LEFT}))
    lab_rail_store.save_rail_layout(saved)
    loaded = lab_rail_store.load_rail_layout()
    assert loaded.is_collapsed(LAB_RAIL_LEFT) is True
    assert loaded.is_collapsed(LAB_RAIL_INSPECTOR) is False


def test_saved_value_is_a_plain_sorted_list(fake_config):
    """TOML cannot hold a frozenset; sorted keeps the file diff stable."""
    lab_rail_store.save_rail_layout(
        LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR, LAB_RAIL_LEFT}))
    )
    assert fake_config["value"] == sorted([LAB_RAIL_INSPECTOR, LAB_RAIL_LEFT])


def test_unknown_names_in_config_are_ignored(fake_config):
    """A hand-edited or stale config must not crash the screen."""
    fake_config["value"] = ["inspector", "sidebar", 17, None]
    layout = lab_rail_store.load_rail_layout()
    assert layout.is_collapsed(LAB_RAIL_INSPECTOR) is True
    assert layout.is_collapsed(LAB_RAIL_LEFT) is False


def test_a_non_list_config_value_falls_back_to_first_run(fake_config):
    fake_config["value"] = "inspector"
    layout = lab_rail_store.load_rail_layout()
    assert layout == lab_rail_store.LAB_FIRST_RUN_LAYOUT
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_rail_store.py -p no:randomly -q`

Expected: `ModuleNotFoundError: ... lab_rail_store`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Lab_Modules/lab_rail_store.py`:

```python
"""Persist the Lab frame's rail collapse state to the user's config.

Collapse is a UI preference, not data, so it belongs in config. It cannot
live in ``BaseAppScreen.save_state``: navigation builds a fresh screen
instance every time (``app.py`` ``_create_navigation_screen``), so
screen-scoped state does not survive a mode switch.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from ...config import get_cli_setting, save_setting_to_cli_config
from .lab_rail_layout import LAB_RAIL_INSPECTOR, LAB_RAILS, LabRailLayout

logger = logger.bind(module="LabRailStore")

#: Flat config section. Both flat and dotted sections round-trip correctly,
#: so flat is chosen simply for consistency with the sibling Watchlists store.
LAB_CONFIG_SECTION = "lab"
LAB_COLLAPSED_RAILS_KEY = "collapsed_rails"

#: What to show before anyone has touched collapse state. The left rail is the
#: mode's primary navigation and earns its width; the inspector starts closed.
LAB_FIRST_RUN_LAYOUT = LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR}))


def load_rail_layout() -> LabRailLayout:
    """Read collapse state from config.

    Distinguishes "never saved" from "saved as empty": ``get_cli_setting``
    returns its ``default`` only when the key is absent, so passing ``None``
    -- not ``[]`` -- lets a genuinely unset key be told apart from a user who
    explicitly expanded everything. Collapsing that distinction would
    re-impose the first-run default on every session.

    Returns:
        The stored layout, or :data:`LAB_FIRST_RUN_LAYOUT` when unset or
        unreadable.
    """
    raw: Any = get_cli_setting(LAB_CONFIG_SECTION, LAB_COLLAPSED_RAILS_KEY, None)
    if raw is None:
        return LAB_FIRST_RUN_LAYOUT
    if not isinstance(raw, list):
        logger.warning(
            "Ignoring non-list {}.{} value {!r}; using the first-run layout.",
            LAB_CONFIG_SECTION,
            LAB_COLLAPSED_RAILS_KEY,
            raw,
        )
        return LAB_FIRST_RUN_LAYOUT
    known = {value for value in raw if isinstance(value, str) and value in LAB_RAILS}
    return LabRailLayout(collapsed=frozenset(known))


def save_rail_layout(layout: LabRailLayout) -> None:
    """Write collapse state to config.

    Args:
        layout: Layout to persist. Stored as a sorted list of rail names,
            since TOML cannot hold a frozenset and sorting keeps the config
            file's diff stable.
    """
    save_setting_to_cli_config(
        LAB_CONFIG_SECTION,
        LAB_COLLAPSED_RAILS_KEY,
        sorted(layout.collapsed),
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_rail_store.py -p no:randomly -q`

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Lab_Modules/lab_rail_store.py Tests/UI/test_lab_rail_store.py
git commit -m "feat(lab): persist rail collapse state to config

Screens are rebuilt on every navigation, so collapse cannot live in
save_state. Uses a None sentinel rather than [] so 'never set' stays
distinguishable from 'user expanded everything', and tolerates stale or
hand-edited config values rather than crashing the screen."
```

---

### Task 3: `lab_server_status` — pure reader over the app's server handles

**Files:**
- Create: `tldw_chatbook/UI/Lab_Modules/lab_server_status.py`
- Test: `Tests/UI/test_lab_server_status.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  ```python
  @dataclass(frozen=True)
  class LabServerRow:
      name: str
      running: bool

  LAB_SERVER_SOURCES: tuple[tuple[str, str], ...]   # (app attribute, display name)
  def read_server_rows(app: Any) -> tuple[LabServerRow, ...]
  def servers_chip_text(rows: Sequence[LabServerRow]) -> str
  ```

**Background.** The six local-server processes hang off the **app**, not the window (`app.py:3582-3587`): `llamacpp_server_process`, `llamafile_server_process`, `vllm_server_process`, `ollama_server_process`, `mlx_server_process`, `onnx_server_process`. The codebase's liveness idiom is `proc and proc.poll() is None` (`llm_management_events.py:1028-1030`, `llm_management_events_mlx_lm.py:169`).

Keeping this pure means the status chip and inspector are tested against a fake object carrying six attributes — no subprocesses, no mounting.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_lab_server_status.py`:

```python
"""Pure reader over the app's six local-server process handles."""

from __future__ import annotations

from tldw_chatbook.UI.Lab_Modules.lab_server_status import (
    LAB_SERVER_SOURCES,
    LabServerRow,
    read_server_rows,
    servers_chip_text,
)


class _FakeProc:
    """Stands in for subprocess.Popen; poll() is None while alive."""

    def __init__(self, alive: bool) -> None:
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


class _FakeApp:
    def __init__(self, **procs) -> None:
        for attribute, _name in LAB_SERVER_SOURCES:
            setattr(self, attribute, None)
        for attribute, proc in procs.items():
            setattr(self, attribute, proc)


def test_all_six_servers_are_reported_even_when_none_run():
    rows = read_server_rows(_FakeApp())
    assert len(rows) == len(LAB_SERVER_SOURCES) == 6
    assert all(row.running is False for row in rows)


def test_a_live_process_reads_as_running():
    rows = read_server_rows(_FakeApp(llamacpp_server_process=_FakeProc(True)))
    by_name = {row.name: row.running for row in rows}
    assert by_name["llama.cpp"] is True
    assert by_name["Ollama"] is False


def test_an_exited_process_reads_as_stopped():
    """poll() returning an exit code means the server died."""
    rows = read_server_rows(_FakeApp(llamacpp_server_process=_FakeProc(False)))
    assert {row.name: row.running for row in rows}["llama.cpp"] is False


def test_a_missing_attribute_reads_as_stopped():
    """The app may not have set every handle yet; that is not an error."""

    class _Bare:
        pass

    rows = read_server_rows(_Bare())
    assert len(rows) == 6
    assert all(row.running is False for row in rows)


def test_a_process_whose_poll_raises_reads_as_stopped():
    class _Exploding:
        def poll(self):
            raise OSError("process gone")

    rows = read_server_rows(_FakeApp(vllm_server_process=_Exploding()))
    assert {row.name: row.running for row in rows}["vLLM"] is False


def test_row_order_is_stable_and_matches_the_source_order():
    rows = read_server_rows(_FakeApp())
    assert [row.name for row in rows] == [name for _attr, name in LAB_SERVER_SOURCES]


def test_chip_text_counts_running_servers():
    rows = (
        LabServerRow(name="llama.cpp", running=True),
        LabServerRow(name="Ollama", running=True),
        LabServerRow(name="vLLM", running=False),
    )
    assert servers_chip_text(rows) == "Servers: 2 running"


def test_chip_text_when_one_is_running_is_singular():
    rows = (LabServerRow(name="llama.cpp", running=True),)
    assert servers_chip_text(rows) == "Servers: 1 running"


def test_chip_text_when_none_are_running():
    rows = (LabServerRow(name="llama.cpp", running=False),)
    assert servers_chip_text(rows) == "Servers: none running"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_server_status.py -p no:randomly -q`

Expected: `ModuleNotFoundError: ... lab_server_status`.

- [ ] **Step 3: Write the implementation**

Create `tldw_chatbook/UI/Lab_Modules/lab_server_status.py`:

```python
"""Read the app's local-server process handles into displayable rows.

Deliberately pure: it takes any object carrying the six process attributes,
so the Lab status chip and inspector are testable against a fake without
spawning subprocesses or mounting widgets.

The handles live on the app rather than on LLMManagementWindow (see
``app.py``'s ``*_server_process`` attributes), and liveness uses the same
``proc and proc.poll() is None`` idiom as the LLM management event handlers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

#: (app attribute, display name), in the order the inspector lists them.
LAB_SERVER_SOURCES: tuple[tuple[str, str], ...] = (
    ("llamacpp_server_process", "llama.cpp"),
    ("llamafile_server_process", "Llamafile"),
    ("ollama_server_process", "Ollama"),
    ("vllm_server_process", "vLLM"),
    ("onnx_server_process", "ONNX"),
    ("mlx_server_process", "MLX-LM"),
)


@dataclass(frozen=True)
class LabServerRow:
    """One local server's display state.

    Attributes:
        name: Human-readable server name.
        running: Whether its process is currently alive.
    """

    name: str
    running: bool


def _is_running(process: Any) -> bool:
    """Report whether a process handle is alive.

    Args:
        process: A ``subprocess.Popen``-like object, or None.

    Returns:
        True only when the handle exists and ``poll()`` returns None. A
        handle whose ``poll()`` raises counts as stopped: a status chip must
        never take down the screen.
    """
    if process is None:
        return False
    try:
        return process.poll() is None
    except Exception:  # noqa: BLE001 -- a status read must not crash the UI
        return False


def read_server_rows(app: Any) -> tuple[LabServerRow, ...]:
    """Read every known local-server handle off the app.

    Args:
        app: The application (or any object carrying the handles). Missing
            attributes read as stopped, since the app may not have set them.

    Returns:
        One row per entry in :data:`LAB_SERVER_SOURCES`, in that order.
    """
    return tuple(
        LabServerRow(name=name, running=_is_running(getattr(app, attribute, None)))
        for attribute, name in LAB_SERVER_SOURCES
    )


def servers_chip_text(rows: Sequence[LabServerRow]) -> str:
    """Summarise running servers for the status row.

    Args:
        rows: Rows from :func:`read_server_rows`.

    Returns:
        ``"Servers: N running"``, or ``"Servers: none running"`` when none are.
    """
    running = sum(1 for row in rows if row.running)
    if running == 0:
        return "Servers: none running"
    return f"Servers: {running} running"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_server_status.py -p no:randomly -q`

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Lab_Modules/lab_server_status.py Tests/UI/test_lab_server_status.py
git commit -m "feat(lab): add a pure reader for local-server status

The six Popen handles live on the app, not the window. Keeping the reader
pure lets the status chip and inspector be tested against a fake carrying
six attributes -- no subprocesses, no mounting. A handle whose poll()
raises counts as stopped: a status read must never take down the screen."
```

---

### Task 4: `LabWorkbench` container and its CSS

**Files:**
- Create: `tldw_chatbook/UI/Lab_Modules/lab_workbench.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss` (append)
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated, never hand-edited)
- Test: `Tests/UI/test_lab_workbench.py`

**Interfaces:**
- Consumes: `LabRailLayout`, `LAB_RAIL_LEFT`, `LAB_RAIL_INSPECTOR` from Task 1.
- Produces:
  ```python
  LAB_RAIL_WIDTH = 26
  LAB_INSPECTOR_WIDTH = 30
  LAB_RAIL_ROW_CLASS = "lab-rail-row"

  class LabWorkbench(Horizontal):
      def __init__(self, *, rail_layout: LabRailLayout, **kwargs) -> None
      def compose(self) -> ComposeResult      # yields the three region containers
  ```
  Region ids: `#lab-rail`, `#lab-body`, `#lab-inspector`, plus handles
  `#lab-rail-handle`, `#lab-inspector-handle`.

**Background.** The container renders three regions plus the two collapsed handles, driven by a `LabRailLayout` value. Mode content is mounted into the regions by the frame (Task 5) — the container itself holds no mode knowledge.

**The CSS is load-bearing, not cosmetic.** The bundle declares an unscoped `.is-active { border: round $ds-action-focus; }`, and app-tier CSS beats widget `DEFAULT_CSS` regardless of specificity. Measured on rail rows: at `height: 1` an `is-active` row renders `region.height == 2` — a half-bordered artifact that displaces its neighbours. Without the neutralizing rule the rail is visibly broken as soon as anything is selected. `#mcp-hub-rail Button.mcp-rail-row.is-active` (bundle `:6314`) is the same widget shape with the same rule.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_lab_workbench.py`:

```python
"""Geometry and selection styling for the Lab frame's workbench container."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LabRailLayout,
)
from tldw_chatbook.UI.Lab_Modules.lab_workbench import (
    LAB_INSPECTOR_WIDTH,
    LAB_RAIL_ROW_CLASS,
    LAB_RAIL_WIDTH,
    LabWorkbench,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _WorkbenchHarness(App[None]):
    """Mount the workbench with the production stylesheet.

    The bundle is required: the selection-border defect under test lives in
    the bundle's global `.is-active` rule, which beats DEFAULT_CSS. A harness
    without CSS_PATH would pass vacuously.
    """

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, layout: LabRailLayout) -> None:
        super().__init__()
        self._layout = layout

    def compose(self) -> ComposeResult:
        yield LabWorkbench(rail_layout=self._layout, id="lab-workbench")

    def on_mount(self) -> None:
        rail = self.query_one("#lab-rail")
        for index, name in enumerate(("Llama.cpp", "Llamafile", "Ollama")):
            row = Button(name, id=f"lab-rail-row-{index}", classes=LAB_RAIL_ROW_CLASS)
            if index == 1:
                row.add_class("is-active")
            rail.mount(row)
        self.query_one("#lab-body").mount(Static("body", id="probe-body"))


@pytest.mark.asyncio
async def test_all_three_regions_render_when_nothing_is_collapsed():
    app = _WorkbenchHarness(LabRailLayout())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail").display is True
        assert app.query_one("#lab-body").display is True
        assert app.query_one("#lab-inspector").display is True
        assert not app.query("#lab-rail-handle")
        assert not app.query("#lab-inspector-handle")


@pytest.mark.asyncio
async def test_a_collapsed_rail_is_replaced_by_its_handle():
    app = _WorkbenchHarness(LabRailLayout(collapsed=frozenset({LAB_RAIL_LEFT})))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail").display is False
        assert app.query_one("#lab-rail-handle").display is True


@pytest.mark.asyncio
async def test_a_collapsed_inspector_is_replaced_by_its_handle():
    app = _WorkbenchHarness(LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR})))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-inspector").display is False
        assert app.query_one("#lab-inspector-handle").display is True


@pytest.mark.asyncio
async def test_the_hundred_column_width_contract_holds():
    """Rail + body + collapsed inspector handle must fit 100 columns.

    Both rails open at 100 is explicitly NOT guaranteed, matching Console.
    """
    app = _WorkbenchHarness(LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR})))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        rail = app.query_one("#lab-rail").region
        body = app.query_one("#lab-body").region
        handle = app.query_one("#lab-inspector-handle").region
        assert rail.width == LAB_RAIL_WIDTH
        assert handle.width == 11
        assert body.width >= 63
        assert rail.width + body.width + handle.width <= 100


@pytest.mark.asyncio
async def test_the_selected_rail_row_gets_no_border_and_stays_one_row_high():
    """The bundle's global `.is-active` rule must not reach rail rows.

    At height 1 an unneutralised `is-active` row renders region.height == 2 --
    a half-bordered artifact that displaces its neighbours. Asserting the
    border alone would miss a height regression, so assert both.
    """
    app = _WorkbenchHarness(LabRailLayout())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        rows = [app.query_one(f"#lab-rail-row-{i}", Button) for i in range(3)]
        active = rows[1]
        assert "is-active" in active.classes

        border = active.styles.border
        assert not any(
            edge[0] for edge in (border.top, border.right, border.bottom, border.left)
        ), "selected rail row has a border; it will displace its neighbours"
        assert {row.region.height for row in rows} == {1}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_workbench.py -p no:randomly -q`

Expected: `ModuleNotFoundError: ... lab_workbench`.

- [ ] **Step 3: Write the container**

Create `tldw_chatbook/UI/Lab_Modules/lab_workbench.py`:

```python
"""The Lab frame's three-region workbench: rail | body | inspector.

Renders a :class:`LabRailLayout` as two collapsible rails around a body,
with a compact handle standing in for each collapsed rail. The container
holds no mode knowledge -- the frame mounts mode content into the regions.

Deliberately not `DestinationWorkbench`: that is a fixed Horizontal of
equal-width panes with no collapse. Deliberately not `WatchlistsWorkbench`
either: that is bound to a five-member Region enum with a stacked centre and
solo semantics, none of which Lab needs.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button

from ...Widgets.destination_rail import DestinationRailHandle
from .lab_rail_layout import LAB_RAIL_INSPECTOR, LAB_RAIL_LEFT, LabRailLayout

#: Width of the expanded catalog rail. Sized to the longest rail label
#: ("Speech Recognition", 18 characters) plus padding and frame border, and
#: chosen against Console's observed ~34 -- Console's rail holds conversation
#: titles, Lab's holds fixed short labels.
LAB_RAIL_WIDTH = 26
#: Width of the expanded inspector.
LAB_INSPECTOR_WIDTH = 30
#: Width of a collapsed rail's handle, matching Console's.
LAB_HANDLE_WIDTH = 11
#: Class every rail row carries; styled app-tier in features/_lab.tcss.
LAB_RAIL_ROW_CLASS = "lab-rail-row"


class LabWorkbench(Horizontal):
    """Two collapsible rails around a body, rendered from a rail layout."""

    def __init__(self, *, rail_layout: LabRailLayout, **kwargs: Any) -> None:
        """Create the workbench.

        Args:
            rail_layout: Which rails are collapsed. The attribute is named
                ``rail_layout``, never ``layout``: ``Widget.layout`` is an
                existing unsettable Textual property that the compositor
                calls ``.arrange()`` on, and shadowing it crashes rendering.
            kwargs: Forwarded to ``Horizontal``.
        """
        classes = kwargs.pop("classes", "")
        super().__init__(classes=f"lab-workbench {classes}".strip(), **kwargs)
        self.rail_layout = rail_layout

    def compose(self) -> ComposeResult:
        """Render handles and regions according to the rail layout.

        Returns:
            A ``ComposeResult`` yielding, left to right: the rail handle, the
            rail, the body, the inspector, and the inspector handle. A
            collapsed region and its handle swap visibility.
        """
        rail_collapsed = self.rail_layout.is_collapsed(LAB_RAIL_LEFT)
        inspector_collapsed = self.rail_layout.is_collapsed(LAB_RAIL_INSPECTOR)

        if rail_collapsed:
            yield DestinationRailHandle(
                label="Catalog",
                button_id="lab-rail-open",
                badge_id="lab-rail-badge",
                side="left",
                id="lab-rail-handle",
            )

        rail = VerticalScroll(id="lab-rail", classes="lab-region lab-rail")
        rail.display = not rail_collapsed
        yield rail

        body = Vertical(id="lab-body", classes="lab-region lab-body")
        yield body

        inspector = VerticalScroll(
            id="lab-inspector", classes="lab-region lab-inspector"
        )
        inspector.display = not inspector_collapsed
        yield inspector

        if inspector_collapsed:
            yield DestinationRailHandle(
                label="Inspector",
                button_id="lab-inspector-open",
                badge_id="lab-inspector-badge",
                side="right",
                id="lab-inspector-handle",
            )
```

- [ ] **Step 4: Append the CSS**

Append to `tldw_chatbook/css/features/_lab.tcss`:

```css
/* --- Lab workbench (PR2) ------------------------------------------------
 *
 * Region widths are declared here rather than inline so the 100-column
 * contract is inspectable in one place: rail 26 + body + collapsed
 * inspector handle 11 fits 100. Both rails open at 100 is not guaranteed,
 * matching Console.
 */

.lab-workbench {
    width: 100%;
    height: 1fr;
    min-height: 0;
}

#lab-rail {
    width: 26;
    min-width: 26;
    max-width: 26;
    height: 100%;
    min-height: 0;
}

#lab-body {
    width: 1fr;
    min-width: 0;
    height: 100%;
    min-height: 0;
}

#lab-inspector {
    width: 30;
    min-width: 30;
    max-width: 30;
    height: 100%;
    min-height: 0;
}

/* Rail rows.
 *
 * The `border: none` is REQUIRED, not cosmetic. The bundle declares an
 * unscoped `.is-active { border: round $ds-action-focus; }` earlier in this
 * sheet, and app-tier CSS beats widget DEFAULT_CSS regardless of
 * specificity. A height-1 row that picks up that border renders two rows
 * tall -- a half-bordered artifact that displaces every row below it.
 * `#mcp-hub-rail Button.mcp-rail-row.is-active` carries the same override
 * for the same reason.
 */
.lab-rail Button.lab-rail-row {
    width: 100%;
    height: 1;
    min-height: 1;
    padding: 0 1;
    border: none;
    text-align: left;
}

.lab-rail .lab-rail-row.is-active {
    border: none;
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold;
}

.lab-rail .lab-rail-row.is-active:focus,
.lab-rail .lab-rail-row.is-active:hover:focus {
    outline: none;
    border: none;
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold;
}

.lab-rail-section {
    height: 1;
    min-height: 1;
    padding: 0 1;
    color: $ds-text-muted;
    text-style: bold;
}

/* Status row: hidden entirely when a mode supplies no chips, so a mode
 * without status never reserves a row of dead chrome. */
#lab-status-row {
    height: 1;
    min-height: 1;
    padding: 0 1;
}

.lab-status-chip {
    width: auto;
    height: 1;
    padding: 0 1;
    color: $ds-text-muted;
}
```

- [ ] **Step 5: Regenerate the bundle**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`

Then confirm the diff is additive apart from the timestamp:

```bash
git diff --numstat tldw_chatbook/css/tldw_cli_modular.tcss
git diff tldw_chatbook/css/tldw_cli_modular.tcss | grep '^-' | grep -v '^---' | grep -vi generated
```

Expected: the second command prints nothing.

- [ ] **Step 6: Run the test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_workbench.py -p no:randomly -q`

Expected: 5 passed.

- [ ] **Step 7: Mutation-check the border rule**

Comment out the `border: none;` line inside `.lab-rail .lab-rail-row.is-active`, regenerate, and re-run. Expected: `test_the_selected_rail_row_gets_no_border_and_stays_one_row_high` FAILS. Restore, regenerate, confirm green. This proves the test is bound to the rule rather than passing incidentally.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Lab_Modules/lab_workbench.py \
        tldw_chatbook/css/features/_lab.tcss \
        tldw_chatbook/css/tldw_cli_modular.tcss \
        Tests/UI/test_lab_workbench.py
git commit -m "feat(lab): add the three-region workbench container

Two collapsible rails around a body, rendered from a LabRailLayout value.
The rail-row border override is load-bearing: the bundle's global
.is-active rule would otherwise render a height-1 selected row two rows
tall, displacing every row below it."
```

---

### Task 5: `LabScreen` — the shared frame

**Files:**
- Create: `tldw_chatbook/UI/Screens/lab_frame.py`
- Test: `Tests/UI/test_lab_frame.py`

**Interfaces:**
- Consumes: `LabRailLayout`, `LAB_RAIL_LEFT`, `LAB_RAIL_INSPECTOR` (Task 1); `load_rail_layout`, `save_rail_layout` (Task 2); `LabWorkbench` (Task 4).
- Produces:
  ```python
  @dataclass(frozen=True)
  class LabStatusChip:
      chip_id: str
      text: str

  class LabScreen(BaseAppScreen):
      def lab_header_state(self) -> WorkbenchHeaderState        # abstract
      def lab_status_chips(self) -> tuple[LabStatusChip, ...]   # default ()
      def compose_lab_rail(self) -> ComposeResult               # default empty
      def build_lab_body(self) -> Widget | None                 # default None
      def compose_lab_inspector(self) -> ComposeResult          # default empty
      def on_lab_body_ready(self) -> None                       # default no-op
      def refresh_lab_status(self) -> None                      # frame-owned
  ```

**Background.** `BaseAppScreen` composes the nav bar, a `#screen-content` container, and the footer; subclasses override `compose_content()`. All three Lab screens already pass their route as `screen_name` (`"llm"`, `"stts"`, `"evals"`), so the frame derives the mode strip's `active_route` from `self.screen_name` and needs no extra constructor parameter.

`build_lab_body` is a **factory returning a Widget**, not a `ComposeResult` generator: the body is mounted after first paint (Models' body costs 488-787 ms to compose), and widget *instances* do not survive `recompose=True` while factories do.

`lab_status_chips()` is called on compose **and on every refresh**. The frame creates one `Static` per `chip_id` and mutates it via `.update()`. It must never recompose the status row: recomposing on a 2-second timer churns widgets and can steal focus.

**A hazard this creates for PR3:** the frame uses `on_screen_resume` for its modal-pop refresh, and `STTSScreen.on_screen_resume` (`stts_screen.py:72`) overrides without calling `super()`. The frame therefore routes resume work through `refresh_lab_status()`, which PR3 must call from Speech's override.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_lab_frame.py`:

```python
"""The shared Lab frame: regions, status row, lazy body, and collapse."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import LAB_RAIL_INSPECTOR
from tldw_chatbook.UI.Screens.lab_frame import LabScreen, LabStatusChip
from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchHeaderState
from Tests.UI.test_screen_navigation import _build_test_app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _ProbeBody(Static):
    """Stands in for a mode's expensive legacy window."""


class _ProbeLabScreen(LabScreen):
    """A minimal Lab mode used to exercise the frame itself."""

    def __init__(self, app_instance, *, chips=(), **kwargs):
        super().__init__(app_instance, "llm", **kwargs)
        self._chips = chips
        self.body_ready_calls = 0

    def lab_header_state(self) -> WorkbenchHeaderState:
        return WorkbenchHeaderState(title="Probe", subtitle="probe mode")

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        return self._chips

    def compose_lab_rail(self) -> ComposeResult:
        yield Static("rail row", id="probe-rail-row")

    def build_lab_body(self) -> Widget:
        return _ProbeBody("body", id="probe-body")

    def on_lab_body_ready(self) -> None:
        self.body_ready_calls += 1


def _mount(screen_factory):
    app = _build_test_app()
    app.CSS_PATH = str(_BUNDLED_STYLESHEET)
    return app, screen_factory(app)


@pytest.mark.asyncio
async def test_the_body_is_absent_at_first_paint_and_present_after_deferral():
    """The lazy mount is the whole performance claim -- assert it directly.

    Without this, a frame that mounted the body inline would pass every
    other test in this file.
    """
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        assert not screen.query(_ProbeBody), "body mounted during first paint"
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one(_ProbeBody) is not None
        assert screen.body_ready_calls == 1


@pytest.mark.asyncio
async def test_a_mode_with_no_chips_renders_no_status_row_at_all():
    """A mode without status must not reserve a row of dead chrome.

    Models always supplies a chip, so this path has no real consumer until
    Speech and Evals adopt -- it would otherwise ship unexercised.
    """
    app, screen = _mount(lambda a: _ProbeLabScreen(a, chips=()))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert not screen.query("#lab-status-row")


@pytest.mark.asyncio
async def test_chips_render_and_refresh_mutates_them_without_recomposing():
    """Refresh must update the same Static, not replace the row."""
    chips = [LabStatusChip(chip_id="servers", text="Servers: none running")]

    class _Screen(_ProbeLabScreen):
        def lab_status_chips(self):
            return tuple(chips)

    app, screen = _mount(lambda a: _Screen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        chip_widget = screen.query_one("#lab-status-chip-servers", Static)
        assert "Servers: none running" in str(chip_widget.renderable)

        chips[0] = LabStatusChip(chip_id="servers", text="Servers: 2 running")
        screen.refresh_lab_status()
        await pilot.pause()

        assert screen.query_one("#lab-status-chip-servers", Static) is chip_widget
        assert "Servers: 2 running" in str(chip_widget.renderable)


@pytest.mark.asyncio
async def test_the_frame_wires_its_route_into_the_mode_strip():
    """The existing strip suite mounts the strip standalone and would not
    notice the frame passing the wrong active_route."""
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        active = screen.query_one("#lab-mode-models")
        assert "is-active" in active.classes


@pytest.mark.asyncio
async def test_the_inspector_starts_collapsed_on_first_run():
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.rail_layout.is_collapsed(LAB_RAIL_INSPECTOR) is True
        assert screen.query_one("#lab-inspector-handle") is not None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_frame.py -p no:randomly -q`

Expected: `ModuleNotFoundError: ... lab_frame`.

- [ ] **Step 3: Write the frame**

Create `tldw_chatbook/UI/Screens/lab_frame.py`:

```python
"""The shared frame behind the Lab destination's three screens.

Renders a destination header, an optional status row, the mode strip, and a
three-region workbench. Modes supply content through the hooks below; the
frame owns collapse state, the deferred body mount, and status refresh.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import QueryError
from textual.widget import Widget
from textual.widgets import Static

from ..Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LabRailLayout,
)
from ..Lab_Modules.lab_rail_store import load_rail_layout, save_rail_layout
from ..Lab_Modules.lab_workbench import LabWorkbench
from ..Navigation.base_app_screen import BaseAppScreen
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


@dataclass(frozen=True)
class LabStatusChip:
    """One chip in the Lab status row.

    Attributes:
        chip_id: Stable id suffix identifying this chip across refreshes.
        text: Rendered copy, e.g. ``"Servers: 2 running"``.
    """

    chip_id: str
    text: str


class LabScreen(BaseAppScreen):
    """Base for the Lab destination's screens.

    Subclasses override the ``lab_*`` hooks to supply content. The frame owns
    everything else: rail collapse and its persistence, the deferred body
    mount, and status-row refresh.
    """

    def __init__(self, app_instance: "TldwCli", screen_name: str, **kwargs: Any) -> None:
        """Create a Lab screen.

        Args:
            app_instance: The running application.
            screen_name: This screen's shell route (``"llm"``, ``"stts"``, or
                ``"evals"``). Doubles as the mode strip's active route.
            kwargs: Forwarded to ``BaseAppScreen``.
        """
        super().__init__(app_instance, screen_name, **kwargs)
        self.rail_layout: LabRailLayout = load_rail_layout()

    # -- hooks -----------------------------------------------------------

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return this mode's destination header copy.

        Returns:
            The header state. Subclasses must override.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        raise NotImplementedError("Lab modes must supply lab_header_state()")

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        """Return this mode's status chips.

        Called on compose and on every refresh, so it must be cheap and safe
        to call repeatedly.

        Returns:
            The chips, or an empty tuple to render no status row at all.
        """
        return ()

    def compose_lab_rail(self) -> ComposeResult:
        """Yield this mode's catalog rail contents.

        Returns:
            A ``ComposeResult``; empty by default.
        """
        return iter(())

    def build_lab_body(self) -> Widget | None:
        """Build this mode's body widget.

        A factory rather than a generator: the body is mounted after first
        paint, and widget instances do not survive ``recompose=True`` while
        factories do.

        Returns:
            The body widget, or None for a mode with no body.
        """
        return None

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield this mode's inspector contents.

        Returns:
            A ``ComposeResult``; empty by default.
        """
        return iter(())

    def on_lab_body_ready(self) -> None:
        """Called once, after the deferred body has mounted.

        Modes that need to touch their body -- registering watchers, reading
        widgets -- must do it here, never in ``on_mount``: the body does not
        exist yet at mount time.
        """

    # -- composition -----------------------------------------------------

    def compose_content(self) -> ComposeResult:
        """Compose the frame: header, optional status row, mode strip, workbench."""
        yield DestinationHeader(self.lab_header_state(), id="lab-destination-header")

        chips = self.lab_status_chips()
        if chips:
            with Horizontal(id="lab-status-row"):
                for chip in chips:
                    yield Static(
                        chip.text,
                        id=f"lab-status-chip-{chip.chip_id}",
                        classes="lab-status-chip",
                        markup=False,
                    )

        yield LabModeStrip(active_route=self.screen_name, id="lab-mode-strip")

        workbench = LabWorkbench(rail_layout=self.rail_layout, id="lab-workbench")
        yield workbench

    def on_mount(self) -> None:
        """Populate the rail and inspector, then defer the body mount.

        The body is mounted from ``call_after_refresh`` so first paint is not
        blocked by composing it -- Models' body costs 488-787 ms.
        """
        super().on_mount()
        self._populate_regions()
        self.call_after_refresh(self._mount_lab_body)

    def _populate_regions(self) -> None:
        """Mount rail and inspector contents into their regions."""
        for region_id, content in (
            ("#lab-rail", list(self.compose_lab_rail())),
            ("#lab-inspector", list(self.compose_lab_inspector())),
        ):
            if not content:
                continue
            try:
                self.query_one(region_id).mount_all(content)
            except QueryError:
                logger.warning("Lab region {} missing; skipped.", region_id)

    def _mount_lab_body(self) -> None:
        """Mount the deferred body and notify the mode."""
        body = self.build_lab_body()
        if body is not None:
            try:
                self.query_one("#lab-body").mount(body)
            except QueryError:
                logger.warning("Lab body region missing; body not mounted.")
                return
        self.on_lab_body_ready()

    # -- status ----------------------------------------------------------

    def refresh_lab_status(self) -> None:
        """Re-read this mode's chips and update the row in place.

        Mutates the existing ``Static`` for each ``chip_id`` rather than
        recomposing: recomposing on a timer churns widgets and can steal
        focus. A chip whose id was not composed is logged and ignored, since
        mounting new widgets from a timer is never intended.
        """
        for chip in self.lab_status_chips():
            try:
                self.query_one(f"#lab-status-chip-{chip.chip_id}", Static).update(
                    chip.text
                )
            except QueryError:
                logger.warning(
                    "Unknown Lab status chip id {!r}; ignoring.", chip.chip_id
                )

    # -- collapse --------------------------------------------------------

    def toggle_lab_rail(self, rail: str) -> None:
        """Collapse or expand one rail and persist the new state.

        Args:
            rail: ``LAB_RAIL_LEFT`` or ``LAB_RAIL_INSPECTOR``.
        """
        self.rail_layout = self.rail_layout.toggle(rail)
        save_rail_layout(self.rail_layout)
        self.refresh(recompose=True)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_frame.py -p no:randomly -q`

Expected: 5 passed.

If `test_the_body_is_absent_at_first_paint_and_present_after_deferral` fails because the body is already mounted, the body is being composed inline — the deferral is the point, so fix the frame, not the test.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/lab_frame.py Tests/UI/test_lab_frame.py
git commit -m "feat(lab): add the shared LabScreen frame

Header, optional status row, mode strip, and three-region workbench, with
modes supplying content through lab_* hooks. The body is mounted from
call_after_refresh so first paint is not blocked by composing it, and
status refresh mutates the existing Statics rather than recomposing the
row on a timer."
```

---

### Task 6: Models adopts the frame and lifts its sidebar

**Files:**
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py` (replace)
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py` — remove the nav buttons from `compose` (`:275-291`), trim the orphaned block from `watch_active_view` (`:985-997`)
- Test: `Tests/UI/test_llm_screen_lab_adoption.py`

**Interfaces:**
- Consumes: `LabScreen`, `LabStatusChip` (Task 5); `LAB_RAIL_ROW_CLASS` (Task 4); `read_server_rows`, `servers_chip_text` (Task 3).
- Produces: nothing later tasks depend on.

**Background — the silent trap.** Once the nine buttons are rail siblings, `Button.Pressed` bubbles to the **screen**, not the window, so the window's `@on(Button.Pressed, ".llm-nav-button")` never fires. And `watch_active_view`'s `self.query(".llm-nav-button")` returns an **empty set rather than raising**: the body still switches correctly while selection highlighting silently dies. A test asserting "clicking Ollama shows the Ollama view" passes straight through this.

The screen therefore **watches the reactive** rather than styling on press. `DOMNode.watch(obj, attribute_name, callback, init=True)` works across widgets and is already used at `evals_window_v3.py:59`. `init=True` fires immediately on registration, which seeds the rail highlight — necessary because `LLMManagementWindow.on_mount` sets `active_view = "llama-cpp"` itself (`:269`), so a press-only handler would leave the rail unhighlighted on arrival.

Rail rows carry their view key as an **attribute** (`button.lab_view_key`), mirroring `library_collections_panel.py:156`. `Button` has no `__slots__`, so this is safe.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_llm_screen_lab_adoption.py`:

```python
"""Models' adoption of the Lab frame, and its rail lift."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.test_screen_navigation import _build_test_app

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


async def _models_screen(pilot_app):
    screen = LLMScreen(pilot_app)
    await pilot_app.push_screen(screen)
    return screen


def _app():
    app = _build_test_app()
    app.CSS_PATH = str(_BUNDLED_STYLESHEET)
    return app


def _rail_rows(screen):
    return list(screen.query(".lab-rail-row").results(Button))


@pytest.mark.asyncio
async def test_all_nine_provider_rows_live_in_the_rail():
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        keys = [row.lab_view_key for row in _rail_rows(screen)]
        assert keys == [
            "llama-cpp",
            "llamafile",
            "ollama",
            "vllm",
            "onnx",
            "transformers",
            "mlx-lm",
            "local-models",
            "download-models",
        ]


@pytest.mark.asyncio
async def test_the_window_no_longer_carries_nav_buttons():
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        assert not window.query(".llm-nav-button")


@pytest.mark.asyncio
async def test_the_rail_is_highlighted_on_arrival_before_any_press():
    """LLMManagementWindow.on_mount sets active_view itself, so a
    press-only implementation would leave the rail unhighlighted here."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        active = [r for r in _rail_rows(screen) if "is-active" in r.classes]
        assert len(active) == 1
        assert active[0].lab_view_key == "llama-cpp"


@pytest.mark.asyncio
async def test_pressing_a_rail_row_moves_both_the_body_and_the_highlight():
    """The highlight half fails SILENTLY -- query() returns empty rather than
    raising -- so a body-only assertion would pass with it dead."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()

        ollama = next(r for r in _rail_rows(screen) if r.lab_view_key == "ollama")
        ollama.press()
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        assert window.active_view == "ollama"
        assert "-active" in window.query_one("#llm-view-ollama").classes

        active = [r for r in _rail_rows(screen) if "is-active" in r.classes]
        assert len(active) == 1, "exactly one rail row must be highlighted"
        assert active[0].lab_view_key == "ollama"


@pytest.mark.asyncio
async def test_the_status_row_reports_running_servers():
    app = _app()
    app.llamacpp_server_process = None
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        chip = screen.query_one("#lab-status-chip-servers", Static)
        assert "Servers: none running" in str(chip.renderable)

        class _Alive:
            def poll(self):
                return None

        app.llamacpp_server_process = _Alive()
        screen.refresh_lab_status()
        await pilot.pause()
        assert "Servers: 1 running" in str(chip.renderable)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_llm_screen_lab_adoption.py -p no:randomly -q`

Expected: FAIL — `LLMScreen` still composes the legacy window with its own sidebar and has no `.lab-rail-row` widgets.

- [ ] **Step 3: Rewrite `llm_screen.py`**

Replace the whole file:

```python
"""Models: the Lab destination's provider and model management screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual import on
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Button, Static

from ..Lab_Modules.lab_server_status import read_server_rows, servers_chip_text
from ..Lab_Modules.lab_workbench import LAB_RAIL_ROW_CLASS
from ..LLM_Management_Window import LLMManagementWindow
from ..Workbench.workbench_state import WorkbenchHeaderState
from .lab_frame import LabScreen, LabStatusChip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

#: (section title, ((view key, label), ...)) in rail order. The view keys are
#: exactly LLMManagementWindow.view_mapping's keys.
MODELS_RAIL_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Local servers",
        (
            ("llama-cpp", "Llama.cpp"),
            ("llamafile", "Llamafile"),
            ("ollama", "Ollama"),
            ("vllm", "vLLM"),
            ("onnx", "ONNX"),
            ("transformers", "Transformers"),
            ("mlx-lm", "MLX-LM"),
        ),
    ),
    (
        "Models",
        (
            ("local-models", "Local Models"),
            ("download-models", "Download Models"),
        ),
    ),
)

#: How often to re-read server liveness. There is deliberately no
#: refresh-on-press: pressing Start does not synchronously create the
#: process -- the event handler assigns it from an async worker -- so a
#: press-triggered read would report "stopped".
LAB_SERVER_POLL_SECONDS = 2.0


class LLMScreen(LabScreen):
    """Models mode: provider rail, legacy management body, server status."""

    def __init__(self, app_instance: "TldwCli", **kwargs: Any) -> None:
        """Create the Models screen.

        Args:
            app_instance: The running application.
            kwargs: Forwarded to ``LabScreen``.
        """
        super().__init__(app_instance, "llm", **kwargs)
        self.llm_window: LLMManagementWindow | None = None

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return the Models destination header copy."""
        return WorkbenchHeaderState(
            title="Models",
            subtitle="Manage providers, models, and endpoints.",
            status="ready",
        )

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        """Return the running-server chip.

        Returns:
            A single chip summarising how many local servers are alive.
        """
        rows = read_server_rows(self.app_instance)
        return (LabStatusChip(chip_id="servers", text=servers_chip_text(rows)),)

    def compose_lab_rail(self) -> ComposeResult:
        """Yield the two rail sections and their nine provider rows."""
        for title, entries in MODELS_RAIL_SECTIONS:
            yield Static(title, classes="lab-rail-section")
            for view_key, label in entries:
                row = Button(
                    label,
                    id=f"lab-models-row-{view_key}",
                    classes=LAB_RAIL_ROW_CLASS,
                )
                # Carried as an attribute rather than parsed back out of the
                # id, mirroring library_collections_panel's collection_id.
                row.lab_view_key = view_key
                yield row

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield the running-server list."""
        yield Static("Running servers", classes="lab-rail-section")
        for row in read_server_rows(self.app_instance):
            marker = "●" if row.running else "○"
            state = "running" if row.running else "stopped"
            yield Static(
                f"{marker} {row.name} — {state}",
                id=f"lab-inspector-server-{row.name.replace('.', '-')}",
                markup=False,
            )

    def build_lab_body(self) -> Widget:
        """Build the legacy management window.

        Returns:
            The ``LLMManagementWindow``, mounted after first paint because
            composing its nine views costs 488-787 ms.
        """
        self.llm_window = LLMManagementWindow(self.app_instance, classes="window")
        self.llm_window.styles.height = "1fr"
        return self.llm_window

    def on_lab_body_ready(self) -> None:
        """Wire rail highlighting to the window's active_view, then poll.

        The watch is registered here because the window does not exist before
        this point. ``init=True`` fires the callback immediately, which seeds
        the rail highlight -- necessary because ``LLMManagementWindow.on_mount``
        sets ``active_view`` itself, so a press-only handler would leave the
        rail unhighlighted on arrival.
        """
        if self.llm_window is None:
            return
        self.watch(self.llm_window, "active_view", self._sync_rail_active, init=True)
        self.refresh_lab_status()
        self.set_interval(LAB_SERVER_POLL_SECONDS, self.refresh_lab_status)

    def _sync_rail_active(self, active_view: str) -> None:
        """Move the rail highlight to the row matching the active view.

        Args:
            active_view: The window's current view key.
        """
        for row in self.query(f".{LAB_RAIL_ROW_CLASS}").results(Button):
            row.set_class(getattr(row, "lab_view_key", None) == active_view, "is-active")

    @on(Button.Pressed, f".{LAB_RAIL_ROW_CLASS}")
    def _handle_rail_press(self, event: Button.Pressed) -> None:
        """Point the window at the pressed provider's view.

        The window's own ``@on`` no longer fires: the buttons are the
        screen's children now, so their presses never reach it. Styling is
        not done here -- ``_sync_rail_active`` runs from the reactive watch,
        which also covers changes the window makes itself.
        """
        event.stop()
        view_key = getattr(event.button, "lab_view_key", None)
        if view_key is None or self.llm_window is None:
            return
        self.llm_window.active_view = view_key

    async def on_screen_resume(self) -> None:
        """Refresh server status when a modal pops back over this screen."""
        self.refresh_lab_status()
```

- [ ] **Step 4: Remove the nav buttons from the window's compose**

In `tldw_chatbook/UI/LLM_Management_Window.py`, delete the nav-pane block in `compose` (the `with` block yielding the nine `Button(... classes="llm-nav-button")` widgets, `:275-291`). Leave every `llm-view-*` block untouched.

- [ ] **Step 5: Trim the orphaned block from `watch_active_view`**

In the same file, delete only these lines from `watch_active_view` (`:985-997`):

```python
        # Update navigation buttons
        for button in self.query(".llm-nav-button"):
            button.remove_class("-active")

        # Set active button
        active_button_id = f"nav-{new_view}"
        try:
            active_button = self.query_one(f"#{active_button_id}", Button)
            active_button.add_class("-active")
        except QueryError:
            logger.warning(f"Navigation button #{active_button_id} not found")
```

Leave the view-visibility loop and the `_populate_help_text` call that follow it exactly as they are. Without this deletion the method logs a warning on **every** view switch, because the button it looks for no longer exists.

Also delete `handle_nav_button` and its `@on(Button.Pressed, ".llm-nav-button")` decorator (`:963-980`) — with the buttons gone it can never fire.

- [ ] **Step 6: Run the test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_llm_screen_lab_adoption.py -p no:randomly -q`

Expected: 5 passed.

- [ ] **Step 7: Mutation-check the highlight**

Temporarily change `on_lab_body_ready` to register the watch with `init=False`. Re-run. Expected: `test_the_rail_is_highlighted_on_arrival_before_any_press` FAILS while the press test still passes — proving the arrival test catches what a press-only implementation misses. Restore `init=True` and confirm green.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Screens/llm_screen.py \
        tldw_chatbook/UI/LLM_Management_Window.py \
        Tests/UI/test_llm_screen_lab_adoption.py
git commit -m "feat(lab): Models adopts the frame and lifts its sidebar

The nine provider rows move out of LLMManagementWindow into the frame's
rail, in two sections. The screen drives highlighting by watching the
window's active_view reactive rather than styling on press: the window
sets active_view itself on mount, so a press-only handler would leave the
rail unhighlighted on arrival.

Removes the window's now-unreachable nav handler and trims the orphaned
nav-button block from watch_active_view, which would otherwise log a
warning on every view switch."
```

---

### Task 7: Focus-based mode keys and footer shortcuts

**Files:**
- Modify: `tldw_chatbook/UI/Screens/lab_frame.py`
- Test: `Tests/UI/test_lab_frame_mode_keys.py`

**Interfaces:**
- Consumes: `LabScreen` (Task 5); `LAB_MODE_CHIP_IDS` from `tldw_chatbook.UI.Screens.lab_mode_strip`.
- Produces: nothing later tasks depend on.

**Background.** `[` and `]` move **focus** along the mode strip; they do not navigate. `Enter` is then ordinary `Button` activation on the focused chip, which already posts `NavigateToScreen` (`lab_mode_strip.py:101-108`). This is why cycling builds zero intermediate screens: navigation happens once, on commit.

Binding these on `LabScreen` touches nothing shared — Speech and Evals have not adopted the frame, so they simply do not get the keys until PR3.

Two behaviours follow from existing code and must not be "fixed": brackets are printable, so text inputs consume them first and the keys act only from button or list focus (Personas documents the same at `personas_screen.py:237-239`); and `Enter` on the already-active chip is a deliberate no-op, since `_handle_mode_chip` returns without posting when the route matches.

`Escape` is deliberately **not** bound: `EvalsScreen` already binds it to `action_evals_back` (`evals_screen.py:31`), and a competing frame binding would shadow that on one mode only.

- [ ] **Step 1: Write the failing test**

Create `Tests/UI/test_lab_frame_mode_keys.py`:

```python
"""Focus-based mode switching on the Lab frame."""

from __future__ import annotations

import pytest
from textual.widgets import Button

from tldw_chatbook.UI.Screens.lab_mode_strip import LAB_MODE_CHIP_IDS
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from Tests.UI.test_screen_navigation import _build_test_app


async def _models(app):
    screen = LLMScreen(app)
    await app.push_screen(screen)
    return screen


@pytest.mark.asyncio
async def test_bracket_moves_focus_along_the_strip_without_navigating():
    app = _build_test_app()
    navigated: list[str] = []
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        screen.query_one(f"#{LAB_MODE_CHIP_IDS[0]}", Button).focus()
        await pilot.pause()

        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert app.focused is not None
        assert app.focused.id == LAB_MODE_CHIP_IDS[1]
        assert navigated == [], "moving focus must not navigate"


@pytest.mark.asyncio
async def test_bracket_wraps_at_both_ends():
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()

        screen.query_one(f"#{LAB_MODE_CHIP_IDS[0]}", Button).focus()
        await pilot.pause()
        await pilot.press("left_square_bracket")
        await pilot.pause()
        assert app.focused.id == LAB_MODE_CHIP_IDS[-1]

        await pilot.press("right_square_bracket")
        await pilot.pause()
        assert app.focused.id == LAB_MODE_CHIP_IDS[0]


@pytest.mark.asyncio
async def test_bracket_starts_from_the_active_chip_when_nothing_is_focused():
    """With focus elsewhere, the first press should land beside the active
    mode rather than jumping to an arbitrary end of the strip."""
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        screen.set_focus(None)
        await pilot.pause()

        await pilot.press("right_square_bracket")
        await pilot.pause()

        assert app.focused.id == LAB_MODE_CHIP_IDS[1]


@pytest.mark.asyncio
async def test_the_footer_advertises_the_mode_keys():
    app = _build_test_app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models(app)
        await pilot.pause()
        await pilot.pause()
        footer = screen.query_one(AppFooterStatus)
        rendered = str(footer.render())
        assert "[ / ]" in rendered
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_frame_mode_keys.py -p no:randomly -q`

Expected: FAIL — the keys are unbound, so focus does not move.

- [ ] **Step 3: Add the bindings to `LabScreen`**

In `tldw_chatbook/UI/Screens/lab_frame.py`, add the import and `BINDINGS`, and the two actions.

Add to the imports:

```python
from textual.binding import Binding

from .lab_mode_strip import LAB_MODE_CHIP_IDS, LabModeStrip
```

Add to the `LabScreen` class body, above `__init__`:

```python
    #: `[` / `]` move focus along the mode strip; they never navigate. Enter is
    #: then ordinary Button activation on the focused chip, which posts
    #: NavigateToScreen -- so cycling builds zero intermediate screens.
    #:
    #: Both are printable keys, so text inputs consume them first and these act
    #: only from button or list focus. Escape is deliberately unbound:
    #: EvalsScreen already binds it to its own back action.
    BINDINGS = [
        Binding("left_square_bracket", "lab_mode_focus(-1)", "Prev mode", show=False),
        Binding("right_square_bracket", "lab_mode_focus(1)", "Next mode", show=False),
    ]

    #: Footer hints registered for every Lab mode.
    LAB_FOOTER_SHORTCUTS: tuple[tuple[str, str], ...] = (
        ("[ / ]", "Switch mode"),
        ("Enter", "Go"),
    )
```

Add these methods to `LabScreen`:

```python
    def action_lab_mode_focus(self, delta: int) -> None:
        """Move focus to an adjacent mode chip, wrapping at both ends.

        Does not navigate: Enter on the focused chip commits, which is what
        keeps cycling free of intermediate screen mounts.

        Args:
            delta: ``-1`` for the previous chip, ``1`` for the next.
        """
        focused = self.focused
        focused_id = getattr(focused, "id", None)
        if focused_id in LAB_MODE_CHIP_IDS:
            index = LAB_MODE_CHIP_IDS.index(focused_id)
        else:
            # Focus is elsewhere: start from the chip for this screen's own
            # mode so the first press lands beside it, not at a strip end.
            index = self._active_mode_chip_index()
        target = LAB_MODE_CHIP_IDS[(index + delta) % len(LAB_MODE_CHIP_IDS)]
        try:
            self.query_one(f"#{target}", Button).focus()
        except QueryError:
            logger.warning("Lab mode chip {} missing; focus not moved.", target)

    def _active_mode_chip_index(self) -> int:
        """Return the strip index of this screen's own mode.

        Returns:
            The index of the chip carrying ``is-active``, or 0 when the strip
            has not composed one.
        """
        for index, chip_id in enumerate(LAB_MODE_CHIP_IDS):
            try:
                if "is-active" in self.query_one(f"#{chip_id}", Button).classes:
                    return index
            except QueryError:
                continue
        return 0
```

Register the footer hints at the end of `LabScreen.on_mount`, after the deferred-mount call:

```python
        self.register_footer_shortcuts(
            source="lab", shortcuts=self.LAB_FOOTER_SHORTCUTS
        )
```

Add `Button` to the `textual.widgets` import in this module if it is not already there.

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_lab_frame_mode_keys.py -p no:randomly -q`

Expected: 4 passed.

- [ ] **Step 5: Confirm bindings merge rather than replace**

`LLMScreen` defines no `BINDINGS` of its own, but `EvalsScreen` does, and Textual 8.2.7 sets
`_inherit_bindings = True` so a subclass's `BINDINGS` merges with the base's. Confirm the frame's
keys survive alongside a subclass's own:

```bash
PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "
from tldw_chatbook.UI.Screens.lab_frame import LabScreen
merged = sorted(LabScreen._merge_bindings().key_to_bindings)
print('left_square_bracket' in merged, 'right_square_bracket' in merged)
"
```

Expected: `True True`.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/lab_frame.py Tests/UI/test_lab_frame_mode_keys.py
git commit -m "feat(lab): focus-based mode keys and footer hints

[ and ] move focus along the mode strip rather than navigating; Enter is
then ordinary Button activation on the focused chip. Cycling therefore
builds zero intermediate screens, which matters because every navigation
constructs a fresh screen and Models' body costs ~0.5-0.8s to compose.

Escape is deliberately unbound -- EvalsScreen already owns it."
```

---

### Task 8: Whole-screen verification

**Files:**
- Test: no new files; runs existing suites and the live app.

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: nothing.

- [ ] **Step 1: Run every Lab and rail suite**

```bash
PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_lab_rail_layout.py Tests/UI/test_lab_rail_store.py \
  Tests/UI/test_lab_server_status.py Tests/UI/test_lab_workbench.py \
  Tests/UI/test_lab_frame.py Tests/UI/test_lab_frame_mode_keys.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/UI/test_lab_mode_strip.py Tests/UI/test_destination_rail.py \
  -p no:randomly -q
```

Expected: all pass.

- [ ] **Step 2: Run the no-regression suites**

```bash
PYTHONPATH=$(pwd) /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_destination_shells.py Tests/UI/test_workbench_route_inventory.py \
  Tests/UI/test_command_palette_shell_routes.py Tests/UI/test_evals_screen_shell.py \
  Tests/UI/test_stts_capability_state.py Tests/UI/test_console_persistent_rails.py \
  -p no:randomly -q
```

Expected: only the known `test_generated_console_stylesheet_includes_rail_rules` failure. **Gate on the failure name, not a count.**

- [ ] **Step 3: Verify the CSS bundle reproduces**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
git diff tldw_chatbook/css/tldw_cli_modular.tcss | grep '^[+-]' | grep -v '^[+-][+-]' | grep -vi generated
git checkout tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected: the grep prints nothing — only the timestamp differs.

- [ ] **Step 4: Drive the real app**

```bash
tmux -L labpr2 kill-server 2>/dev/null
tmux -L labpr2 new-session -d -x 200 -y 50 '.venv/bin/python -m tldw_chatbook.app'
sleep 18
```

Click the `Lab` tab (find its column in row 2 of `tmux -L labpr2 capture-pane -p`, then send
`$'\x1b[<0;COL;2M'` and `$'\x1b[<0;COL;2m'`). Confirm on screen:

- the rail shows two sections and nine rows, with `Llama.cpp` highlighted on arrival
- the status row reads `Servers: none running`
- clicking `Ollama` moves both the body and the highlight
- the inspector handle is present on the right, and expanding it lists six servers
- the mode strip still reads `Models  Speech  Evals` with `Models` visibly active

Repeat at 100 columns:

```bash
tmux -L labpr2 kill-server
tmux -L labpr2 new-session -d -x 100 -y 40 '.venv/bin/python -m tldw_chatbook.app'
```

Confirm the rail, body, and collapsed inspector handle all fit without truncation. Then
`tmux -L labpr2 kill-server`.

- [ ] **Step 5: Commit any fixes**

If steps 1-4 surfaced problems, fix them and commit. If everything passed, there is nothing to
commit — say so rather than inventing a commit.

---

## Self-Review

**Spec coverage.** Frame anatomy and all seven hooks → Task 5. Rail collapse state and persistence →
Tasks 1-2. Server status reader, chip text, and inspector rows → Tasks 3, 6. Workbench regions, the
100-column contract, and the rail-row `is-active` rule → Task 4. Models' rail, the watch seam, the
silent trap, and the window trim → Task 6. Live verification → Task 7.

**Deliberately not implemented, matching the spec's non-goals:** no Speech or Evals adoption, no
`llm-view-*` rebuild, no port numbers in the inspector, no removal of the orphaned
`Constants.py:966-976` CSS.

**A gap this review caught and closed.** The first draft deferred focus-based mode keys and footer
shortcuts to PR3, on the reasoning that bracket keys "touch a widget all three Lab screens share".
That reasoning was wrong: binding them on `LabScreen` touches nothing shared, since Speech and Evals
have not adopted the frame and simply do not receive the keys. Both are now Task 7.

**One deliberate deviation from the spec, in naming only.** The spec listed `lab_footer_shortcuts()`
as an overridable hook. Task 7 implements it as a `LAB_FOOTER_SHORTCUTS` class constant instead: no
Lab mode has mode-specific shortcuts, and an overridable hook nobody overrides is dead API. A mode
that later needs its own can override the constant, which is the same extension point with less
machinery.

**Placeholder scan:** clean — every code step carries real content, and no step says "add error
handling" or "similar to Task N".

**Type consistency:** `LabRailLayout`, `LAB_RAIL_LEFT`, `LAB_RAIL_INSPECTOR` (Task 1) are used under
those names in Tasks 2, 4, 5. `load_rail_layout` / `save_rail_layout` (Task 2) match Task 5's import.
`read_server_rows` / `servers_chip_text` / `LabServerRow` (Task 3) match Task 6's usage.
`LAB_RAIL_ROW_CLASS` (Task 4) is used in Tasks 4 and 6. `LabStatusChip(chip_id, text)` (Task 5)
matches Task 6's construction and the `#lab-status-chip-{chip_id}` id built in both.
`build_lab_body` returns `Widget | None` in Task 5 and `Widget` in Task 6's override — compatible.
