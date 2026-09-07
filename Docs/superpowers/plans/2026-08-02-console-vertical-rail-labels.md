# Console Vertical Rail Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render the collapsed Console Context and Inspector handles top-to-bottom in stable three-cell rails without changing expanded headers, tooltips, persistence, or Personas handles.

**Architecture:** Extend the shared `ConsoleRailHandle` with an explicit opt-in vertical presentation flag whose default preserves every existing caller. The widget normalizes and stacks visible text and owns child sizing; `ChatScreen` opts in at its two Console call sites and pins the parent widths to the widget's shared width constant. Component TCSS remains authoritative and is rebuilt into the committed bundle.

**Tech Stack:** Python 3.11+, Textual widgets/TCSS, pytest/pytest-asyncio

---

## File Map

- `tldw_chatbook/Widgets/Console/console_rail_handle.py`: shared opt-in text transformation and child geometry.
- `tldw_chatbook/UI/Screens/chat_screen.py`: Console-only opt-in and parent handle widths.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`: authoritative vertical-handle classes.
- `tldw_chatbook/css/tldw_cli_modular.tcss`: generated production stylesheet.
- `Tests/UI/test_console_rail_handle.py`: focused widget behavior and mounted geometry without the full application harness.
- `Tests/UI/test_console_persistent_rails.py`: existing Console integration expectations and multiline-aware helpers.
- `Tests/UI/test_personas_workbench.py`: unchanged-default regression command only; no planned edit.
- `backlog/tasks/task-1335 - Stack-collapsed-Console-rail-labels-vertically.md`: plan, checked acceptance criteria, and implementation notes.

### Task 1: Pin the opt-in widget contract

**Files:**
- Create: `Tests/UI/test_console_rail_handle.py`
- Modify: `tldw_chatbook/Widgets/Console/console_rail_handle.py`

- [ ] **Step 1: Write failing pure rendering tests**

Create focused tests that instantiate horizontal and vertical handles directly:

```python
def test_vertical_console_labels_stack_without_direction_glyphs() -> None:
    context = _handle(
        label=CONSOLE_RAIL_CONTEXT_LABEL,
        side="left",
        vertical=True,
    )
    inspector = _handle(
        label=CONSOLE_RAIL_INSPECTOR_LABEL,
        side="right",
        vertical=True,
    )

    assert context._display_label() == "C\no\nn\nt\ne\nx\nt"
    assert inspector._display_label() == "I\nn\ns\np\ne\nc\nt\no\nr"


def test_horizontal_default_preserves_existing_labels() -> None:
    assert _handle(
        label=CONSOLE_RAIL_CONTEXT_LABEL,
        side="left",
    )._display_label() == CONSOLE_RAIL_CONTEXT_LABEL
    assert _handle(
        label=CONSOLE_RAIL_INSPECTOR_LABEL,
        side="right",
    )._display_label() == "Inspector"
```

Also pin whitespace collapsing and vertical badge display:

```python
def test_vertical_text_normalizes_whitespace_and_stacks_badges() -> None:
    handle = _handle(
        label="  Review\n queue  ",
        badge="1 approval",
        side="right",
        vertical=True,
    )

    assert handle._display_label() == "R\ne\nv\ni\ne\nw\n \nq\nu\ne\nu\ne"
    assert handle._display_badge() == "1\n \na\np\np\nr"
```

- [ ] **Step 2: Run the pure tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -c pyproject.toml -q \
  Tests/UI/test_console_rail_handle.py -k 'vertical or horizontal_default'
```

Expected: FAIL because `ConsoleRailHandle` does not accept the vertical opt-in and still returns horizontal text.

- [ ] **Step 3: Implement the minimal widget opt-in**

In `console_rail_handle.py`:

```python
class ConsoleRailHandle(Vertical):
    VERTICAL_WIDTH = 3
    VERTICAL_CONTENT_WIDTH = 1

    def __init__(
        self,
        *,
        label: str,
        badge: str = "",
        button_id: str,
        badge_id: str,
        side: str,
        vertical: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.label = label
        self.badge = badge
        self.button_id = button_id
        self.badge_id = badge_id
        self.side = side
        self.vertical = vertical
        self.add_class("console-rail-handle")
        self.add_class(f"console-rail-handle-{side}")
        if vertical:
            self.add_class("console-rail-handle-vertical")
```

Keep the existing horizontal return values. For the opt-in path, first normalize
with `" ".join(text.split())`; compare that result with equivalently normalized
Console constants and remove only their known leading/trailing direction glyph,
then return `"\n".join(text)`. Apply the same whitespace normalization and
stacking to the existing compact badge text. In `compose()`, add the vertical
button and badge classes, use the one-cell `VERTICAL_CONTENT_WIDTH` for children
inside the three-cell framed outer handle, and let the button consume `1fr` so a
badge can remain below it. Add a case that combines a known direction glyph with
surrounding whitespace to pin this normalization-before-glyph-removal order.

- [ ] **Step 4: Run the pure tests and verify GREEN**

Run the Step 2 command.

Expected: all selected tests PASS.

- [ ] **Step 5: Commit the widget contract**

```bash
git add Tests/UI/test_console_rail_handle.py \
  tldw_chatbook/Widgets/Console/console_rail_handle.py
git commit -m "feat(console): support vertical collapsed rail labels"
```

### Task 2: Opt Console into stable three-cell handles

**Files:**
- Modify: `Tests/UI/test_console_persistent_rails.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`

- [ ] **Step 1: Update the integration expectations first**

Change the existing first-start and collapse tests to expect:

```python
assert right_handle.region.width == ConsoleRailHandle.VERTICAL_WIDTH
assert str(open_button.label) == "I\nn\ns\np\ne\nc\nt\no\nr"
assert open_button.tooltip == "Open Inspector rail"
```

and:

```python
assert left_handle.region.width == ConsoleRailHandle.VERTICAL_WIDTH
assert str(open_button.label) == "C\no\nn\nt\ne\nx\nt"
assert open_button.tooltip == "Open Context rail"
```

Make helper checks multiline-aware by comparing the longest `splitlines()` row
to the available width. Normalize badge text with `text.replace("\n", "")`
inside `_wait_for_badge`, so existing semantic badge assertions remain explicit.
Replace right-handle assumptions about a three-row horizontal button with the
full-height vertical containment contract.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -c pyproject.toml -q \
  Tests/UI/test_console_persistent_rails.py::test_console_first_start_renders_left_rail_and_right_handle \
  Tests/UI/test_console_persistent_rails.py::test_console_context_rail_collapse_hides_left_rail_and_expands_main_column
```

Expected: vertical label/width assertions FAIL on the old Console call sites. If
the committed baseline's unrelated application-harness database isolation error
occurs first, record it as pre-existing and continue using the focused mounted
widget test from Task 1 for the red/green geometry proof.

- [ ] **Step 3: Opt in only the two Console call sites**

For both `ConsoleRailHandle` instances in `ChatScreen.compose()`, pass
`vertical=True` and replace the inline `13` / `11` parent width triplets with:

```python
handle_width = ConsoleRailHandle.VERTICAL_WIDTH
handle.styles.width = handle_width
handle.styles.min_width = handle_width
handle.styles.max_width = handle_width
```

Do not modify the two Personas call sites.

- [ ] **Step 4: Run focused widget and default-consumer tests**

Run:

```bash
.venv/bin/python -m pytest -c pyproject.toml -q \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_library_rail_collapses_and_reopens_from_handle \
  Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_inspector_rail_collapses_and_reopens_from_handle
```

Expected: PASS, including horizontal Personas handles.

- [ ] **Step 5: Commit the Console opt-in**

```bash
git add Tests/UI/test_console_persistent_rails.py \
  tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "feat(console): stack collapsed rail labels vertically"
```

### Task 3: Add production geometry and rebuild TCSS

**Files:**
- Modify: `Tests/UI/test_console_rail_handle.py`
- Modify: `Tests/UI/test_console_persistent_rails.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`

- [ ] **Step 1: Write failing mounted geometry and stylesheet assertions**

Mount vertical left/right handles in a minimal Textual `App` that loads the
production stylesheet. Assert each outer handle is three cells wide, each stacked
child is one cell wide, the left handle's solid frame contains that one-cell
content column, the handle uses full available height, the button leaves room for
a badge, and the full tooltips remain horizontal. Extend the existing stylesheet
test to require:

```python
vertical_handle = _css_block(css, ".console-rail-handle-vertical")
vertical_button = _css_block(css, ".console-rail-handle-button-vertical")
assert "width: 3;" in vertical_handle
assert "min-width: 3;" in vertical_handle
assert "max-width: 3;" in vertical_handle
assert "height: 100%;" in vertical_handle
assert "width: 1;" in vertical_button
assert "height: 1fr;" in vertical_button
```

- [ ] **Step 2: Run the mounted/style tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -c pyproject.toml -q \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_console_persistent_rails.py::test_generated_console_stylesheet_includes_rail_rules
```

Expected: FAIL because the vertical TCSS classes do not exist.

- [ ] **Step 3: Add the authoritative component rules**

Add after the existing side-specific handle/button blocks:

```tcss
.console-rail-handle-vertical {
    width: 3;
    min-width: 3;
    max-width: 3;
    height: 100%;
    min-height: 20;
    max-height: 100%;
}

.console-rail-handle-button-vertical {
    width: 1;
    min-width: 1;
    max-width: 1;
    height: 1fr;
    min-height: 7;
    max-height: 100%;
}

.console-rail-handle-badge-vertical {
    width: 1;
    min-width: 1;
    max-width: 1;
}
```

- [ ] **Step 4: Rebuild the bundled production stylesheet**

Run:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
```

Expected: `tldw_chatbook/css/tldw_cli_modular.tcss` is regenerated and contains
the same three vertical selectors exactly once.

- [ ] **Step 5: Run the style and mounted tests and verify GREEN**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 6: Commit the production geometry**

```bash
git add Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_console_persistent_rails.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss
git commit -m "style(console): narrow vertical rail handles"
```

### Task 4: Verify, document, and close TASK-1335

**Files:**
- Modify: `backlog/tasks/task-1335 - Stack-collapsed-Console-rail-labels-vertically.md`

- [ ] **Step 1: Run focused verification**

```bash
.venv/bin/python -m pytest -c pyproject.toml -q \
  Tests/UI/test_console_rail_handle.py \
  Tests/UI/test_console_persistent_rails.py::test_generated_console_stylesheet_includes_rail_rules \
  Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_library_rail_collapses_and_reopens_from_handle \
  Tests/UI/test_personas_workbench.py::TestWorkbenchShell::test_inspector_rail_collapses_and_reopens_from_handle
.venv/bin/python -m pytest -c pyproject.toml -q Tests/UI/test_css_build_integrity.py
git diff --check
```

Expected: all selected checks PASS and `git diff --check` emits no output.

- [ ] **Step 2: Attempt the existing mounted Console integration tests**

Run the Task 2 Step 2 command. Expected: PASS in a healthy application harness;
if the previously observed committed-baseline readonly-database isolation failure
still blocks app construction, record the exact failure rather than weakening or
expanding this presentation task.

- [ ] **Step 3: Self-review scope and behavior**

Confirm the diff contains no changes to persistence, responsive thresholds,
expanded headers, Personas call sites, or full tooltip strings. Confirm generated
TCSS is the only mechanical bulk change.

- [ ] **Step 4: Complete task documentation**

Check all acceptance criteria that have passing evidence and add concise
Implementation Notes covering the opt-in widget, Console-only call sites,
three-cell TCSS, tests, ADR decision, and any pre-existing harness blocker. Set
the task to Done only if every Definition of Done item is satisfied.

- [ ] **Step 5: Commit the task closeout**

```bash
git add 'backlog/tasks/task-1335 - Stack-collapsed-Console-rail-labels-vertically.md'
git commit -m "docs(console): close vertical rail label task"
```
