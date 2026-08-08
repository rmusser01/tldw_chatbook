# Task 4 Report — Prompt Editor Shell and Action Groups

## Scope

Implemented only the TASK-202 editor-shell layout, semantic action grouping,
and compiled stylesheet update. No Task 5 clipboard or delete-host wiring was
changed.

## RED evidence

Before production edits, ran:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -k "geometry or copy or action_group" -q --tb=no --show-capture=no --disable-warnings
```

Result:

```text
15 failed, 5 passed, 67 deselected, 2 warnings in 27.73s
```

The failures correctly identified the missing editor-shell/content/action
regions, missing semantic action groups, old `Copy text` label, and the
intentionally unwired Task 5 copy behavior.

## GREEN evidence

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Widgets/Library/library_prompts_canvas.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
```

Result: Python compilation succeeded, the generated bundle rebuilt, and the
bundle-sync check reported `CSS bundle reproduces from its source modules.`

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -k "geometry or action_group or copy_and_duplicate_relabeled" -q --tb=short --show-capture=no --disable-warnings
```

Result:

```text
11 passed, 76 deselected, 2 warnings in 16.92s
```

This is the four-size (80x24, 100x30, 140x40, 200x50), normal/conflict
geometry evidence. It verifies that the action area is nonzero and contained,
the content area alone scrolls, actions do not cover the Author field after
scrolling, and every required action stays on screen. The action-group and
Copy Markdown label contracts also passed.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_prompt_delete_confirmation_modal.py Tests/UI/test_css_build_integrity.py Tests/UI/test_css_bundle_sync_guard.py -q --tb=short --show-capture=no --disable-warnings
```

Result:

```text
30 passed, 1 warning in 2.19s
```

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Widgets/Library/library_prompts_canvas.py
git diff --check
```

Result: `All checks passed!`; `git diff --check` produced no output.

Task 5 remains intentionally RED:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -k "library_prompt_copy_" -q --tb=no --show-capture=no --disable-warnings
```

Result:

```text
4 failed, 83 deselected, 2 warnings in 7.60s
```

Those failures are the absent live-copy handler/unavailable-clipboard behavior
assigned to Task 5, not a layout regression.

## Changes

- `library_prompts_canvas.py` now composes `#library-prompt-editor-shell` with
  one `VerticalScroll` content owner (`#library-prompt-editor-content`) and a
  separate auto-height `#library-prompt-editor-actions` region.
- Normal actions are grouped structurally, in source and focus order, as
  `#library-prompt-actions-primary`, `#library-prompt-actions-content`, and
  `#library-prompt-actions-lifecycle`. Conflict Save as new/Reload actions
  occupy the same persistent region.
- Save uses the existing `console-action-primary` treatment; Delete retains the
  existing `library-media-action-danger` treatment. The stable action IDs are
  unchanged, while the visible Copy label is now `Copy Markdown`.
- The action groups intentionally stack at every width. Textual has no safe
  CSS media-query reflow here; the always-stacked layout preserves source order
  and full labels without clipping at the narrow target sizes.
- `_agentic_terminal.tcss` gives the bounded content owner `height: 1fr` /
  `min-height: 0` and the action area `height: auto`; the generated bundle was
  rebuilt only through `build_css.py`.

## Files changed

- `tldw_chatbook/Widgets/Library/library_prompts_canvas.py`
- `tldw_chatbook/css/components/_agentic_terminal.tcss`
- `tldw_chatbook/css/tldw_cli_modular.tcss` (generated)
- `.superpowers/sdd/2026-08-02-task-202-library-prompt-editor-actions/task-4-report.md`

## Risks / handoff

- The Task 5 handler is deliberately absent: pressing Copy Markdown still has
  no clipboard effect until the host wiring lands.
- Delete remains the existing host behavior until Task 5 changes it to the
  confirmation flow; no modal or host source was edited here.

## Review round 1 correction

### Root cause

The first Task 4 implementation correctly introduced
`#library-prompt-editor-content` as the outer `VerticalScroll`, but its
structured `PromptBlockEditor` child still composed the standalone
`#prompt-editor-scroll` `VerticalScroll`. That created a nested scroll owner.
The pre-existing duplicate-order test also still assumed Copy, Duplicate, and
Delete shared one direct toolbar parent, which no longer holds after semantic
grouping.

### Correction RED evidence

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -k "geometry_keeps_actions or duplicate_button_between" -q --tb=short --show-capture=no --disable-warnings
```

Result:

```text
8 failed, 1 passed, 78 deselected, 2 warnings in 16.86s
```

The eight four-size normal/conflict cases each found the nested
`VerticalScroll(id='prompt-editor-scroll')`; the corrected descendant-order
test passed.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_prompt_block_editor.py -k "embedded_editor_uses_a_plain" -q --tb=short --show-capture=no --disable-warnings
```

Result: one expected failure. `PromptBlockEditor(..., embedded=True)` reached
the base widget with an unsupported keyword argument, proving the explicit
public embedded-mode contract did not yet exist.

### Correction GREEN evidence

`PromptBlockEditor` now has an explicit `embedded` mode. The default standalone
mode retains its `#prompt-editor-scroll` scroll owner. Only the Library Prompt
editor opts into embedded mode; it renders the natural-height
`#prompt-editor-body` `Vertical`, so its parent owns the sole vertical scroll.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_prompt_block_editor.py -k "embedded_editor_uses_a_plain or genuinely_wide" -q --tb=short --show-capture=no --disable-warnings
```

Result:

```text
2 passed, 22 deselected in 0.83s
```

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -k "geometry or action_group or copy_and_duplicate_relabeled or duplicate_button_between" -q --tb=short --show-capture=no --disable-warnings
```

Result:

```text
12 passed, 75 deselected, 2 warnings in 18.56s
```

The action-order assertion now checks the ordered button descendants of
`#library-prompt-editor-actions`, preserving Copy Markdown → Duplicate →
Delete without requiring an obsolete common parent.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile tldw_chatbook/Widgets/Prompts/prompt_block_editor.py tldw_chatbook/Widgets/Library/library_prompts_canvas.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_prompt_block_editor.py Tests/UI/test_css_build_integrity.py Tests/UI/test_css_bundle_sync_guard.py -q --tb=short --show-capture=no --disable-warnings
```

Result: compilation and CSS synchronization succeeded; the focused
PromptBlockEditor/CSS suite reported `34 passed in 8.58s`.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check tldw_chatbook/Widgets/Library/library_prompts_canvas.py tldw_chatbook/Widgets/Prompts/prompt_block_editor.py Tests/UI/test_library_prompts_canvas.py Tests/UI/test_prompt_block_editor.py
git diff --check
```

Result: ruff and diff checks completed without findings.

The unchanged Task 5 copy selection remains intentionally RED: `4 failed,
83 deselected, 2 warnings in 7.75s`.

## Task 7 affected-suite correction

Task 7's affected suite exposed one final stale Task 4 characterization. The
field-order test still inspected direct `LibraryPromptsListCanvas.children`,
but the intentional editor shell now owns the content fields.

### RED

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py::test_prompts_canvas_editor_field_order_author_last_beside_keywords -q --tb=short --show-capture=no --disable-warnings
```

Result:

```text
1 failed, 2 warnings in 1.78s
ValueError: 'library-prompt-name' is not in list
```

### GREEN

The test now pins `#library-prompt-editor-content` as a child of
`#library-prompt-editor-shell`, then compares the ordered content descendants.
The semantic contract remains exact: Name < Description < System < User <
Keywords < Author.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py::test_prompts_canvas_editor_field_order_author_last_beside_keywords -q --tb=short --show-capture=no --disable-warnings
```

Result: `1 passed, 2 warnings in 2.01s`.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_library_prompts_canvas.py -k "geometry or action_group or copy_and_duplicate_relabeled or duplicate_button_between or field_order_author_last" -q --tb=short --show-capture=no --disable-warnings
```

Result: `13 passed, 88 deselected, 2 warnings in 18.82s`.

The full canvas command was also invoked, but this worker returned only 42
progress markers and no normalized pytest completion summary; no pass/fail
claim is made from that incomplete output. No production source changed in
this correction.
