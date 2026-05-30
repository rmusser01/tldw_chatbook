# Console Paste Threshold QA

## Scope

Focused QA for the Console composer large-paste collapse workflow. This pass verifies that pasted chunks over the configured threshold remain visible while the composer is active, can enter the two-step unfurl flow, and can expand back into literal pasted text.

## Visual Evidence

- `screenshots/console-paste-00-initial.png`: Console loaded in textual-web before paste input.
- `screenshots/console-paste-01-collapsed-token.png`: Active composer shows `Pasted Text: 135 Characters`.
- `screenshots/console-paste-02-unfurl-confirm.png`: First token activation shows `Unfurl?`.
- `screenshots/console-paste-03-expanded-text.png`: Second activation expands the literal pasted text in the active composer.

## Verification Commands

```bash
python -m pytest -q \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_real_click_enters_unfurl_prompt \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_composer_row_click_enters_unfurl_prompt \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_textual_web_row_click_enters_unfurl_prompt \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_textual_web_bottom_boundary_click_enters_unfurl_prompt \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_row_click_keeps_focus_on_composer \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_second_click_unfurls_literal_text \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_enter_on_focused_composer_matches_click_flow \
  Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_click_elsewhere_resets_unfurl_prompt \
  Tests/UI/test_console_internals_decomposition.py::test_console_native_composer_click_focuses_composer_not_visible_static \
  Tests/UI/test_console_internals_decomposition.py::test_console_native_composer_clicking_visible_draft_captures_typing \
  --tb=short
```

Result: `10 passed, 1 warning`.

## Notes

The root cause was the visible draft `Static` receiving focus on click. In textual-web that focus paint obscured the active composer text even though the draft renderable and payload state were correct. The fix keeps focus on `#console-native-composer`, treats `#console-command-visible-text` as a display surface, and preserves click/keyboard activation for collapsed paste tokens.
