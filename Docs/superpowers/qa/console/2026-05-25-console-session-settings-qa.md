# Console Session Settings QA

Date: 2026-05-25

## Visual QA

Mounted `ChatScreen` with the generated modular stylesheet and exported Textual SVG captures for:

- `visual-captures/console-settings-default-160x44.svg`
- `visual-captures/console-settings-long-label-160x44.svg`
- `visual-captures/console-settings-compact-100x32.svg`
- `visual-captures/console-settings-modal-compact-100x32.svg`
- `visual-captures/console-settings-left-collapsed-160x44.svg`
- `visual-captures/console-settings-modal-default-160x44.svg`

Checks covered:

- Settings summary remains below staged context and above workspace context.
- Long model names stay on one row and ellipsize instead of wrapping.
- Compact-height left rail keeps the Settings button reachable.
- Compact modal stays inside the viewport and scrolls vertically so Save and validation text remain reachable.
- Collapsing the left rail hides the settings summary with the rest of the rail.

Note: the 100x32 capture keeps the Console workbench's existing horizontal minimums; this pass verified the new settings surface's vertical behavior within that current layout.

## Verification

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py`
- `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/console_settings_visual_qa.py`

The CSS build completed with the repository's existing missing-module warning for `features/_evaluation_v2.tcss`.
