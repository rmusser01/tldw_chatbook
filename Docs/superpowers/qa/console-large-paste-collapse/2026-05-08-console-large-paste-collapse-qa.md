# Console Large Paste Collapse QA - 2026-05-08

## Scope

Actual rendered screenshot evidence for the Console composer large-paste collapse states.

## Capture Method

- Browser capture through `textual_serve.server.Server` and Playwright.
- Harness mounted the production `ChatScreen` and `ConsoleComposerBar` with the real modular TCSS bundle.
- Screenshots are PNG captures of the rendered Textual web surface, not SVGs, generated mockups, or code layout diagrams.

## Evidence

- Collapsed paste token: `Docs/superpowers/qa/console-large-paste-collapse/console-large-paste-collapsed.png`
- First-click confirmation state: `Docs/superpowers/qa/console-large-paste-collapse/console-large-paste-unfurl.png`
- Expanded literal paste state: `Docs/superpowers/qa/console-large-paste-collapse/console-large-paste-expanded.png`

## Verified Visual States

- Large pasted text renders as `Pasted Text: 660 Characters` in the composer without expanding the composer height.
- First-click state renders as `Unfurl?`.
- Expanded state renders literal pasted text and wraps within the composer.

## Notes

- These captures are implementation QA evidence only. Per project workflow, visual approval still requires user review of the actual screenshots.
