# Live verification — Console top-area layout

Date: 2026-07-22. Branch: `feat/console-top-area-layout`. Method: Textual `save_screenshot`
(SVG) via a `run_test` driver on the branch code against the real DB, rendered to PNG.

## Confirmed live
1. **One-line header.** `Console — Chat, source handoffs, live runs, and control actions.`
   renders on a SINGLE row with the `Ready` status badge pinned flush to the right edge —
   at both 230 and 100 columns (`top-wide-230col.svg`, `top-narrow-100col.svg`).
2. **Action row stays at the top.** `New tab / Settings / Attach context / Run Library RAG /
   Save Chatbook / Help` sits directly under the header, unchanged.
3. **Status pills moved above the composer.** The strip
   `Provider | Model | Assistant | RAG | Sources | Tools | Approvals` now renders as a
   full-width row directly above the composer, between the chat area and the composer.
4. **Scope chip** is present in the strip but hidden by default (unscoped → hidden, its
   pre-existing behavior); it renders when a retrieval scope is set.

## Covered by automated test (not a screenshot)
Narrow-width subtitle ellipsize + `Ready` flush-right at ~60 cols is pinned by
`test_console_header_inline_subtitle_ellipsizes_when_narrow` (bundled-CSS App, asserts the
subtitle width shrinks wide→narrow and the badge's right edge equals the terminal width).
100 cols is wide enough to show the full subtitle, so the ellipsis itself is asserted at 60.

## Captures
- `top-wide-230col.svg` — full rework at a wide terminal.
- `top-narrow-100col.svg` — one-line header + pills-above-composer hold at a narrow terminal.
