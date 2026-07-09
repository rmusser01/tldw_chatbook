# Console Keyboard Layer Phase 3 — QA evidence (2026-07-05)

Branch: claude/console-keyboard-phase3 (base ee1425c3). Captured from
textual-serve (real app CSS) in headless bundled chromium, ready-seeded
llama_cpp HOME.

- keyboard-ctrl-k-switcher-filtered-2026-07-05.png — Ctrl+T created a second
  tab, Ctrl+K opened "Switch Session" with the query focused; typing
  "chat 1" filters to the single matching open session.
- keyboard-model-popover-via-palette-2026-07-05.png — the Model popover
  (provider select, model select, temperature, streaming toggle,
  Full settings… / Apply) opened via the palette command.
- keyboard-alt-m-popover-2026-07-05.png — KNOWN CAPTURE LIMITATION: Alt+M did
  not open the popover through the BROWSER terminal (xterm.js does not
  encode Alt as ESC-prefix by default in this client). Terminal-level
  encoding is sound (ESC-prefixed alt is standard; verified none of the
  chosen keys alias Enter/Tab), and the pilot suite covers the binding; the
  palette provides a universal fallback. Real-terminal verification of
  alt+m / alt+1..9 recommended at first manual use.
- keyboard-footer-hints-composer-2026-07-05.png — native footer with the
  composer focused: `^k Switch session · alt+m Model · ^t New tab` + app
  defaults (post-fix 72b21e05: hints routed through Textual's native Footer
  after review found the parallel AppFooterStatus mechanism unreachable).
- keyboard-footer-hints-transcript-2026-07-05.png — transcript focused: its
  message bindings (↑/↓ select, ⏎ actions, esc clear, c/e/r) plus the
  screen set — contextual per pane via the binding display, as spec §3 asks.
- keyboard-palette-console-commands-2026-07-05.png — Ctrl+P filtered to the
  five scoped `Console:` commands with key hints in help text.

Verification (2026-07-05, post-fix): seven affected suites = 480 passed,
1 failed — only the documented pre-existing dev failure
(test_console_left_rail_prioritizes_attach_and_active_conversation).
