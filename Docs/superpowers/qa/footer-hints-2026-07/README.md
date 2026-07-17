# Footer Shortcut Hints — Live Verification (task-264, AC #2)

- **Date:** 2026-07-17
- **Branch:** `claude/footer-hints-264` at a237c108 (impl fa443e29 + review fix wave e183ce68 + minors a237c108), off origin/dev 1bf8c0d3.
- **Recipe:** textual-serve (`serve_plain.py`, isolated QA HOME, splash disabled,
  `[console.onboarding] first_send_completed = true`, no provider key) + playwright bundled
  Chromium via the established `cap.py` driver — viewport 2050×1240 dsf=1, fontsize 12,
  `https://**` route-abort, `body.-first-byte` gate. Real served app, no fixtures/mocks.

## What this verifies

Before task-264, `AppFooterStatus` was mounted only on the App's default screen, which is
occluded the moment any `BaseAppScreen` is pushed — every `set_workbench_shortcuts()`
registration "succeeded" against an invisible widget. The branch mounts a per-screen
`AppFooterStatus` (`BaseAppScreen.compose()`), makes callers screen-aware, and (fix wave)
persists registrations on the screen so they survive screen-level recompose.

## Captures

| Capture | Footer text observed | Verdict |
|---|---|---|
| `console-footer-hints-2026-07-17.png` | `F6 next pane \| Shift+F6 previous pane \| F1 help \| Enter send \| Ctrl+K switch session \| Ctrl+T new tab \| Ctrl+P palette` + `Tokens: -- \| P/C/N/M DB sizes` right-docked | ✅ Console registered set renders, incl. the Ctrl+K/Ctrl+T hints migrated off the retired Textual Footer |
| `mcp-footer-hints-2026-07-17.png` | `1-4 mode \| a add server \| r refresh \| t test tool \| space cycle permission` | ✅ MCP hint set renders on its own screen |
| `library-footer-hints-2026-07-17.png` | `u use Library context in Console` | ✅ Library registration (added in fix wave — its `u` binding hint was silently lost when the Textual Footer was retired) |
| `settings-footer-hints-2026-07-17.png` | `s save category \| r revert category \| t test category` | ✅ Settings registration (fix wave, mirrors its show=True bindings) |
| `settings-recompose-survival-2026-07-17.png` | Same `s/r/t` text **after** switching category Overview → Providers & Models | ✅ Registration survives screen-level recompose (`active_category` is `recompose=True`, which replaces the footer widget — the persisted registration re-seeds it) |

Notes:
- The Console capture is the landing screen of a fresh served session (real
  `ChatScreen.on_mount()` registration path).
- The Settings recompose capture is the load-bearing one for the fix wave: without
  persistence, the footer resets to `Ctrl+Q quit | Ctrl+P palette` on the first category
  switch (reproduced during development — the naive on_mount-only registration failed
  exactly this way).
- DB-size/token segments render on the Console capture's right edge, confirming the
  per-screen instance also carries the status displays (`_active_footer_status` path).
