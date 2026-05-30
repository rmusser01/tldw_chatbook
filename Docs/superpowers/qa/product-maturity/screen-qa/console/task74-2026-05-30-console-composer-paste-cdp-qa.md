# TASK-74 Console Composer Paste CDP QA

Date: 2026-05-30
Branch: `codex/console-cdp-qa-run`
Base: `origin/dev`
App code base: `b1bcff71eeef67302b3dd3245c1a8685a09bddc5`
Backlog task: `TASK-74`
Screen: Console
Browser/CDP URL: `http://127.0.0.1:8895`
Viewport: `1280x720`
Capture method: actual textual-web rendering captured with Playwright/Chromium screenshots.

## Preflight

- Clean worktree verified before starting QA: `git status --short` returned no output before `TASK-74` was moved to `In Progress`.
- Branch was rebased onto `origin/dev` at `b1bcff71eeef67302b3dd3245c1a8685a09bddc5` before the final screenshot refresh because shared terminal/tab CSS landed while this QA ticket was in progress.
- Runtime path: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.
- Serve wrapper: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve`.
- Isolated profile root used for final refresh: `/private/tmp/tldw-chatbook-task74-refresh.SrMjDC`.
- Temporary config set `general.default_tab = "chat"`, `splash_screen.enabled = false`, `console.collapse_large_pastes = true`, and `console.paste_collapse_threshold = 50`.
- The first launch without a provider key correctly showed the provider setup blocker. The paste workflow was then run with an isolated dummy `OPENAI_API_KEY=DUMMY_TASK74_KEY` so the composer was editable without touching real user credentials.

## Commands

Initial server attempt, sandboxed, expected failure:

```bash
env PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-cdp-qa-run \
HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC \
XDG_CONFIG_HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC/.config \
XDG_DATA_HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC/.local/share \
XDG_CACHE_HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC/.cache \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve \
--host 127.0.0.1 --port 8895 --title 'Task74 Console QA Refresh'
```

Result: failed with `PermissionError: [Errno 1] operation not permitted` while binding localhost, matching the textual-web CDP runbook.

Server used for final QA, outside sandbox:

```bash
env PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-cdp-qa-run \
HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC \
XDG_CONFIG_HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC/.config \
XDG_DATA_HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC/.local/share \
XDG_CACHE_HOME=/private/tmp/tldw-chatbook-task74-refresh.SrMjDC/.cache \
OPENAI_API_KEY=DUMMY_TASK74_KEY \
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve \
--host 127.0.0.1 --port 8895 --title 'Task74 Console QA Refresh'
```

Standalone Playwright inside the sandbox failed with the expected macOS Chromium Mach-port permission error:

```text
mach_port_rendezvous_mac.cc:156 ... Permission denied (1100)
```

The same Playwright script was rerun outside the sandbox and produced the final PNG evidence listed below. The final refresh used a longer terminal startup wait to avoid capturing the textual-web splash screen before the Console was ready.

Focused automated verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
Tests/UI/test_console_internals_decomposition.py::test_console_large_paste_collapses_visible_token_but_preserves_payload \
Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_real_click_enters_unfurl_prompt \
Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_second_click_unfurls_literal_text \
Tests/UI/test_console_internals_decomposition.py::test_console_collapsed_paste_click_elsewhere_resets_unfurl_prompt \
Tests/UI/test_console_internals_decomposition.py::test_console_normal_typing_remains_literal_over_paste_threshold \
Tests/UI/test_console_native_chat_flow.py::test_console_collapsed_paste_sends_full_payload_not_visible_token \
--tb=short
```

Result after final rebase and screenshot refresh: `6 passed, 1 warning in 15.53s`. The warning is the existing `requests` dependency warning for `urllib3/chardet/charset_normalizer`.

## Screenshot Evidence

- Baseline Console, provider ready: `captures/task74-2026-05-30-01-baseline.png`
- Visible typed input: `captures/task74-2026-05-30-02-visible-typed-input.png`
- Large paste collapsed: `captures/task74-2026-05-30-03-large-paste-collapsed.png`
- First click shows `Unfurl?`: `captures/task74-2026-05-30-04-first-click-unfurl-prompt.png`
- Second click expands literal pasted text: `captures/task74-2026-05-30-05-second-click-expanded-paste.png`
- Normal keyboard typing over threshold remains literal: `captures/task74-2026-05-30-06-normal-typing-over-threshold-literal.png`

`file` verified all six final captures as `PNG image data, 1280 x 720, 8-bit/color RGB, non-interlaced`.

## Walkthrough Result

- CDP/browser QA ran against the rebased current-dev worktree with app code from `b1bcff71eeef67302b3dd3245c1a8685a09bddc5`, not a mockup, SVG, or static layout render.
- Short typing is visible in the composer and keeps the Send action available.
- A 143-character browser clipboard paste collapses to `Pasted Text: 143 Characters`.
- First click on the collapsed token changes the visible token to `Unfurl?`.
- Second click on `Unfurl?` expands the original pasted text back into the composer as normal text.
- Normal per-key keyboard typing beyond 50 characters stays literal and does not transform into a paste token.

## Failed Or Uncertain Observations

- Clicking the hidden textual-web textarea directly is not a reliable interaction path; the first attempt moved focus to the top navigation and opened Home. Final QA used coordinate focus on the rendered composer, which matches the runbook guidance for xterm/canvas surfaces.
- Browser-plugin screenshots were visually valid but produced JPEG bytes despite `.png` filenames. The final evidence was recaptured with standalone Playwright using explicit `type: "png"`.
- Bulk browser insertion APIs can behave like paste/insert chunks and therefore collapse over the configured threshold. The normal typing verification used per-key keyboard events to represent actual user typing.

## Residual Risks

- The visual QA used a dummy provider key to unblock the composer and did not send a message to OpenAI.
- The QA validates the default 50-character threshold path and enabled preference. Disabled preference behavior remains covered by mounted tests rather than this CDP pass.
- No Console composer regression was found during this ticket; no code fix or new regression was required.
