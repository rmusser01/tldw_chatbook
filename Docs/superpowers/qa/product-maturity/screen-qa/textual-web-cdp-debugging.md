# Textual-Web CDP Debugging Runbook

Date: 2026-05-09
Scope: tldw_chatbook screenshot QA and live UI debugging through `textual-web`

## Purpose

Use this runbook when a screen must be inspected, driven, or captured as an actual rendered PNG from the running app. The goal is to avoid rediscovering the same textual-web, browser automation, sandbox, and worktree pitfalls during each screen-QA pass.

This workflow is for debugging outside the sandbox when local port binding or Chromium automation is blocked by macOS permissions. It is not a replacement for mounted Textual tests; use tests for deterministic assertions and use this workflow for visual evidence and interaction QA.

## Binding Screenshot Rule

- A screen is visually approved only after the user approves an actual PNG screenshot from the running Textual app or textual-web surface.
- SVG exports, geometry dumps, code layouts, generated mockups, and ASCII diagrams are diagnostic artifacts only.
- Store approved and rejected screen evidence under `Docs/superpowers/qa/product-maturity/screen-qa/<screen>/`.
- Verify screenshot files with `file <path>.png` before citing them as evidence.

## When To Use This

- Capturing baseline and final screenshots for the 12-screen QA campaign.
- Verifying text visibility, focus, composer behavior, hover/focus states, and other visual details.
- Debugging why the browser shows a loader, wrong screen, stale code, or an unresponsive terminal.
- Driving high-level interactions where mounted Textual tests are insufficient as visual proof.

## Why Outside The Sandbox

Two host-level failures are expected under the sandbox:

- The textual-web server can fail to bind localhost with `PermissionError: [Errno 1] operation not permitted`.
- Chromium can fail to launch with a macOS Mach port permission error such as `mach_port_rendezvous_mac.cc ... Permission denied`.

If either happens, rerun the server or browser automation command outside the sandbox with an explicit escalation request. Do not work around this by accepting SVG-only evidence.

## Always Pin The Current Worktree

The repository-level virtualenv may contain an editable install that points at an older worktree. Always set `PYTHONPATH` to the worktree being tested. If logs mention a stale path such as another `.worktrees/codex-*` directory, the server is importing the wrong code.

```bash
WORKTREE=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/codex-screen-qa-artifacts
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
SERVE=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-serve
PORT=8824
QA_HOME=/private/tmp/tldw-chatbook-screenqa-home
QA_CONFIG=/private/tmp/tldw-chatbook-screenqa-config
QA_DATA=/private/tmp/tldw-chatbook-screenqa-data
```

Use a per-run HOME/XDG profile so first-run state, settings, and screen defaults are deterministic and do not mutate the developer's real profile.

```bash
mkdir -p "$QA_HOME" "$QA_CONFIG" "$QA_DATA"

PYTHONPATH="$WORKTREE" \
HOME="$QA_HOME" \
XDG_CONFIG_HOME="$QA_CONFIG" \
XDG_DATA_HOME="$QA_DATA" \
"$SERVE" --host 127.0.0.1 --port "$PORT"
```

## Deterministic Startup Screen

For screen-specific captures, prefer a temporary config over click-navigation through the xterm canvas. Let the app create a config once, then edit the temp profile only.

Example for Artifacts:

```toml
[general]
default_tab = "artifacts"

[splash_screen]
enabled = false
```

Expected config location for the variables above:

```bash
$QA_HOME/.config/tldw_cli/config.toml
```

Do not commit temp config files from `/private/tmp`.

## Recommended Playwright Capture

Playwright is the preferred wrapper for CDP-style browser control because it handles Chromium startup, page lifecycle, and screenshots with less protocol boilerplate.

Run outside the sandbox when Chromium cannot launch inside it.

```python
from pathlib import Path
from playwright.sync_api import sync_playwright

screen = "artifacts"
out = Path(
    f"Docs/superpowers/qa/product-maturity/screen-qa/{screen}/final-2026-05-09-{screen}.png"
).resolve()

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(
        viewport={"width": 2050, "height": 1240},
        device_scale_factor=1,
    )
    page.goto("http://127.0.0.1:8824", wait_until="domcontentloaded")
    page.wait_for_timeout(7000)

    # textual-web may keep an intro/shade overlay in the DOM after the terminal is visible.
    page.evaluate(
        """
        document.body.classList.add("-first-byte");
        for (const el of document.querySelectorAll(".intro-dialog,.closed-dialog,.shade")) {
          el.style.pointerEvents = "none";
        }
        """
    )

    page.screenshot(path=str(out), full_page=True)
    print(out)
    browser.close()
```

Then verify:

```bash
file Docs/superpowers/qa/product-maturity/screen-qa/artifacts/final-2026-05-09-artifacts.png
```

## Manual CDP Attach Option

Use this only when you need DevTools inspection or a persistent browser session. The Playwright capture path above is simpler for normal screen QA.

Start Chrome or Chromium with a debug port and an isolated profile:

```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=/private/tmp/tldw-chatbook-cdp-profile \
  http://127.0.0.1:8824
```

Attach with Playwright over CDP:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
    context = browser.contexts[0]
    page = context.pages[0]
    page.screenshot(path="cdp-capture.png", full_page=True)
    browser.close()
```

## Textual-Web Gotchas

- The app is rendered through xterm/canvas-like terminal DOM. Browser text selectors such as `get_by_text("Artifacts")` are not reliable for app content.
- DOM snapshots usually show textual-web chrome, not semantic Textual widgets.
- If clicks do nothing, check `document.elementFromPoint(x, y)`. A `.shade` or `.intro-dialog` overlay may be intercepting pointer events.
- Prefer deterministic startup config for the target screen. Use coordinate clicks only for interaction states that must be demonstrated visually.
- Keyboard input should target the xterm helper textarea when possible:

```python
page.locator(".xterm-helper-textarea").focus()
page.keyboard.press("Control+P")
```

- If the screenshot shows a loading screen, splash screen, wrong tab, blank terminal, or stale UI, inspect server stdout/stderr before capturing again.

## Debugging Checklist

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Server logs reference an old worktree | Missing `PYTHONPATH` pin | Relaunch with `PYTHONPATH="$WORKTREE"` |
| `PermissionError` while binding `127.0.0.1` | Sandbox blocked local server bind | Rerun server outside sandbox with approval |
| Chromium Mach port permission error | Sandbox blocked Chromium launch | Rerun Playwright outside sandbox with approval |
| Browser text selectors time out | App content is terminal-rendered | Use startup config, keyboard, coordinates, or screenshots |
| Coordinate clicks do nothing | textual-web overlay intercepts input | Disable `.intro-dialog`, `.closed-dialog`, and `.shade` pointer events |
| Screenshot captures wrong screen | Default tab/profile state is wrong | Set temp `[general] default_tab` and disable splash |
| Screenshot looks visually stale | Editable install imported old code | Check logs and relaunch with the correct `PYTHONPATH` |
| Screenshot is not acceptable evidence | It is SVG/mockup/geometry output | Capture an actual PNG from running app/textual-web |

## Cleanup

- Stop the textual-web server with `Ctrl-C` when the capture pass is done.
- Keep `/private/tmp` HOME/XDG profiles out of commits.
- Keep `.playwright-cli/` and other transient browser output out of commits unless explicitly required.
- Commit only the source changes, relevant tests, QA notes, and approved/rejected PNG evidence required for the screen PR.
