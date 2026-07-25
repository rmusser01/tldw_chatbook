# Page template & authoring guide (not user-facing)

Copy everything between the BEGIN/END markers into a new page and fill it in.
Every page is written from a LIVE driving session: execute every claim and
every how-to on-screen before writing it down.

<!-- BEGIN TEMPLATE -->
# <Screen> — <one-line purpose>

## What this screen is for
(2–4 sentences; when to reach for it)

## Getting there
(nav key/number, command palette entry, startup config)

## Layout tour
(capture + region-by-region walk; regions named exactly as labeled on screen)

## Features & controls
(reference table per region: control → what it does)

## Common tasks
(3–8 numbered step-by-step how-tos, imperative voice)

## Keyboard & commands
(table: key / slash command → action; SCREEN-SPECIFIC only — globals live in
the [guide index](index.md))

## Related settings & docs
(Settings panes, config.toml keys, Docs/Features links)

## Quirks & troubleshooting
(honest limitations with backlog refs; common errors and their fixes)

—
*Verified against dev @ <short-sha> — <YYYY-MM-DD>*
<!-- END TEMPLATE -->

## Authoring rules

- On-screen labels verbatim. No internal jargon (no "native id", "store",
  "recompose").
- No aspirational features. Limitations stated honestly with a backlog ref
  where one exists.
- Screen-specific keys only in "Keyboard & commands"; global keys live in
  index.md.
- Stub pages: first two sections + a "🚧 This page is a stub" banner + links
  to any existing Docs/Features deep dive.
- Form-heavy panes (chiefly Settings): self-describing form fields may be
  summarized at field-group level; interactive/behavioral controls are always
  enumerated individually.
- Before the phase PR merges: re-check dev history for the documented
  screen's modules; re-verify and re-stamp affected sections if it moved.

## Capture recipe

**Winner: Textual's built-in `App.save_screenshot()` SVG export**, driven
headlessly through `run_test()`. This approach passed the fidelity bar
immediately under the Step 1 timebox, so the two fallback candidates (tmux ANSI → rich SVG, and a textual-serve +
Playwright PNG harness) were never exercised.

Fidelity bar: nav bar labels present and complete, box-drawing glyphs intact,
theme colors rendered (not monochrome), no truncated right edge.

1. Scratch profile (create once, reuse across every page's captures — never
   point `TLDW_CONFIG_PATH` at a real profile):
   ```bash
   S=<scratch-root>            # any writable scratch dir, e.g. $TMPDIR/guide_capture
   mkdir -p "$S/g0_profile"
   cat > "$S/g0_profile/config.toml" <<'EOF'
   [general]
   users_name = "guide_g0"
   default_tab = "chat"

   [splash_screen]
   enabled = false
   EOF
   ```

2. Driver script — copy per page and add pilot actions (clicks, key presses,
   `await pilot.pause(...)`) between entering `run_test()` and calling
   `save_screenshot()` to drive to the state being documented:
   ```bash
   cat > "$S/capture_<screen>.py" <<'EOF'
   """Drive the real app under run_test and export an SVG screenshot."""
   import asyncio, os

   if "TLDW_CONFIG_PATH" not in os.environ:
       raise SystemExit("Refusing to run against the real profile: set TLDW_CONFIG_PATH")

   async def main() -> None:
       from tldw_chatbook.app import TldwCli
       app = TldwCli()
       async with app.run_test(size=(200, 50)) as pilot:
           # --- drive to the state being documented, e.g.: ---
           # await pilot.click("#some-nav-label")
           # await pilot.pause(0.5)
           app.save_screenshot(os.environ["G0_CAPTURE_OUT"])

   asyncio.run(main())
   EOF
   ```

3. Run it against the scratch profile, writing straight into the page's image
   directory:
   ```bash
   cd "$(git rev-parse --show-toplevel)"   # your checkout of the branch being documented
   mkdir -p "Docs/User_Guide/images/<screen>"
   TLDW_CONFIG_PATH="$S/g0_profile/config.toml" \
     G0_CAPTURE_OUT="Docs/User_Guide/images/<screen>/<name>.svg" \
     .venv/bin/python "$S/capture_<screen>.py"
   ```

4. Sanity-check before embedding: open the `.svg` in a browser, or thumbnail
   it locally (macOS: `qlmanage -t -s 2458 -o /tmp/out file.svg`) and eyeball
   it against the fidelity bar above.

**Standard size: 200×50 cells.** The live nav survey (Task 1) drove the app
at 235×52; this recipe deliberately uses a smaller, fixed 200×50 so every
page's captures are the same size. Verified during the Step 1 experiment
that 200 columns does *not* clip the 13-item nav bar — every destination
through "More: Ctrl+P" renders inside the frame. Always pass
`size=(200, 50)` to `run_test()`.

**Output path convention:** `Docs/User_Guide/images/<screen>/<name>.svg`,
where `<screen>` is the same kebab-case slug used for that screen's guide
page (e.g. `console`, `library`) and `<name>` describes the specific view
(e.g. `overview`, `get-started-card`).

Demo-content rules: scratch profile only (TLDW_CONFIG_PATH), canned demo
text, no personal data; local llama endpoint for live replies; delete the
scratch data dir afterwards.
