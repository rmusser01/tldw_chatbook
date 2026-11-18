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
- No "Verified against" lines or stamps in a page. Record what you verified,
  and against which branch and date, in the task's Implementation Notes
  (parallel PRs appending stamps caused most User Guide sync conflicts).

## Capture recipe

**Winner: Textual's built-in `App.save_screenshot()` SVG export**, driven
headlessly through `run_test()`. This approach passed the fidelity bar
immediately under the Step 1 timebox, so the two fallback candidates (tmux ANSI → rich SVG, and a textual-serve +
Playwright PNG harness) were never exercised.

Fidelity bar: nav bar labels present and complete, box-drawing glyphs intact,
theme colors rendered (not monochrome), no truncated right edge.

Use the checked-in driver. `scripts/capture_user_guide.py <target>` boots the
real app on a throwaway profile and writes a self-contained SVG (targets
`search-rag`, `file-notes`, `watchlists-workflow`, `plugins`);
`scripts/regen_approval_card_svg.py` renders the approval card. Before it
imports the app, the driver isolates the whole profile, not just the config
file: HOME, USERPROFILE, XDG_CONFIG_HOME, XDG_DATA_HOME, XDG_CACHE_HOME and
TLDW_CONFIG_PATH point into a fresh temporary directory, the keyring is the
null backend, proxies and credential-like environment variables are dropped,
writes to the real profile are refused, and sockets reach loopback only. It
deletes that profile when it finishes.

1. Add a target for your page to `scripts/capture_user_guide.py`: an
   `async def _<name>(app, pilot) -> Path` that drives the app to the state
   being documented (pilot clicks and key presses, with `_wait(...)` for each
   state) and returns `IMAGES / "<screen>" / "<name>.svg"`. Register it in
   `TARGETS`, with any setup that must run before the app boots in the second
   slot (as `watchlists-workflow` does), and list it in the module docstring.

2. Run it from the repository root, one target per process:
   ```bash
   cd "$(git rev-parse --show-toplevel)"   # your checkout of the branch being documented
   mkdir -p "Docs/User_Guide/images/<screen>"
   PYTHONPATH=. .venv/bin/python scripts/capture_user_guide.py <target>
   ```

3. Sanity-check before embedding: open the `.svg` in a browser, or thumbnail
   it locally (macOS: `qlmanage -t -s 2458 -o /tmp/out file.svg`) and eyeball
   it against the fidelity bar above.

4. Check that no remote font source is left. Textual/rich exports embed
   `@font-face` rules that fetch Fira Code from cdnjs.cloudflare.com; the
   driver strips them (keeping the `local(...)` source and the `monospace`
   fallback), because docs images must be self-contained:
   ```bash
   grep -rl "cdnjs" Docs/User_Guide/images/ && echo "STILL DIRTY" || echo clean
   ```

Never drive the app for a capture with only `TLDW_CONFIG_PATH` set: HOME, the
XDG directories and the OS keyring would still be your real profile's. If a
one-off script is unavoidable, start it the way the driver's `main()` does:
call the driver's `_isolate()` on a fresh temporary directory before anything
imports `tldw_chatbook`, then install the same DB-layer ports before building
`TldwCli()`.

**Standard size: 235×52 cells**, the size `scripts/capture_user_guide.py`
uses (its `SIZE`) and the size of nearly every capture already in the
guide. Always pass `size=(235, 52)` to `run_test()` so every page's captures
match.

**Output path convention:** `Docs/User_Guide/images/<screen>/<name>.svg`,
where `<screen>` is the same kebab-case slug used for that screen's guide
page (e.g. `console`, `library`) and `<name>` describes the specific view
(e.g. `overview`, `get-started-card`).

Demo-content rules: the driver's throwaway profile only, canned demo text, no
personal data; a loopback endpoint or a scripted model gateway for replies
(the driver refuses connections to any other address).
