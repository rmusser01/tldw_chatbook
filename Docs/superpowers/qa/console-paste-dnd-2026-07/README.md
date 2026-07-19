# Console paste / drag-drop attach — live QA evidence (2026-07-13)

TASK-216. Branch `worktree-console-paste-dnd-216`, HEAD **f4885aa1**
("feat(console): Alt+V grabs clipboard images into the attach pipeline").

Feature under test (all four behaviours):

1. Dropping a file onto a terminal *pastes its PATH as text*; the app
   intercepts that paste and auto-attaches — an **image** path stages a
   `📎 pending` indicator, a **text/code** path inlines as a collapsed `📄`
   segment. The raw path never lands as draft text.
2. Prose that merely contains a path (`what does /etc/hosts do?`) stays text.
3. A multi-path drop attaches the **first** path and toasts
   `Attached first of N dropped files.`
4. `Alt+V` grabs an OS-clipboard image into the pending slot (display name
   `clipboard-*.png`); empty clipboard → `No image on the clipboard.`;
   unreadable platform → `Clipboard images aren't readable on this platform —
   use Attach or drop a file.`. Footer advertises `alt+v Paste image`.

## Rig

Captured from **textual-serve** (real app CSS; `textual serve` of
`python -m tldw_chatbook.app`) in headless bundled Playwright chromium,
viewport **2050x1240** dsf=1, external `https://**` requests aborted
(fonts/CDNs only — localhost HTTP/WS untouched). App served **from this
worktree**: the serve launcher sets `PYTHONPATH=<worktree>:<QA-HOME>` and
`cwd=<QA-HOME>`, guaranteeing the branch code runs (a proven necessity from
the prior attachments walk).

Isolated **HOME** `/private/tmp/tldw-qa-paste-20260713` (HOME + XDG_DATA_HOME
+ XDG_CONFIG_HOME redirected there). Seeded `config.toml`: `default_tab=chat`,
splash off, `[console.onboarding] first_send_completed = true` (this makes the
first-run setup card go *quiet* / non-blocking so paste/keys reach the
composer), Llama_cpp provider + model pointed at the live llama.cpp server at
`http://127.0.0.1:9099` (Qwen3.6-27B, non-vision, `capabilities:["completion"]`).
No messages are sent in this walk — every capture is a *staging* state — so
model vision capability is irrelevant (the `📎`/`📄` indicators stage
regardless; capability only gates Send).

Test files created **in the QA HOME root** (so they pass `looks_attachable`,
whose allowed root is `~`): `red-square.png` (64x64, 184 B),
`blue-square.png` (64x64, 185 B), `zorblatt-notes.md` (196 B raw).

Each capture is a **fresh app process** (textual-serve spawns one per browser
connection), so no state leaks between shots. Serve on port 9111.

### Input injection (rig deviations — honest disclosure)

The documented `navigator.clipboard.writeText()` + `Ctrl+V` recipe from the
prior walk did **not** deliver a paste to the served xterm.js in this headless
chromium (draft stayed empty — see Observations). Two faithful substitutions
were used, both of which reproduce exactly the bytes a real terminal emits:

- **Paste**: capture the app WebSocket (via a Playwright init-script wrapper)
  and inject `["stdin", "\x1b[200~" + text + "\x1b[201~"]` — a real bracketed
  paste, which Textual parses into a `Paste` event (the same event a real
  terminal drop/paste produces). Verified: prose lands as draft text; an image
  path stages `📎`; an `.md` path inlines `📄`.
- **Alt+V**: inject the Kitty-keyboard CSI `["stdin", "\x1b[118;3u"]`
  (codepoint 118 `v`, modifier 3 = alt), which Textual's parser decodes
  atomically to key `alt+v`. This was required because the served xterm.js
  runs `macOptionIsMeta:false`, so a browser `Alt+v` keystroke — via
  `keyboard.press` **or** CDP `dispatchKeyEvent` — degrades to a literal `v`
  and never reaches the binding. The Kitty CSI does not depend on the fragile
  ESC-prefix timeout that a bare `\x1bv` relies on.

Driver: `cap3.py` (steps-JSON). Neither substitution changes what the app
sees vs. a real terminal — they are transport fidelity fixes, not behavioural
shortcuts.

## Alt+V clipboard stub (SANCTIONED, disclosed)

A headless serve process cannot reach an OS clipboard. Per the task's
sanctioned approach, an **env-gated `sitecustomize.py`** in the QA HOME
monkeypatches the *same module seam the mounted unit tests target* —
`tldw_chatbook.Chat.console_paste_attach._grabclipboard`. It installs only
when `TLDW_QA_CLIPBOARD_STUB` is set, and reads a mode file at *call time*:
`image` → returns a 96x96 PIL image (grab kind `image`); `raise` → raises
`OSError` (grab kind `unavailable`); anything else → passthrough to the real
`PIL.ImageGrab`. The **real** ImageGrab seam is exercised by
`Tests/UI/test_console_native_chat_flow.py` (`test_alt_v_*`); the stub only
gives the live screenshot walk something deterministic to grab. Everything
downstream of `_grabclipboard` (bytes → `process_attachment_bytes` →
`clipboard-*.png` display name → pending slot → toast) is the **real** code
path.

## Captures

- **drop-image-path-attached.png** — `/…/red-square.png` pasted (as a dropped
  path). Composer draft is **empty** (placeholder "Ask, command, or paste
  task…"); the pending indicator reads `📎 red-square.png · 184 B` with the
  `📎✓` attach button, `✕` clear, and Send controls. The path never became
  draft text. (Toast allowed to expire so the indicator shows unobstructed;
  see `multi-drop` for the toast.)
- **drop-md-inlined.png** — `/…/zorblatt-notes.md` pasted. The draft shows a
  **collapsed cyan** segment `📄 zorblatt-notes.md · 267 B` (267 B = processed
  content with the "--- Contents of … ---" framing; raw file is 196 B), and a
  `zorblatt-notes.md content inserted` toast. Text/code files inline; they are
  NOT staged as a `📎` attachment.
- **prose-with-path-stays-text.png** — `what does /etc/hosts do?` pasted →
  lands verbatim as draft text ("Composer: what does /etc/hosts do?"), Send
  lit, **no** `📎`, Attach button unchanged, "No sources attached". Prose
  containing a path is not mistaken for a drop.
- **multi-drop-first-of-n.png** — two newline-separated paths pasted
  (`red-square.png` then `blue-square.png`). The **first** attaches: two
  stacked toasts are visible — `Attached first of 2 dropped files.` and
  `red-square.png attached` — with the `📎 red-` pending indicator staged and
  the draft empty.
- **alt-v-attached.png** — `Alt+V` with the image stub. Composer shows the
  `📎` pending indicator and the toast `clipboard-20260713-191916.png attached`
  — the `clipboard-<ts>.png` display name synthesized by the real bytes-entry
  path. Draft empty.
- **alt-v-unavailable-toast.png** — `Alt+V` with the raising stub. The
  warning toast `Clipboard images aren't readable on this platform — use
  Attach or drop a file.` renders; nothing is staged, draft untouched. (Empty
  clipboard → `No image on the clipboard.` is covered by the mounted tests;
  not separately shot.)

The footer's `alt+v Paste image` hint (show=True binding) is visible in every
frame.

## Observations / notes

- **No product defects found.** All four behaviours reproduce exactly as
  specified against the real app code at f4885aa1.
- Rig-only: the served xterm.js `macOptionIsMeta:false` + the headless
  `clipboard+Ctrl+V` no-op are environment limitations of textual-serve in
  headless chromium, worked around with faithful stdin-byte injection
  (documented above) — they are not app behaviour.
- The staging toast pops at the composer's bottom-right, directly over the
  `📎`/`📄` pending-indicator region, so a shot taken immediately after
  staging shows the toast obscuring part of the indicator label. The published
  `drop-image-path-attached.png` waited for the toast to expire; the
  `alt-v-attached.png` and `multi-drop` frames deliberately keep the toast to
  evidence the notify copy.
- The left-rail "Staged Context / No sources attached" is the RAG *sources*
  panel and is independent of the composer-level pending attachment — it
  correctly stays "No sources attached" while a `📎` image is staged.

## Reproduce

Scripts live in the session scratchpad (not committed): `serve_paste.py`
(serve launcher, sets PYTHONPATH/HOME/XDG/ESCDELAY + clipboard-stub env),
`cap3.py` (Playwright steps-JSON driver with ws-stdin paste + Kitty-CSI
alt+v), and per-capture `steps_cap*.json`. QA HOME + its `sitecustomize.py`
live under `/private/tmp/tldw-qa-paste-20260713`.
