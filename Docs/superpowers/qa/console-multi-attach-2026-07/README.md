# Console multi-attachment — live QA evidence (2026-07-13/14)

TASK-217. Worktree `console-multi-attach-217`, HEAD **9227c3dc**
("feat(console): transcript chip per attachment").

> **Post-fix update — 2026-07-14, HEAD `64584872`.** The P0 crash below is
> **FIXED** and the three previously-blocked send-dependent captures are now
> delivered (`sent-chips-per-attachment.png`, `save-all.png`,
> `resume-multi-rehydrated.png`) as the live proof. Defect 2 is
> **RESOLVED-AS-DOCUMENTED** (the spec was amended: `display_name` persists for
> positions ≥ 1; position 0 keeps the mime·size resume label). See the
> **Post-fix re-capture** section at the end for the full re-run. The original
> pre-fix findings below are preserved verbatim for the record.

Feature under test: Console messages carry up to **5** attachments — staging
appends (picker / path-paste / Alt+V), a `📎 N files` indicator, a 5-per-message
cap, multi-path-drop truncation, one `🖼` chip per attachment on send, Save-all,
and a v18→v19 `message_attachments` table so resumed conversations rehydrate
every chip.

## Headline result

**The staging layer works. The multi-attachment SEND path is broken by a P0
crash** — see **Defect 1**. Sending a message with **2 or more** attachments
raises an unhandled `TypeError` in the persist worker that **terminates the
Console app** (textual-serve shows "Session ended. Restart"). This blocks the
three send-dependent captures (`sent-chips-per-attachment`, `save-all`,
`resume-multi-rehydrated`). Single-attachment send is unaffected and works
(215 behavior preserved — see `CONTROL-single-send-ok.png`).

## Rig

- **Serve**: raw `textual_serve.Server` running `python -m tldw_chatbook.app`
  **from this worktree** (real app TCSS). `statics_path` points at a patched
  `textual.js` that stashes the xterm driver on `window.__drv`, so the
  Playwright driver reads the real xterm buffer (`translateToString`) and maps
  cells→pixels for accurate button clicks. Serve env sets
  `PYTHONPATH=<worktree>:<QA-HOME>` and `cwd=<QA-HOME>` (branch code wins; the
  HOME `sitecustomize.py` clipboard stub is importable), `TERM=xterm` with
  iTerm/VTE markers unset (terminal `auto` → pixels), and `ESCDELAY=1500`.
  Port 9141, bundled Playwright chromium, headless, viewport **2050×1240**
  dsf=1, external `https://**` aborted (localhost HTTP/WS untouched).
- **Isolated HOME** `/private/tmp/tldw-qa-multi-20260713`
  (`HOME`+`XDG_CONFIG_HOME`+`XDG_DATA_HOME`). Seeded `config.toml`:
  splash off; `[console.onboarding] first_send_completed=true`;
  `[chat_defaults]`/`[api_settings.llama_cpp]` → live llama.cpp @
  `127.0.0.1:9099` (`Qwen3.6-27B-…gguf`, **non-vision**); vision override
  `[model_capabilities.models."Qwen3.6-…gguf"]={vision=true,max_images=10}` so
  image sends pass the pre-flight vision gate (verified
  `is_vision_capable('llama_cpp', model) → True`); `[chat.images]
  save_location=<HOME>/Downloads`, `default_render_mode=pixels`.
- **Six distinct PNGs** in the HOME root (so they pass `looks_attachable`,
  whose allowed root is `~`): red/green/blue/yellow/purple (184–186 B, 64×64)
  + orange (the 6th, for the cap/truncation tests).
- Each browser connection = one fresh app subprocess (textual-serve spawns one
  per WS), so state does not leak between captures.

### Input injection (rig deviations — honest disclosure)

Same faithful transport substitutions proven in the TASK-216 paste/dnd walk
(`console-paste-dnd-2026-07/README.md`), because the served xterm.js runs
`macOptionIsMeta:false` and headless chromium won't deliver a real paste:

- **Paste / terminal drop**: a WebSocket init-script wrapper captures the app
  tty socket; we inject `["stdin","\x1b[200~"+path+"\x1b[201~"]` — a real
  bracketed paste → Textual `Paste` event → `on_paste` auto-attaches.
- **Alt+V**: inject the Kitty CSI `["stdin","\x1b[118;3u"]` → key `alt+v`.
- **Send**: geometry click on the composer **Send** button (verified reliable;
  the browser Enter keystroke was not reliably delivered by the served xterm).
- **Clipboard-paths route** (Alt+V grabbing Finder-copied files): an env-gated
  `sitecustomize.py` stub monkeypatches the SAME seam the mounted unit tests
  target (`console_paste_attach._grabclipboard`), returning a **list of file
  paths** for the `"paths:"` mode. Everything downstream — `grab_clipboard_image`
  → `kind="paths"` → the attach-until-cap loop + truncation toast — is the real
  product code path. (Modes: `image` / `raise` / `paths:<a>|<b>` / passthrough.)

None of these change what the Textual app sees vs. a real terminal.

## Captures

### Delivered — staging layer (works as specified)

- **`staged-three-files.png`** — three image paths pasted sequentially
  (red, green, blue). Composer indicator reads exactly **`📎 3 files`** (one
  paperclip; the second paperclip glyph at right is the separate Attach
  button). Draft empty (placeholder), transcript "No messages yet", left-rail
  "No sources attached" (RAG panel, independent of composer staging). Staging
  **appends** as specified.
- **`cap-toast-sixth.png`** — five images staged, then a 6th attach attempted.
  Warning toast **`Attachment limit reached (5 per message).`** is shown; the
  6th is rejected. The toast overlays the bottom-right indicator region (a
  documented rig characteristic — same in the 216 walk), so `📎 5 files` is not
  visible *in this frame*; **count-stays-5 verified out-of-frame** by buffer
  reads: `📎 5 files` present immediately before the 6th paste **and** again
  after the toast dismisses (`MAX_PENDING_ATTACHMENTS=5` rejects the 6th, count
  unchanged).
- **`multi-drop-truncation.png`** — fresh session; Alt+V clipboard-**paths**
  route with 6 attachable paths (fresh capacity 5). The attach-until-cap loop
  stages 5 and toasts **`Attached first 5 of 6 dropped files.`** The frame shows
  the full toast stack: five per-file `…attached` toasts (red/green/blue/
  yellow/purple) + the truncation toast; orange (the 6th) is dropped at the cap.
  After the toasts clear, `📎 5 files` is confirmed staged.
  *Note on route choice*: this is the clipboard-paths route (feature bullet
  "multi-path drop attaches files until the cap"), which produces the variable
  `n`. The terminal-drop `on_paste` route only ever surfaces the first decoded
  path (`extract_dropped_path`), so it always toasts "Attached first **1** of
  m" regardless of capacity — that is by design (see `chat_screen.py` comment
  at the `on_paste` truncation branch), not the capacity-driven truncation this
  capture demonstrates.

### Delivered — migration (BONUS, works)

- **`migration-upgrade-clean.png`** — a real **v18 fixture** was built by
  downgrading a copy of the DB (drop `message_attachments`, set version 18)
  that held a genuine single-image conversation ("One square", 184 B PNG in the
  legacy `image_data` column). Booting the branch ran the **v18→v19 migration**
  (verified: `db_schema_version` → 19, `message_attachments` re-created). The
  old conversation resumed **intact** — user message + red square rendered
  inline from DB bytes. Its chip reads **`🖼 image/png · 184 B`** (the legacy
  position-0 mime·size form — see Defect 2), which is the documented 215 resume
  behavior; the migration itself is clean and non-destructive.

### Supporting — the blocking defect

- **`DEFECT-multi-send-crash.png`** — three images staged (`📎 3 files`) + text,
  then **Send**. The message never posts; the whole Console app **terminates**
  ("Session ended. Restart"). The xterm buffer's last frame before the crash
  showed the transcript still "No messages yet" with the draft + `📎 3 files`
  **stuck** (no user-visible error toast). The `serve.log` for this connection
  ends with the `TypeError` traceback + the app's `FINALLY block after
  app.run()` (process exit). See Defect 1.
- **`CONTROL-single-send-ok.png`** — the same flow with **one** image ("One
  square" + red-square.png). Sends fine: user message renders the chip
  **`🖼 red-square.png`** (real filename, from the in-memory staged label) with
  the red square inline; Assistant **`[failed]`** (the honest HTTP-500 from the
  non-vision llama.cpp). Proves the crash is **specific to >1 attachment**.

### Delivered post-fix (were blocked by Defect 1 pre-fix)

- **`sent-chips-per-attachment.png`**, **`save-all.png`**,
  **`resume-multi-rehydrated.png`** — pre-fix these were blocked because the
  multi-attachment send crashed before the message could be
  appended/rendered/persisted (`<HOME>/Downloads` empty). **At HEAD `64584872`
  all three are captured** — see the **Post-fix re-capture** section at the end
  of this file for the full story and `ls` evidence.

## Defects

### Defect 1 (P0, blocking) — multi-attachment send crashes the Console app — **FIXED in `64584872`**

> **Status: FIXED.** At HEAD `64584872` a ≥2-attachment send persists cleanly
> (legacy columns hold #0, `message_attachments` holds ≥1) and the app does not
> crash — see `sent-chips-per-attachment.png` (send), `save-all.png`, and
> `resume-multi-rehydrated.png`. Original pre-fix analysis preserved below.

Sending any message with **≥2** attachments raises, in the persist worker:

```
TypeError: ChatPersistenceService.create_message() missing 2 required
keyword-only arguments: 'image_data' and 'image_mime_type'
```

Traceback: `console_chat_store.append_message` (persist=True) →
`_persist_new_message_or_defer` (:759) → `_persist_new_message` (:821) →
`self.persistence.create_message(**create_kwargs)`.

Root cause (`tldw_chatbook/Chat/console_chat_store.py:790-821`): `create_kwargs`
is built **without** `image_data`/`image_mime_type`. For `len(attachments) > 1`
the code sets `create_kwargs["attachments"] = attachments_payload` and takes the
`if` branch, so the `else` branch that would set `image_data`/`image_mime_type`
never runs. But `ChatPersistenceService.create_message`
(`chat_persistence_service.py:295`) declares `image_data` and `image_mime_type`
as **required keyword-only** parameters (no defaults) — so the call raises.

The unhandled exception propagates out of the send worker and **terminates the
app** (observed live: "Session ended"). Impact: the entire multi-attachment
value proposition (send → chips → save-all → resume) is unreachable; the user
loses their composed message with no error surfaced.

Minimal deterministic repro (rig-independent, through the real persistence
layer): control single-attachment `create_message(..., image_data=, image_mime_type=)`
succeeds; the multi-attachment call the store makes
(`create_message(..., attachments=[...])` with no `image_data`/`image_mime_type`)
raises the exact `TypeError`.

Single-attachment send is safe: `len(attachments) > 1` is False →
`attachments_payload` stays None → `else` branch sets the scalar image kwargs.

### Defect 2 (functional, secondary) — position-0 attachment loses its filename on resume — **RESOLVED-AS-DOCUMENTED in `64584872`**

> **Status: RESOLVED-AS-DOCUMENTED.** This is inherent to the zero-duplication
> schema (position 0 lives in the legacy `messages.image_data` columns, which
> carry no filename). The design spec was amended to state the true behavior:
> **`display_name` persists for positions ≥ 1; position 0 keeps the pre-existing
> mime·size resume label.** Live-confirmed by `resume-multi-rehydrated.png`
> (chip #0 = `🖼 image/png · 184 B`; chips #1/#2 = `green-square.png` /
> `blue-square.png`). Original analysis preserved below.

Position 0 is persisted only in the legacy `messages.image_data` /
`image_mime_type` columns, which have **no filename field**;
`ChatPersistenceService.create_message` drops the position-0 `display_name`, and
`chat_screen._console_messages_from_conversation_tree` rebuilds position 0 with
`display_name=""`. So on resume the **first** attachment always renders as the
mime·size fallback (`🖼 image/png · 184 B`), never the original filename. Only
positions ≥1 (the new `message_attachments` rows) carry `display_name` and
resume with real names.

This contradicts the task's stated behavior "resumed conversations rehydrate
ALL chips WITH REAL FILENAMES now (the old mime·size fallback is only for
pre-migration rows)": the fallback is inherent to position 0 for **any** row,
pre- or post-migration. Live-corroborated by `migration-upgrade-clean.png`
(a post-migration single-image message resuming as `🖼 image/png · 184 B`).
This is secondary/moot until Defect 1 is fixed (no multi-attachment message can
be persisted to resume at all), and it also affects the working single-image
resume path.

## Observations / notes

- Staging is the strong part of the feature: append semantics, the `📎 N files`
  indicator (exactly one glyph), the 5-cap with its toast, and the
  attach-until-cap paths-route truncation all behave exactly as specified.
- The vision override is required only so image sends pass the pre-flight gate;
  the model still 500s (non-vision), which is the expected honest failure and
  is irrelevant to persistence — a *single*-image user message persists and
  resumes fine despite it.
- Toast-over-indicator overlap (cap toast, truncation toast) is a textual-serve
  rig characteristic, worked around by bracketing buffer reads, not a defect.
- DB pollution note: the crashed multi-sends leave empty "Three squares"
  conversation shells in the rail (conversation created, message persist
  crashed) — visible in some frames; itself a symptom of Defect 1.

## Reproduce

Scripts (session scratchpad, not committed):
`serve_multi.py` (serve launcher), `drv.py` (geometry + ws-stdin paste/altv +
Send-click helpers), `setup_home.py` (HOME/config/stub/PNGs),
`cap12.py`/`caps_rest.py`/`cap7.py` (captures), `repro_crash.py` (the Defect 1
unit repro). Post-fix add-ons: `cap_send_save.py` (send + Save-all),
`cap_resume.py` (fresh-process resume). QA HOME + `sitecustomize.py` under
`/private/tmp/tldw-qa-multi-20260713`.

---

## Post-fix re-capture (2026-07-14, HEAD `64584872`)

Same rig (textual-serve from the worktree, headless Playwright chromium
2050×1240, isolated HOME `/private/tmp/tldw-qa-multi-20260713`, live llama.cpp @
127.0.0.1:9099 with the vision override, ws-stdin bracketed-paste transport).
The `ChaChaNotes` DB in the isolated HOME was wiped before the run so the rail
holds exactly one real conversation (unambiguous resume). The three
send-dependent captures that Defect 1 had blocked are now **delivered**:

- **`sent-chips-per-attachment.png`** — three images (red/green/blue) staged by
  sequential path-paste + text "Three squares", then **Send for real**. The
  send **does not crash** (P0 proof): the user message posts with **three `🖼`
  chips carrying the real filenames in-session** (`red-square.png`,
  `green-square.png`, `blue-square.png`) and the first image (red) rendered
  inline below. Assistant `[failed]` is the honest non-vision llama.cpp 500 and
  is irrelevant to persistence. DB after send: one `Three squares` conversation;
  legacy `messages.image_data` holds #0 (red, 184 B); `message_attachments`
  holds position 1 = `green-square.png` and position 2 = `blue-square.png`
  (185 B each) — exactly the fixed persistence shape.
- **`save-all.png`** — the sent user message is selected; its action row shows
  Copy / Edit / Save as… / 👍 👎 / **View** / **Save Image** (tooltip "Save
  image to disk."). Clicking **Save Image** writes all three attachments and
  toasts **`Saved 3 images to /private/tmp/tldw-qa-multi-20260713/Downloads`**.
  `ls` evidence (3 distinct files, distinct md5s):

  ```
  184  console_image_20260714_011125.png    md5 b1174b58…  (red,   position 0)
  185  console_image_20260714_011125_1.png  md5 1dd12f9b…  (green, position 1)
  185  console_image_20260714_011125_2.png  md5 f23024d5…  (blue,  position 2)
  ```

- **`resume-multi-rehydrated.png`** — serve was **killed and restarted** (a
  truly fresh app process, no in-memory leak); the `Three squares` conversation
  is resumed from the rail. All **three chips rehydrate from the DB** and the
  first image re-renders inline. Per the amended spec (Defect 2 resolution):
  chip **#0 = `🖼 image/png · 184 B`** (position 0, legacy columns carry no
  filename → mime·size label), chip **#1 = `🖼 green-square.png`** and chip
  **#2 = `🖼 blue-square.png`** (real `display_name`s from `message_attachments`).

### Post-fix observations

- The `Console` left-rail "Chats" section renders several **static sample
  entries** ("One square", extra "Three squares — saved chat" rows with no
  timestamp) that **do not exist in the DB** (`conversations` holds one row; a
  filesystem grep across the isolated HOME finds these strings nowhere). They
  are pre-existing decorative rail rows, not produced by this feature and not
  from persistence — the real conversation is the timestamped
  "Chats — active session / saved chat — Nm" entry. Cosmetic; noted for honesty,
  fixed nothing.
- Send-session chips show real filenames for **all three** positions;
  resume-session chip #0 degrades to the mime·size label — the documented
  position-0 behavior, identical to the working single-image resume path.
