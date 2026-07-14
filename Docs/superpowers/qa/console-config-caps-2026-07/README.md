# Console config-driven attachment filters + tiff/svg — live QA evidence (2026-07-14)

TASK-222. Worktree `console-config-caps-222`, HEAD **b578c3b1**
("test(chat): realign bytes-cap fault injection to the policy function").

Feature under test: the `[chat.images]` config section now genuinely governs the
Console attachment allowlist, picker filters, and image caps — and **.tiff**
(PIL → transcode-to-PNG) and **.svg** (cairosvg rasterization) are new,
first-class image attachments that are both **advertised** (shown by the picker
filter) **and accepted** (staged, sent, rendered inline).

## Headline result

**CLEAN — all four required captures delivered, no defects in the feature.**
The seeded `[chat.images].supported_formats` drives the picker "Image Files"
filter (tiff + svg listed, not hidden), a TIFF stages+sends and renders inline
as its transcoded PNG, and an SVG stages+sends and renders inline as its
cairosvg-rasterized PNG. The only non-success in the frames is the Assistant
`[failed]` / HTTP-500 from the local **non-vision** llama.cpp model — the
expected, honest failure (identical to the TASK-215/217 walks); it is
downstream of and irrelevant to attachment staging/rendering.

## Rig

Reuses the proven TASK-215/216/217 geometry+transport rig verbatim.

- **Serve**: raw `textual_serve.Server` running `python -m tldw_chatbook.app`
  **from this worktree** (real app TCSS). `statics_path` points at the patched
  `textual.js` (`/private/tmp/tldw-qa-inline-20260713/static_patched`) that
  stashes the xterm driver on `window.__drv`, so the Playwright driver reads the
  real xterm buffer (`translateToString`) and maps cells→pixels for accurate
  clicks. Serve env: `PYTHONPATH=<worktree>:<QA-HOME>`,
  `cwd=<QA-HOME>` (branch code wins; `create_subprocess_shell` inherits the
  serve cwd, so the picker's `location="."` opens at the QA HOME), `TERM=xterm`
  with iTerm/VTE markers unset (terminal `auto` → **pixels**, so images render
  inline), `ESCDELAY=1500`. Port 9141, bundled Playwright chromium, headless,
  viewport **2050×1240** dsf=1, external `https://**` aborted.
- **Isolated HOME** `/private/tmp/tldw-qa-config-caps-20260714`
  (`HOME`+`XDG_CONFIG_HOME`+`XDG_DATA_HOME`). Hard guardrail honored: nothing
  outside `/private/tmp` and the worktree evidence dir was written; the real
  `~/.config/tldw_cli` was never touched.
- **Seeded `config.toml`** (provenance — the feature is seeded explicitly so the
  capture is unambiguous): splash off; `[console.onboarding]
  first_send_completed=true`; `[chat_defaults]`/`[api_settings.llama_cpp]` →
  live llama.cpp @ `127.0.0.1:9099` (`Qwen3.6-27B-…gguf`, **non-vision**);
  vision override `[model_capabilities.models."Qwen3.6-…gguf"] =
  {vision=true, max_images=10}` so image sends pass the pre-flight vision gate;
  and the section under test —

  ```toml
  [chat.images]
  supported_formats = [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg"]
  max_size_mb = 10.0
  resize_max_dimension = 2048
  save_location = "<HOME>/Downloads"
  default_render_mode = "pixels"
  ```

- **Test files** in `<HOME>/attachments/`: `pic.png` (PIL 64×64 red, 184 B),
  `scan.tiff` (PIL 200×120 RGB, two-tone: orange band / teal band / white centre
  bar, 72 140 B), `logo.svg` (200×100: blue rect + white circle + blue inner
  square, 309 B) — each drawing deliberately distinctive so the transcoded /
  rasterized inline render is visually recognizable.

### Pre-flight verification (before any capture)

- `ensure_svg_rendering()` → **True** (cairosvg found in the venv).
- `curl http://127.0.0.1:9099/v1/models` → **up**, returns the Qwen model.
- Headless pipeline check through the real product code (env → QA HOME):
  - `supported_image_formats()` = `(.png .jpg .jpeg .gif .webp .bmp .tiff .tif .svg)`
    — driven by the seeded config; `max_image_bytes()` = 10 MB;
    `resize_max_dimension` = 2048.
  - `attachment_filter_specs()` → both "All Supported Files" and "Image Files"
    patterns contain `*.tiff` and `*.svg`.
  - `process_attachment_path("scan.tiff")` → mime `image/png`, 72 140 B →
    **326 B PNG** (PNG magic) = TIFF transcoded.
  - `process_attachment_path("logo.svg")` → mime `image/png`, 309 B →
    **1 820 B PNG** (PNG magic) = SVG rasterized.

### Input injection (rig deviation — honest disclosure)

- **File picker confirm**: after clicking the file option in the picker's
  `DirectoryNavigation` (which fills the filename `Input` via
  `DirectoryNavigation.Selected`), the selection is confirmed with a terminal
  Enter injected over the app tty WebSocket (`["stdin","\r"]` → Textual
  `Input.Submitted` → `file_dialog._confirm_file`). This is the real product
  confirm path; only the keystroke transport is substituted (the served
  `xterm.js` won't deliver a native Enter reliably). Directory navigation, the
  filter `Select` overlay, and the composer **Send** button are driven by real
  pixel-accurate mouse clicks computed from the xterm buffer geometry.

None of these change what the Textual app sees vs. a real terminal.

## Captures (2050×1240)

1. **`picker-image-files-extended.png`** — the Console composer **Attach** button
   (`#console-attach-context`) opens the "Select File to Attach" picker; breadcrumb
   at `/private/tmp/tldw-qa-config-caps-20260714 / attachments`; the filter
   `Select` set to **"Image Files"**. The listing shows **`logo.svg`, `pic.png`
   AND `scan.tiff`** all visible — the config-driven "Image Files" filter now
   advertises tiff/svg (old behavior hid/rejected them). Verified by buffer read:
   filter label reads "Image Files" and all three names present.
2. **`tiff-staged-chip.png`** — `scan.tiff` picked through the real picker →
   staged. Composer shows the persistent indicator **`📎 scan.tiff · 326 B`**,
   the Attach button flips to **`📎✓`** with a **✕** clear button. `326 B` is
   exactly the transcoded-PNG size (matches the headless check) — proof the TIFF
   is transcoded to a payload-safe PNG at staging time.
3. **`tiff-sent-inline.png`** — message "Scanned TIFF document" + `scan.tiff`,
   **Send**. Transcript posts the User message with chip **`🖼 scan.tiff · 326 B`**
   and the **inline (pixels) render of the transcoded PNG** — the exact two-tone
   drawing (orange top band, teal bottom band, white centre bar) is visible.
   Assistant `[failed]` + System "Server error '500 …'" is the honest non-vision
   llama.cpp response.
4. **`svg-sent-rasterized.png`** — message "Company logo SVG" + `logo.svg`,
   **Send**. Transcript posts the User message with chip **`🖼 logo.svg · 2 KB`**
   (the ~1.8 KB rasterized PNG, not the 309 B SVG source) and the **inline render
   of the cairosvg rasterization** — the blue rectangle + white circle + blue
   inner square are clearly drawn, proving cairosvg rasterization is live
   end-to-end. Assistant `[failed]` (same honest 500).

## Defects

**None.** The TASK-222 feature works end-to-end at HEAD b578c3b1: config governs
the allowlist/filters/caps, and tiff/svg are advertised, accepted, transcoded/
rasterized, staged, sent, and rendered inline.

## Observations / notes (not defects)

- **Assistant `[failed]` in #3/#4** is expected: the seeded model is a real
  non-vision llama.cpp GGUF; the vision override only lets the *send* pass the
  pre-flight gate, and the server 500s on the image payload. This is downstream
  of attachment handling and identical to the 215/217 walks. The User message,
  chip, and inline render (the feature under test) all succeed regardless.
- **Picker Open/Cancel buttons** are not visible in the buffer at this dialog
  width (the `InputBar` filename Input + filter Select consume the row); the
  file is confirmed via Enter (`Input.Submitted`) instead. Pre-existing picker
  layout characteristic, unrelated to the config-caps feature.
- **Rail decorative rows**: after the send captures the left-rail "Chats" section
  shows the just-created conversations (e.g. "Company logo SVG", "Scanned TIFF
  docu…") — real DB rows from these sends; each fresh app boot still opens an
  empty "Chat 1" transcript.

## Reproduce

Scripts (session scratchpad, not committed) under
`…/scratchpad/caps222/`: `setup_home.py` (HOME/config/test-files),
`verify_pipeline.py` (headless config+transcode+rasterize proof), `serve_caps.py`
(serve launcher), `drv.py` (geometry + ws-stdin driver with picker helpers),
`cap1.py`…`cap4.py` (the four captures). QA HOME under
`/private/tmp/tldw-qa-config-caps-20260714`.

## Verdict

**CLEAN.** All four required captures delivered and visually verified;
config-driven filters/caps and tiff+svg attachment support are proven live in
the Console attach flow. No defects.
