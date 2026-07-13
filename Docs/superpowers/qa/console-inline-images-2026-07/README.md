# Console inline image rendering — live QA evidence (2026-07-13)

TASK-215. Worktree `console-inline-images-215` @ **fd3f0972**
("feat(console): wire inline image rendering — prep worker, toggle handler,
screen state"). Real streamed sends to a live model; no fixtures.

## Rig

- **Serve**: `textual_serve.Server` running `python -m tldw_chatbook.app` from
  the worktree (real app TCSS). Raw Server (not the app's `Web_Server/serve.py`
  wrapper) so I could serve a **patched `textual.js`** (`statics_path`) that
  stashes the xterm driver on `window.__drv`. That handle lets the Playwright
  driver read the full screen via the xterm **buffer API**
  (`terminal.buffer.active.getLine(y).translateToString()`) and compute cell
  geometry (`cols/rows` + `.xterm-screen` rect) for pixel-accurate clicks. The
  browser-side renderer (xterm canvas) is a display detail only; the Textual
  app renders identical output either way, so this is faithful.
- **Browser**: bundled Playwright chromium, headless, viewport **2050x1240**,
  device_scale_factor 1. Cell grid observed: 227x59, cellW 9, cellH 21.
- **Isolated HOME**: `/private/tmp/tldw-qa-inline-20260713`
  (`HOME` + `XDG_CONFIG_HOME` + `XDG_DATA_HOME`). Serve cwd = HOME, so the file
  picker opens there. HOME holds only `gradient.png` (512x256 colour gradient,
  ~6 KB) and `red-square.png` (64x64, 184 B); driver scripts live in a sibling
  `/private/tmp/tldw-qa-driver` so they don't clutter the picker.
- **Serve port**: 9131. Driver scripts run one continuous browser connection
  per app process (textual-serve spawns one app per websocket), so captures
  1/2/3/4/6 are ONE process; capture 5 is a fresh process after a serve restart.
- **Config seed** (`~/.config/tldw_cli/config.toml`, deep-merged over defaults):
  splash off; `[console.onboarding] first_send_completed=true`;
  `[chat_defaults] provider="llama_cpp" model="Qwen3.6-…gguf"`;
  `[chat.images] default_render_mode="pixels"`;
  `[api_settings.llama_cpp] api_url="http://127.0.0.1:9099"`;
  vision override `[model_capabilities.models]."Qwen3.6-…gguf"={vision=true,max_images=5}`.
  (The app also auto-wrote `[filepicker]` bookmarks/recents into this file.)
- **TERM neutralised**: serve launched with `TERM=xterm` (no `256color`) and
  `ITERM_SESSION_ID`/`VTE_VERSION`/`WT_SESSION` unset, so the served app's
  terminal detection resolves `auto` → **pixels** (see Defect 1 for why this
  matters).
- **Provider**: live **llama.cpp @ 127.0.0.1:9099**
  (`Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf`, NON-vision, no
  mmproj). Every image send is rejected with **HTTP 500** — the anticipated,
  honest failure. The user image message (chip + inline render) still lands,
  which is all these captures need.

## Captures

1. **inline-pixels-default.png** — `gradient.png` + "Describe this" sent for
   real. User message renders the `🖼 gradient.png · 6 KB` chip and, below it,
   the gradient **inline in pixels mode** (rich-pixels half-blocks, full
   colour). Assistant `[failed]` + System "Provider stream failed: … HTTP 500
   … http://127.0.0.1:9099/v1/chat/completions" — the expected vision reject.
   The image appeared within a moment of the chip (off-loop prep worker).
2. **toggle-graphics.png** — message selected, **View** clicked once → graphics
   mode. Under textual-serve, `textual-image` has no sixel/kitty/iterm graphics
   protocol available, so it **negotiates down to a monochrome half-cell /
   dithered rendering** (expected and documented). Visibly distinct from the
   colour pixels mode.
3. **toggle-hidden-chip-only.png** — **View** clicked again → hidden. The inline
   image row is gone; only the `🖼 gradient.png · 6 KB` chip remains. The action
   row (with **View** focused) is shown below the chip.
4. **image-with-actions-row.png** — the image message selected: the inline
   action row shows Copy / Edit / Save as… / ♻ / ---> / 👍 / 👎 / 🗑 / **View** /
   **Save Image**. Note the spec-flagged consequence: the action row sits
   **below** the inline image (you scroll past the image to reach it — see
   Observation 3).
5. **resume-inline-rehydrated.png** — serve killed + restarted (**fresh app
   process**), conversation resumed from the rail ("Describe this"). Both image
   messages **re-prep from DB bytes and render inline again** at the config
   default (pixels); the previous hidden/graphics view choices were reset by the
   relaunch (screen state, not persisted). The rehydrated chips read
   `🖼 image/png · 6 KB` / `🖼 image/png · 184 B` (generic mime, not the original
   filename — see Observation 4).
6. **two-images-modes.png** — a second image message (`red-square.png`) sent in
   the same transcript while the first (gradient) is **hidden**: the first shows
   chip-only, the second renders inline in pixels. Proves **per-message** view
   state.
7. **config-regular-default.png** — **live proof that `[chat.images].**
   **default_render_mode` now drives the default** (config-shape defect fixed in
   **5b2a7e26**). Fresh app process, isolated HOME config set to
   `default_render_mode = "regular"`, TERM neutralised to `xterm` (so terminal
   `auto` would resolve to **pixels/colour** — see Defect 1). The resumed
   "Describe this" gradient re-preps from DB and renders inline in **graphics**
   mode **by default, with no manual toggle** — the monochrome dithered
   half-cell look, **pixel-identical to `toggle-graphics.png` (#2)** and clearly
   distinct from the full-colour `inline-pixels-default.png` (#1). The config,
   not the terminal, decided the mode. Sanity cross-check under the same
   isolated HOME + neutral TERM: `resolve_default_mode(load_settings())` →
   `graphics` while terminal-only `get_image_render_mode("auto")` → `pixels`
   (terminal_type `xterm`).

## Observations / defects

1. **DEFECT (P2) — FIXED in 5b2a7e26 (live-proven by capture #7,
   `config-regular-default.png`).** Originally: `[chat.images].default_render_mode`
   and `terminal_overrides` were silently ignored in the running app.
   `resolve_default_mode`
   (`Chat/console_image_view.py`) reads `app_instance.app_config`, but the app
   sets `app_config = load_settings()` (`app.py:2431`), whose return nests the
   raw TOML under `COMPREHENSIVE_CONFIG_RAW` and exposes **no top-level `chat`
   key**. So `app_config.get("chat")` is `None`, the configured
   `default_render_mode` is never seen, and resolution always falls through to
   `auto` (terminal detection). Repro:
   `resolve_default_mode(load_settings())` → `graphics` under my shell TERM but
   `pixels` for `resolve_default_mode(load_settings()["COMPREHENSIVE_CONFIG_RAW"])`.
   This mirrors the exact both-shapes bug that `config.resolve_tldw_api_config`
   (config.py:601) was written to fix. **Consequence for QA**: the pixels
   default in capture 1 comes from the `auto` path, achieved by neutralising
   TERM in the serve env (a non-iTerm host resolves `auto`→pixels). An explicit
   config value would NOT change it. Non-blocking — the observable default is
   still pixels — but the headline "these config keys now actually work" claim
   is not true through `app_config` as wired.
   **FIX (5b2a7e26):** `console_image_view._chat_images_config` now reads
   `[chat.images]` through the live `COMPREHENSIVE_CONFIG_RAW` shape (falling
   back to the plain-dict shape), so the configured value is honored.
   **Live proof (capture #7):** with `default_render_mode = "regular"` and TERM
   neutralised to `xterm` (so `auto` would pick pixels/colour), the resumed
   gradient renders inline in **graphics by default with no toggle** — matching
   `toggle-graphics.png` and unlike the colour `inline-pixels-default.png`. The
   config, not the terminal, now drives the default.
2. **Related — `[api_endpoints].llama_cpp` is likewise ignored.** The Console
   provider gateway reads `api_settings.llama_cpp.api_url` (not `[api_endpoints]`),
   and only `api_settings` survives into the normalised `app_config`. Seeding
   `[api_endpoints]` left the gateway probing the default `localhost:8080`
   ("Provider blocked: … not reachable"); moving the endpoint to
   `[api_settings.llama_cpp] api_url` fixed it. Config-plumbing note, not part of
   this feature.
3. **Observation (spec-flagged) — action row sits below the inline image.**
   Because the action row renders after the (tall) inline image, once the image
   is in pixels/graphics mode the row is scrolled below the fold; you must
   scroll the transcript down to reach **View**/**Save Image**. This also means
   re-toggling requires scrolling to the row again (the message stays selected
   across a toggle; the row is not lost, just off-screen). Worth a product look.
4. **Observation — rehydrated chip loses the filename.** After resume the chip
   shows a synthesised `image/png · <size>` label rather than the original
   filename, because the DB never stored the attachment filename. Consistent
   with the prior attachments phase (its Defect 1 fix synthesises mime+size).
5. **Observation — HTTP 500 on every image send** is the anticipated honest
   failure (Qwen has no mmproj); the failed assistant/system rows do not
   rehydrate on resume (only the user image messages do), matching the prior
   phase.
