# Config-Driven Attachment Filters and Image Caps (TASK-222) — Design

- **Date:** 2026-07-14
- **Status:** Approved pending user spec review
- **Scope anchor:** TASK-222 — drive attachment filters and image caps from `[chat.images]` config; fix the tiff/svg picker-vs-pipeline drift. Applies to both Console (`chat_screen.py`) and legacy Chat (`chat_attachment_handler.py`) because they share `attachment_core` and `ChatImageHandler`.

## Decisions (user-approved)

| Decision | Choice |
|---|---|
| tiff/svg drift direction | **Extend both**: `.tiff`/`.tif` become real pipeline formats (PIL-native decode + payload transcode); `.svg` becomes a real format via cairosvg rasterization. (User chose this over narrowing the picker.) |
| SVG availability | **Capability-gated**: `.svg` is dropped from the effective format list when cairosvg is unavailable — picker never advertises it, routing never selects it. cairosvg ships as a new optional extra, registered in `optional_deps.py`. |
| macOS native dep | User authorized and completed `brew install cairo`; cairosvg installed in the dev venv. Verified: `svg2png` produces a PIL-readable PNG. |
| Existing configs | **Present key governs** (standard semantics). Existing configs (including the live user config, which pins the old 6-format list) keep their pinned list until edited via config.toml or Settings → Chat Images. No live-config mutation by us, ever. Union-with-defaults rejected: it would make formats impossible to remove. New installs get the extended default list. |

## Verified constraints (bind the implementation)

1. `Tests/Event_Handlers/Chat_Events/test_chat_image_events.py:349` asserts `ChatImageHandler.MAX_IMAGE_SIZE == 10 * 1024 * 1024`. The class constants stay as unchanged default literals; config-driven behavior comes from new call-time policy functions that default to them.
2. `test_chat_image_events.py:248` calls `ChatImageHandler._process_image_data(data, ext, mime) -> bytes` directly (and asserts 2048px + shrink for a `.jpg`). Its signature and bytes-only return are pinned; it must not change shape.
3. `test_chat_image_events.py` `test_process_large_image_resize` pins ≤ 2048 under default config; `test_process_valid_image` pins `image/png` mime for a png. Default-config behavior must remain byte-identical for png/jpg/jpeg/gif/webp inputs.
4. `Tests/UI/test_chat_image_attachment.py:36` constructs its own local `Filters` literal — it does NOT import production specs and stays green unedited. No gate test asserts tiff/svg rejection. The legacy regression gate (AC #3) requires **zero edits** to existing tests.
5. No test imports `ATTACHMENT_FILTER_SPECS` or `console_paste_attach._SUPPORTED_PATTERNS`; production consumers are `console_paste_attach.py:24-29` (import-time — must become call-time), `chat_attachment_handler.py:77-80` and `chat_screen.py:7757-7769` (already call-time contexts).
6. ~~`get_cli_setting("chat.images", key, default)` is the canonical accessor~~ **FALSIFIED at final review (C1 Critical):** `get_cli_setting` does a FLAT `config.get(section)` lookup and never resolves the nested `[chat.images]` TOML table — the dotted-section call shape silently returns the default, always. The working shape is `get_cli_setting("chat", "images", None)` with the key resolved locally; all policy reads go through `attachment_core._chat_images_setting(key, default)`, and an UNMOCKED integration test (scratch TOML via the config-path env override, no accessor patching) pins the real read path. Pre-existing callers elsewhere using the dotted shape (`show_attach_button`, `save_location` readers from #621/#626) are broken on dev — follow-up backlog task, out of scope here.
7. cairocffi cannot find Homebrew's libcairo via plain `dlopen`, and `ctypes.CDLL` preload does NOT fix it (verified). Verified fix: append `/opt/homebrew/lib` to `os.environ["DYLD_FALLBACK_LIBRARY_PATH"]` in-process **before the first cairosvg import** — `ctypes.util.find_library` (cairocffi's fallback) reads the environment at call time.

## Latent bugs this design repairs (discovered during verification)

- **Bytes/mime mismatch on resize:** `process_image_file` returns the path-guessed mime even when `_process_image_data`'s else-branch converts resized bytes to PNG (e.g. a large `.gif` returns PNG bytes labeled `image/gif`).
- **Provider-unsafe passthrough:** a small `.bmp` (and, post-extension, `.tiff`) passes through unconverted with `image/bmp`/`image/tiff` mime — formats vision providers reject. Both are fixed by the payload normalization step below; without it, "supporting tiff" would mean attaching files that fail at send time.

## Components

### 1. Policy functions — `Chat/attachment_core.py`

All call-time (no import-time config reads), all reading via `get_cli_setting("chat.images", key, default)`:

- `DEFAULT_SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".svg")` — the single default-list source.
- `supported_image_formats() -> tuple[str, ...]` — reads `supported_formats` (default: the tuple above); normalizes each entry (lowercase, prepend `.` if missing, strip whitespace), dedupes preserving order, drops non-string entries with a logged warning; **drops `.svg` when cairosvg is unavailable** (the capability gate). Invalid/empty config value → default list + warning.
- `max_image_bytes() -> int` — `max_size_mb` (default 10.0) × 1024 × 1024, int. Non-numeric or ≤ 0 → default + warning. `MAX_IMAGE_BYTES = 10 * 1024 * 1024` stays as the default constant.
- `image_resize_max_dimension() -> int` — `resize_max_dimension` (default 2048). Non-numeric or ≤ 0 → default + warning.
- `attachment_filter_specs() -> tuple[tuple[str, str], ...]` — replaces the module constant `ATTACHMENT_FILTER_SPECS`. The "Image Files" row's pattern is built from `supported_image_formats()`, and the **"All Supported Files" row's image segment derives from the same call** (verified gap: that union row also hardcodes tiff/tif/svg today at `attachment_core.py:31`); the non-image rows (text/code/data files) keep today's literals verbatim. Same `(label, "*.a;*.b")` shape.

`ChatImageHandler.MAX_IMAGE_SIZE` and `.SUPPORTED_FORMATS` remain as class-constant defaults (test-pinned); handler *behavior* reads the policy functions.

### 2. Payload normalization — `Event_Handlers/Chat_Events/chat_image_events.py`

New `ChatImageHandler.prepare_image_payload(image_data: bytes, extension: str) -> tuple[bytes, str]` (async, staticmethod — the single pipeline):

1. **SVG branch:** if `extension == ".svg"` and cairosvg is available → rasterize with a **bounded output size** (see below); data is now PNG, extension `.png`. (Rasterization happens before PIL open — PIL cannot read SVG.) cairosvg errors → `ValueError` with a clear message (invalid SVG is a rejection, not a crash).
   - **Raster bound (verified necessity):** the 10 MB size cap applies to the SVG *source*; a small SVG declaring `width="100000"` would make cairo allocate the giant surface before PIL's bomb guard runs. So: read the root element's aspect via `defusedxml.ElementTree` (ships as a cairosvg dependency — available whenever the branch is) from `viewBox`, else numeric `width`/`height`; call `svg2png` with `output_width=cap` when wide or `output_height=cap` when tall, where cap = `min(intrinsic longer side, image_resize_max_dimension())` — single-dimension calls preserve aspect (verified, incl. viewBox-only SVGs). If no aspect is parseable (e.g. percentage sizes without viewBox), pass **both** `output_width=cap, output_height=cap` — a hard memory bound that may distort in that degenerate case (logged). PIL's decompression-bomb guard remains the post-render backstop (caught → rejection, not crash).
   - **Safety posture (verified live):** `unsafe` stays at its default `False` — XML entities are hard-blocked (`EntitiesForbidden`) and external `file://` image refs are not read (probe: referenced file's pixels never appear in the raster). Never pass `unsafe=True`; a test pins the entity rejection.
2. PIL open; resize with `thumbnail(...)` if either dimension exceeds `image_resize_max_dimension()` (LANCZOS, aspect preserved; save kwargs exactly as today: PNG optimize / JPEG optimize+q85 / WEBP q85 / else PNG).
3. **Payload-safe transcode:** if the resulting bytes' actual format (PIL `format`) is not in `{PNG, JPEG, WEBP, GIF}` → re-encode PNG.
4. Return `(bytes, mime)` where mime is derived from the **actual final bytes** (fixes both latent bugs).

Under default config, png/jpg/jpeg/gif/webp inputs that don't need resizing return original bytes with the same mime as today — byte-identical, keeping the gate green.

- `process_image_file` rewires: extension allowlist via `supported_image_formats()` (error copy unchanged: `"Unsupported image format: ..."` listing the effective formats), size check via `max_image_bytes()` (copy unchanged), then `prepare_image_payload`. Its existing fall-back-to-original-bytes on processing failure is kept for non-SVG inputs; for `.svg` there is nothing usable to fall back to, so the error propagates as a rejection.
- `_process_image_data` stays with its pinned signature as a thin adapter over steps 2–3 returning bytes only (gate-pinned callers unaffected; the only behavioral delta — a small `.bmp` now transcodes to PNG — is exercised by no existing test and repairs the provider-rejection bug).
- `attachment_core.process_attachment_bytes` rewires its size check to `max_image_bytes()` and its processing call to `prepare_image_payload(data, extension="")` (bytes path has no extension → SVG branch never triggers; clipboard flow already produces PNG), keeping the #628 fallback-to-original semantics. The returned mime is the payload function's actual-bytes mime — today the function echoes the caller's `mime_type` param even when `_process_image_data` re-encoded the bytes (`attachment_core.py:195`), the same latent-mismatch family, closed by the same fix. On fallback-to-original, mime is derived from the original bytes.

### 3. Routing and pickers

- `Utils/file_handlers.py` `ImageFileHandler.SUPPORTED_EXTENSIONS` (line 59, today's superset with tiff/tif/svg) → `can_handle` reads `supported_image_formats()` at call time **via a function-local import** — `attachment_core` imports `file_handlers` at module level (`attachment_core.py:18`), so a module-level back-import would be a cycle; `ImageFileHandler.process` already lazily imports `chat_image_events` for exactly this reason (same pattern, same class). Routing == pipeline == picker, by construction.
- `Chat/console_paste_attach.py` — `_SUPPORTED_PATTERNS` import-time constant becomes a call-time helper `_supported_patterns()` over `attachment_filter_specs()`; `looks_attachable()` unchanged otherwise (case-insensitive matching from #628 preserved).
- `UI/Screens/chat_screen.py:7757-7769` and `UI/Chat_Modules/chat_attachment_handler.py:77-80` — swap `ATTACHMENT_FILTER_SPECS` for `attachment_filter_specs()` (both already build filters inside the method).
- `UI/Tools_Settings_Window.py:1890` — the hardcoded fallback list imports `DEFAULT_SUPPORTED_IMAGE_FORMATS` instead of repeating the literal.
- `config.py` `CONFIG_TOML_CONTENT` `[chat.images] supported_formats` template line (≈2057) gains `".tiff", ".tif", ".svg"` — matching the policy default exactly (a drift-by-construction test asserts this equality).

### 4. cairosvg optional dependency

- `pyproject.toml`: new extra `svg = ["cairosvg"]` (repo snake_case convention).
- `Utils/optional_deps.py`: new `DEPENDENCIES_AVAILABLE['svg_rendering']` key + `ensure_svg_rendering() -> bool` (cached). The check: on darwin, if `/opt/homebrew/lib` exists and is not already in `DYLD_FALLBACK_LIBRARY_PATH`, append it to `os.environ` **before** attempting `import cairosvg` (verified fix; harmless no-op elsewhere). Import success/failure sets the flag once. `attachment_core` consults it through a module-level seam `svg_rendering_available()` that delegates to `ensure_svg_rendering()` — the seam unit tests mock.
- Belt-and-suspenders: if an `.svg` somehow reaches the handler while cairosvg is unavailable (stale picker, direct path), the extension check already rejects it (svg absent from effective formats) — clean `ValueError`, no crash.

## Semantics notes

- The effective format list is what the picker advertises, what routing selects, what the pipeline accepts. There is no second list anywhere.
- Existing configs keep their pinned `supported_formats` until edited (see Decisions). Release-note line + post-merge pointer to the one-line edit for the user's live config.
- Config is read call-time on each attach action — attach-frequency, not hot-path; no caching, no invalidation problem. Settings changes apply on the next attach without restart.

## Testing

1. **Policy units** (`Tests/Chat/test_attachment_policy.py`, new): defaults; normalization (`"PNG"` → `.png`, `"jpg"` → `.jpg`, dedupe, junk entries); override via patched `get_cli_setting`; `max_image_bytes`/`image_resize_max_dimension` overrides and invalid-value fallbacks; svg dropped when `svg_rendering_available()` is False (seam mocked), present when True.
2. **Drift-by-construction**: parse `CONFIG_TOML_CONTENT` → `[chat.images].supported_formats == list(DEFAULT_SUPPORTED_IMAGE_FORMATS)`; `attachment_filter_specs()` "Image Files" patterns == `supported_image_formats()`; `ImageFileHandler` routing accepts exactly the effective formats.
3. **Pipeline**: tiff end-to-end (real PIL-generated `.tiff` through `process_image_file` → PNG bytes + `image/png` mime); svg end-to-end (real svg through rasterize → PNG; `pytest.importorskip`-gated — the dev venv has cairosvg); large-gif resize now returns matching bytes/mime; small-bmp transcodes to PNG; custom `max_size_mb` rejects at the new bound; custom `resize_max_dimension` honored.
   - **SVG safety** (importorskip-gated): oversize-declared SVG (`width="100000"`) rasterizes bounded (result ≤ resize cap, no bomb); XML-entity SVG → clean `ValueError` rejection (pins `unsafe=False`); viewBox-only SVG preserves aspect; unparseable-aspect SVG hits the both-dims hard-bound fallback.
4. **Config-driven caps through the consumer path**: at least one test drives a non-default `max_size_mb`/`supported_formats` through `process_attachment_path`/`process_attachment_bytes` (the real call path, per the T217 P0 lesson — no `**kwargs` fakes).
5. **Legacy regression gate**: `test_chat_image_events.py`, `test_chat_image_properties.py`, `test_chat_image_attachment.py`, `test_chat_image_unit.py` — green, **zero edits** (AC #3).
6. **QA + user gate**: textual-serve captures — picker "Image Files" row advertising the extended list; a `.tiff` attach → chip → send; an `.svg` attach → chip (rasterized); Console screenshot approval before merge (standing rule).

## Out of scope

Per-provider payload format negotiation (PNG-everywhere is safe); animated-gif frame handling; `[chat.images.terminal_overrides]`; caching policy reads; migrating existing user configs; `max_history_images`/staging-cap config-driving (staging cap 5 stays a named constant — different task); legacy `_process_image_data` removal.

## Key file touch list

| File | Change |
|---|---|
| `Chat/attachment_core.py` | policy functions, `DEFAULT_SUPPORTED_IMAGE_FORMATS`, `attachment_filter_specs()`, rewired `process_attachment_bytes` |
| `Event_Handlers/Chat_Events/chat_image_events.py` | `prepare_image_payload`, SVG branch, rewired `process_image_file`, `_process_image_data` adapter |
| `Utils/file_handlers.py` | call-time routing via `supported_image_formats()` |
| `Chat/console_paste_attach.py` | `_supported_patterns()` call-time |
| `UI/Screens/chat_screen.py`, `UI/Chat_Modules/chat_attachment_handler.py` | `attachment_filter_specs()` swap |
| `UI/Tools_Settings_Window.py` | fallback imports the default constant |
| `config.py` | template default list extended |
| `Utils/optional_deps.py` | `svg_rendering` registration + darwin dyld fix |
| `pyproject.toml` | `svg = ["cairosvg"]` extra |
