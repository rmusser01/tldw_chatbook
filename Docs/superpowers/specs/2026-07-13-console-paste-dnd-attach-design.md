# Console Clipboard Paste + Drag-Drop Attach (TASK-216) — Design

- **Date:** 2026-07-13
- **Status:** Approved pending user spec review
- **Scope anchor:** TASK-216 — clipboard image paste and drag-drop attach in the Console composer, routing through `attachment_core` and the per-session pending-attachment state (both from PR #621).

## Terminal reality (drives the whole design; user-approved reinterpretation of the ACs)

- **Drag-drop** onto a terminal pastes the file's PATH as text (bracketed paste). "Drop support" therefore = intercepting path-like pastes.
- **Clipboard images** produce NO terminal paste event. Reading them requires an explicit trigger calling `PIL.ImageGrab.grabclipboard()` — verified working on this platform (returns a PIL image, a list of file paths for Finder-copied files, or None); unavailable on most Linux setups → honest toast.

## Decisions (user-approved)

| Decision | Choice |
|---|---|
| Mechanisms | Both: path-paste interception (drag-drop + copied paths) AND explicit clipboard-image grab. |
| Path-paste UX | Auto-attach + easy undo: a single pasted path that exists, is in-root, and matches supported types routes straight to the attach pipeline (toast + existing ✕ undo). Prose containing paths, multi-line text, unknown/missing paths stay text. |
| Image trigger | Alt+V on the composer (kitty-safe, mirrors alt+m precedent; verified unbound) + a command-palette entry. No new composer button. |
| Architecture | Pure helper module + a bytes entry in attachment_core (Approach A). |

## Components

### New pure module — `tldw_chatbook/Chat/console_paste_attach.py`

- `extract_dropped_path(pasted_text) -> DroppedPaste | None` where `DroppedPaste(path: str, total_dropped: int)`:
  returns a candidate when the paste is a single path-like token — handles single/double-quoted forms, backslash-escaped spaces, `file://` URIs (percent-decoded), trailing newline. Multi-path drops (newline-separated paths, all path-like) yield the FIRST path with `total_dropped=N` so the screen can toast "attached first of N" (single-attachment constraint from Phase 1). Any paste containing non-path prose returns None. Pure string logic; no filesystem access.
- `looks_attachable(path, allowed_root) -> bool` — `os.path.exists` + `path_validation.is_safe_path(path, allowed_root)` + extension matches the shared `ATTACHMENT_FILTER_SPECS` union. **Known inherited wart:** the specs still advertise tiff/svg that the image pipeline rejects (TASK-222 drift) — a dropped `.tiff` auto-attaches then error-toasts, identical to picking it in the picker; TASK-222 fixes both at the source.
- `grab_clipboard_image() -> ClipboardGrab` where `ClipboardGrab(kind: Literal["image","paths","empty","unavailable"], png_bytes: bytes | None, paths: list[str])`:
  wraps `ImageGrab.grabclipboard()`; PIL image → PNG-encoded bytes; list-of-paths (Finder file copies) → paths for the path pipeline; None → empty; ImportError/OSError → unavailable. Sync/blocking — callers run it off-loop.

### `attachment_core` addition

- `process_attachment_bytes(data: bytes, *, display_name: str, mime_type: str = "image/png") -> PendingAttachment` — the bytes twin of `process_attachment_path` for clipboard images: PIL validate, resize cap (`resize_max_dimension` semantics via the same `ChatImageHandler._process_image_data` path or equivalent core logic), 10 MB image cap, no temp files. `file_path=""`, `display_name` like `clipboard-YYYYmmdd-HHMMSS.png`.

### Screen wiring — `chat_screen.py`

- `on_paste` (currently line 8305): AFTER the existing setup-modal and `_should_capture_console_input` guards (verified — interception inherits focus/modal semantics for free), BEFORE `insert_pasted_text`: if `extract_dropped_path` yields a candidate and `looks_attachable` passes → `event.stop()`, dismiss guidance, and route into the existing `_process_console_attachment(path)` worker (same toasts, indicator, ✕ undo, control-bar sync as the picker path; skip `_sync_console_workbench_actions_from_draft` — no draft changed); multi-drop adds the "attached first of N" toast. Otherwise the handler proceeds unchanged.
- Alt+V: composer-scope binding + command-palette entry → screen handler runs `grab_clipboard_image` via `asyncio.to_thread` in worker group `"console-clipboard-grab"` (`exclusive=True`; dedicated group per the PR #621 lesson) → `image` → `process_attachment_bytes` (also off-loop) → stage pending + indicator; `paths` → route first through the path pipeline; `empty` → toast "No image on the clipboard."; `unavailable` → toast "Clipboard images aren't readable on this platform — use Attach or drop a file."

### Composer

No changes — Phase 1's pending indicator/✕ covers undo for both new routes.

## Edge cases

- Paste while a pending attachment exists: replace-on-reattach (existing store semantics).
- Path pastes that stay text: multi-line prose, non-existent paths, out-of-root, unsupported extensions, anything with surrounding words.
- Non-image pending from a drop (e.g. dropped .md): identical to picking it — inline collapsed segment route.
- ImageGrab needing no permissions on macOS pasteboard; failures logged + unavailable toast, never a crash.
- textual-serve/browser: OS clipboard unreachable from the server process's ImageGrab in a meaningful way for the browser user; QA documents this and drives paste-interception via real bracketed paste, clipboard route via monkeypatched grab (disclosed honestly in QA README).

## Testing

1. Pure: extraction matrix (quoted/escaped/URI/multi-drop/prose/trailing-newline), `looks_attachable` gates (missing/out-of-root/unsupported), `grab_clipboard_image` kind mapping (monkeypatched ImageGrab), `process_attachment_bytes` (resize cap, corrupt bytes ValueError).
2. Mounted: path-paste → pending staged + no draft text; prose-paste → unchanged text path; Alt+V happy path (monkeypatched grab) → pending staged; unavailable → toast; multi-drop first-of-N toast.
3. Standing visual gate: textual-serve captures (drop-as-paste staging, clipboard toast paths) + user approval before merge.

## Out of scope

Multi-attachment staging (TASK-217 — first-of-N is the bridge); filter/caps config unification (TASK-222); Linux clipboard backends (xclip/wl-paste — a follow-up if demand exists); any legacy chat changes.

## Key file touch list

| File | Change |
|---|---|
| `Chat/console_paste_attach.py` | **New** — extraction, attachability, clipboard grab |
| `Chat/attachment_core.py` | `process_attachment_bytes` |
| `UI/Screens/chat_screen.py` | on_paste interception, Alt+V handler + palette entry, grab worker |
| `Widgets/Console/console_composer_bar.py` | Alt+V binding surface only if composer-scoped bindings require it (else screen-level) |
