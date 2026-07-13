# Console native attachments Phase 1 — live QA evidence (2026-07-12)

Branch: console-attachments-phase1 worktree, HEAD fbe47b77 ("fix(console):
harden Save Image worker"). Captured from textual-serve (real app CSS,
`textual serve` of `python -m tldw_chatbook.app`) in headless bundled
Playwright chromium (1.58.0), viewport 2050x1240, isolated HOME
`/private/tmp/tldw-qa-attach-20260712` (HOME + XDG_DATA_HOME +
XDG_CONFIG_HOME), ready-seeded config (Llama_cpp provider, splash off,
`console.onboarding.first_send_completed = true`), live llama.cpp server at
http://127.0.0.1:9099 (Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf,
NON-vision, reports `capabilities: ["completion"]`). Serve ports 9107 then
9108 (restarted before the resume capture). Real streamed sends, no fixtures.
The serve process cwd was the isolated HOME so the picker (`location="."`)
opens inside the allowed root; test files were created there:
`red-square.png` (64x64 PIL red square, 184 B) and `zorblatt-notes.md`
(207 B of made-up "zorblatt" facts the model cannot know).

Each capture session is a fresh app process (textual-serve spawns one per
browser connection), so state that persists across captures below is DB/config
persistence, not process memory.

## Captures

Phase A — no vision override (Qwen is non-vision):

- attach-picker-open.png — composer Attach pressed: native EnhancedFileOpen
  modal "Select File to Attach" over the Console, breadcrumbs inside the
  isolated HOME, both test files listed, file-type filter dropdown at the
  bottom showing "All Supported Files". (The old always-erroring bridge is
  gone — the button opens a real picker.)
- image-staged-indicator.png — red-square.png picked (click row + Enter):
  composer shows the `📎 red-square.png · 184 B` indicator, the `✕` clear
  button, and the attach button relabeled `📎✓`. Send rendered subdued
  (blocked). A "red-square.png attached" toast appears on staging (allowed to
  expire before this shot).
- image-blocked-send.png — hover over Send while the image is staged: tooltip
  with the full blocked reason "Console send blocked: Qwen3.6-27B-…gguf can't
  accept images. Remove the attachment, switch to a vision model, or mark
  this model as vision-capable under [model_capabilities.models] in
  config.toml."
- image-blocked-send-system-row.png — Send clicked while blocked: the same
  copy lands as a visible System row in the transcript; nothing is sent
  (token counter only reflects the local system text; no provider call).
  A separate errant run (wrong ✕ coordinate in the driver script, see
  Deviations) also proved the block holds when the draft additionally
  contains text + an inlined file: a second System row was appended and no
  provider call was made.
- attachment-cleared.png — `✕` pressed: indicator and `✕` gone, attach button
  label restored to "Attach", "Attachment cleared" toast (expired before this
  shot).
- text-inline-segment.png — zorblatt-notes.md attached: text file is NOT
  staged as an attachment; it is inlined into the draft as a collapsed cyan
  labeled segment `📄 zorblatt-notes.md · 278 B` (278 B = processed content
  with the "--- Contents of … ---" framing, raw file is 207 B). Send primary.
- text-inline-prompt.png — a literal prompt typed after the collapsed
  segment: "In one short sentence, what is a zorblatt according to the
  attached note?" (block caret visible; segment stays collapsed).
- text-inline-response.png — REAL send to llama.cpp: the user message shows
  the full inlined file text (proving the collapsed token carried the whole
  payload), and the real streamed assistant answer reads "According to the
  note, a zorblatt is a fictional nocturnal bird that hoards blue buttons and
  sings only during thunderstorms." — content the model can only know from
  the attached file. Stop hidden again after completion; conversation saved
  to the rail. (Answer captured at t≈138 s after Send; see Deviations for the
  reasoning-model latency.)

Phase B — vision override enabled in the seeded config
(`[model_capabilities.models]."Qwen3.6-…gguf" = { vision = true,
max_images = 5 }`), app process restarted:

- vision-override-staged-not-blocked.png — same staged red-square.png, but
  with the override active Send renders PRIMARY (no block, no tooltip):
  the blocked state is capability-driven, matching the "clears when the
  model becomes vision-capable" contract (evidenced across an app restart
  with the config override; not via a live in-session capability flip).
- image-chip-sent.png — image + "Describe this image briefly." sent for
  real: user message renders the `🖼 red-square.png · 184 B` chip line.
  llama.cpp (no mmproj loaded) rejected the multimodal payload with HTTP 500
  — the anticipated acceptable outcome — and the failure UX is honest:
  Assistant `[failed]`, System row "Provider stream failed: provider
  returned HTTP 500 … /v1/chat/completions", Inspector shows "failed".
  The user image message with its chip persists in the transcript.
- image-chip-save-action.png — the user image message selected (focus
  background + underline, non-obscuring contract): inline action row shows
  Copy / Edit / Save as… / regenerate / continue / rate / delete / and the
  image-specific "Save Image" action.
- resume-chip-rehydrated.png — textual-serve killed and restarted (new
  server on :9108, fresh app process), conversation resumed from its rail
  row ("Describe this ima… saved chat - 1m"): the image chip renders after
  rehydration — but with a degraded label, see Defect 1.
- resume-message-selected.png — the rehydrated message selected in the
  resumed session: action row with "Save Image" available after relaunch.
- save-image-clicked.png — "Save Image" clicked in the resumed session:
  tooltip "Save image to disk." and toast "Image saved to
  /private/tmp/tldw-qa-attach-20260712/Downloads/console_image_20260712_232457.png".
  Filesystem check: the file exists (184 B) and `cmp` proves it is
  byte-identical to the original red-square.png. (~/Downloads under the
  isolated HOME; no `[chat.images].save_location` set.)

## Defects found

1. MINOR — rehydrated chip label loses filename/size. After relaunch+resume
   the chip renders `🖼 image/png` instead of `🖼 red-square.png · 184 B`.
   Root cause: `_console_messages_from_conversation_tree`
   (tldw_chatbook/UI/Screens/chat_screen.py:2138-2167) rebuilds resumed
   messages with `image_data`/`image_mime_type` from the DB row but never
   sets `attachment_label`, so `_message_image_chip`
   (tldw_chatbook/Widgets/Console/console_transcript.py:88-93) falls back to
   the mime type. In-process screen-state save/restore DOES carry
   `attachment_label` (chat_screen.py:5633-5676), so the loss only shows
   after a DB rehydration. Save Image still works after resume (it re-fetches
   image bytes by persisted message id). Repro: attach image → send → quit →
   relaunch → resume conversation → chip label reads "image/png".
   Not a walk-stopper; flagged for the approval gate.

2. OBSERVATION (possibly pre-existing design) — the `[failed]` assistant row
   and the "Provider stream failed…" System row do NOT rehydrate on resume;
   only the user image message comes back. Whether failed assistant rows
   should persist is a product question outside this feature's AC, noted for
   completeness.

3. OBSERVATION (pre-existing shared widget, not this branch) — the
   EnhancedFileOpen dialog under the Console has layout quirks: a large empty
   top region, a vestigial tiny file-name input, and the InputBar's
   Open/Cancel buttons pushed out of view by the near-full-width filter
   Select. Selection works (click row fills the input, Enter confirms;
   Escape cancels). Same dialog is used by legacy chat attach.

4. OBSERVATION — a conversation whose draft begins with an inlined file
   segment gets auto-titled from the file framing ("--- Contents of z…")
   rather than the user's typed prompt, and that title is what the rail
   shows. Cosmetic; existing 30-char auto-title rule applied to the new
   canonical draft layout.

## Deviations / method notes

- Send/✕ button coordinates shift ~5 cells left when the attachment
  indicator appears (actions row grows 37→42 cells). One driver run clicked
  the attach button's edge instead of ✕; that run was discarded and re-shot
  (its only side effect is documented under capture 4's "errant run" note).
- Textual tooltips did not render on a single teleport mouse-move; a
  two-step hover (adjacent button → Send) shows them reliably.
- The Qwen3.6 gguf is a reasoning model (~7 tok/s on this host, hidden
  `reasoning_content` first): visible answer tokens only arrive ~60-140 s
  after Send. Direct curl of the same payload: 64 s total, 435 completion
  tokens of which 1641 chars were hidden reasoning. Capture shots were taken
  at t≈48/93/138 s; the t≈138 s frame is published. `/no_think` is ignored
  by this server's chat template.
- The transcript shows no "Generating…"/partial indicator between Send and
  the first visible content token (Assistant row stays empty while the model
  reasons); matches the known residual noted in the core-loop upgrades QA.
- The blocked-send-reason auto-clear on capability change was evidenced via
  the config override across a process restart, not a live in-session flip
  (the Console has no in-session vision toggle).
