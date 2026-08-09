# Library Import & Export — getting content in, packaging it out

## What this screen is for

These are the Library's two doorways. **Import media** turns files, folders,
and URLs into Library media — checked before they run, queued while they
run, and searchable afterwards. **Export bundle (.zip)** packages your local
media, conversations, and notes into a single portable `.zip` you can
archive or share. Come here when you want the app to know about a
document, a recording, or a web page — or when you want to carry your
content somewhere else.

## Getting there

Press **Ctrl+3** for Library (see [the Library overview](../library.md)),
then either:

- Click **"Import…"** — the primary button at the very top of the
  left rail — to land on Import media directly.
- In the rail's **Import / Export** section, click **"Import…"** or
  **"Export"**.

Scoped exports also arrive here on their own: pressing **"Export…"** or
**"Export selected"** in the Media, Notes, or Conversations panels opens
the same Export bundle (.zip) form, pre-limited to that content (see
[Media & Conversations](media-and-conversations.md)).

In server mode the **Export** rail row is disabled, with the tooltip
"Export packages local content only."

## Layout tour

![Import media](../images/library/import-media.svg)

**Import media**, top to bottom:

- **Header** — "Import media", then a target line ("Imports run on this
  machine." / "Imports run on the server."). When a server is available, a
  switch button ("Import on the server" ↔ "Import on this machine")
  follows the line.
- **Path field** — "Path to a local file or a URL…", with "Browse…" and
  (once something is typed) "Clear". On a fresh visit, two orientation
  lines sit below: "Import a file, a whole folder, or a URL. Supported:
  PDF documents, Word/Office documents, audio/video files, e-books, plain
  text files, web pages." and "Imported items are searchable in your
  Library and can be used as context in chat."
- **Pre-check summary** — as soon as you enter a path the form shows
  "Checking…", then replaces it with a type breakdown ("1 PDF document,
  2 audio/video files"; a URL reads "1 web page"), a size estimate
  ("3 files · 1.2 MB" — omitted for URLs, whose size isn't known ahead of
  time), any "⚠" warnings about missing tooling, and — if some files
  can't be handled — "2 unsupported files will be skipped: …". An
  unreachable URL reports a plain reason ("URL unreachable — the server
  name could not be found."), never a raw error dump.
- **Options** — "Expand all" / "Collapse all", then one fold per detected
  content type, titled with its current settings (for example "Plain text
  & HTML — Analyze after import: off, Chunk content: on, …"). Word/Office
  documents (.doc/.docx/.odt/.rtf) get their own fold; the Plain text &
  HTML fold's Analyze/Chunk/Encoding options still apply to them as the
  always-present base. Each fold ends with "Reset to defaults".
- **Metadata** — "Title (optional)", "Author (optional)", "Keywords,
  comma-separated (optional)". These apply to everything in the import.
- **Start** — a quiet gate line ("Enter a file path or URL to start.")
  and the "Start import" button.
- **Queue** — the "Queue" heading, a per-state count line while jobs
  exist ("1 parsing · 2 queued · 1 done"), one line per job with action
  buttons underneath, "Clear finished", and a collapsed "Recent imports"
  fold listing the last finished jobs. Empty state: "No import jobs yet."
  Pressing "Start import" scrolls the Queue heading into view, so the
  freshly queued rows are the first thing you see after a submit.
- **Fold indicator** — while the form is taller than the pane, a pinned
  "▼ more — scroll for the rest" row holds the bottom edge; it disappears
  once everything fits.

**Export bundle (.zip)** is a single form: the "Export bundle (.zip)"
header, a scope line ("Everything: 128 media · 542 conversations · 87
notes", or "Notes · 87 items" when you arrived scoped; "Counting…" while
it tallies), the "Export name" and "Description (optional)" fields, a
"quality: … ▸" button with the helper "original copies full media files
into the zip" (shown only when media is in scope), "Choose destination…"
above "No destination chosen", and the "Export bundle (.zip)" submit
button. A "Cancel" button appears while an export is running. Once an
export finishes, a "Last export: <path> · <relative time>" line appears
above the submit button and stays there — it updates in place after each
further export and survives switching to another rail row and back, for
the rest of the session.

## Features & controls

| Import control | What it does |
|---|---|
| "Browse…" | Opens the "Import media" file picker (remembers your last folder). The listing shows Name / Size / Modified column headers, human-readable sizes ("512 B", "2.4 MB", never a bare byte count), no size on folder rows (including ".."), and a labeled "File name:" input at the bottom. Folders and URLs are typed or pasted into the path field instead. |
| Pre-check warnings ("⚠ …") | Name a missing optional package, what it's needed for, and the install command that fixes it. A compact "Copy install command" button sits right under the warnings — you no longer have to open the guardrail dialog to copy it (with several distinct commands, each button names its extra, e.g. "Copy install command (.[audio])"). |
| "Choose a file…" / "Retry" | Offered under pre-check errors — pick a different path, or re-run the check after a network hiccup. |
| Per-type options | Every dropdown shows a plain-language choice (the internal value still travels to the pipeline). PDF documents: "PDF engine" ("PyMuPDF (plain text)" / "PyMuPDF4LLM (Markdown)" / "Docling (layout-aware · OCR-capable)" / "Docext (vision-model OCR)"), "Enable OCR (docling or docext engines only)", "OCR language", "OCR backend" ("Auto (let Docext choose)" / Docext / Tesseract / EasyOCR / PaddleOCR / Docling — docext engine only). Word/Office documents: "Processing method" ("Auto (Docling when installed)" / Docling / "Native per-format parser"), "Enable OCR (docling method only)", "OCR language". Audio & video: "Transcription provider" ("Auto (faster-whisper)" / "Parakeet (ONNX)" / "Faster Whisper" / "transcribe.cpp (GGUF)"), "Local Parakeet model folder", "Transcription model" ("Tiny (fastest · least accurate)" through "Large (most accurate · slowest)"), "Language", "Translate to English (via faster-whisper)", "Include timestamps", "Speaker diarization", "Voice activity detection (VAD) filter". E-books: "Extraction method" ("Filtered (skips covers & front matter)" / "Markdown (keeps headings & structure)" / "Basic (every section · plain text)"), "Chunking method" ("By chapter" / "By sentence" / "By word count" / "By paragraph"), "Include table of contents". Plain text & HTML: "Analyze after import", "Chunk content", "Chunk size", "Chunk overlap", "Encoding". Web pages (URLs): "What to fetch" ("This page only" / "Site map" / "Pages under this URL" / "Follow links (recursive)"), "Maximum pages", "Maximum depth". |
| PDF / document OCR | The OCR checkbox is inert under engines that cannot OCR — the label names the capable ones. "OCR language" rides the OCR toggle; the PDF "OCR backend" applies to the docext engine only. |
| Inert (grayed) options | Any option whose precondition isn't met renders dimmed on a darker field AND says why at its label — "Local Parakeet model folder — needs the parakeet-onnx provider", "Maximum pages — single-page fetch selected", "Chunk size — needs Chunk content on", "OCR language — needs Enable OCR on"; a missing optional package reads "— needs <package> installed". Flip the named gate and the field wakes up. |
| "Translate to English" | Transcribes audio/video into English regardless of the spoken language. Runs via faster-whisper — the toggle is inert under parakeet-onnx and transcribe-cpp, which cannot translate. |
| E-book "Chunking method" | "chapters" (the default) stores one retrieval chunk per chapter; sentences / words / paragraphs chunk by that unit using the Chunk size/overlap values. |
| Web "What to fetch" on a local import | The multi-page methods (sitemap / url_level / recursive_scraping) run only on the server. Selecting one while importing on this machine shows "Multi-page fetch runs on the server — this local import fetches one page." right under the control. |
| "Analyze after import" | Runs an LLM summary of each imported item, stored alongside it (visible from the media viewer's analysis panel). The whole `[analysis_defaults]` section travels — provider, model, temperature, top_p, min_p, max_tokens, system_prompt — so the stored analysis matches what the Media analysis panel would produce under the same config; the key comes from `[api_settings.<provider>]` or the provider's usual environment variable. When the option is on but no provider is callable — including a configured provider the analysis pipeline cannot dispatch ("provider 'X' is not supported for ingest analysis") — a line above Start says so ("Analyze after import is on, but … Imports will run without analysis.") and finished rows read "Imported name — analysis skipped: <reason>". If the analysis call itself fails (provider error), the import still succeeds and the row reads "Imported name — analysis failed: <reason>" instead of silently storing nothing (or worse, the error text). |
| "Chunk content" | Governs every type: off means no retrieval chunks are stored at all; on chunks plain text / documents / HTML too (not just PDF/e-book/audio), using "Chunk size" and "Chunk overlap" — both measured in words. |
| "Encoding" | How plain text and HTML files are decoded: "Auto-detect (UTF-8 first)" (strict UTF-8, then detection) or an explicit UTF-8 / UTF-16 / "Latin-1 (ISO-8859-1)" / "Windows-1252 (Western)". A wrong explicit choice shows up as replacement characters rather than failing the import. |
| "Install verified Parakeet v2 INT8 (630.6 MiB)…" | In the Audio & video fold, enabled when the provider is parakeet-onnx (under any other provider the button is inert and its label ends "— needs the parakeet-onnx provider"). Opens a consent dialog listing Source, Revision, License, Download size, and Destination, ending "All four files are checked against pinned sizes and SHA-256 digests before the bundle becomes usable." Buttons: "Cancel" / "Install". |
| "Start import" | Queues everything the pre-check found. If warnings are outstanding, the "Some files may fail to import:" dialog appears first (see below). |
| Queue rows | "● queued / parsing / writing · name" while working, "✓ done · name · 4s" on success, "✗ failed · name · reason" (plus " · retry 1" after a retry) on failure, "⊘ cancelled · name" when stopped on purpose. Server jobs carry an " · on server" suffix. |
| Row actions | "Open in Library" (done, local) jumps to the new media item; "View on server" (done, server); "Show details" shows the full error; "Retry" re-queues a failed job; "Cancel" stops an in-flight server job; "Dismiss" removes a failed row. |
| "Clear finished" | Removes all done and failed rows at once. |

The guardrail dialog — "Some files may fail to import:" — lists each
problem as "- <package> (N files): <what needs it>" ("1 file" when only
one is affected) with a "Copy install command" button per line, then
"Cancel" and "Start import anyway". Starting anyway is safe: affected
files simply fail individually and show up as ✗ rows you can retry after
installing.

| Export control | What it does |
|---|---|
| "Export name" | Pre-filled "Library export 2026-07-31" (today's date); becomes the bundle's display name. |
| "quality: thumbnail ▸" | Cycles thumbnail → compressed → original. Only "original" copies full media files into the zip; the others keep the package small. |
| "Choose destination…" | Opens "Choose Export Destination". Whatever you pick is normalized to end in `.zip`; if that file already exists, an "Overwrites <name>" note appears (informational — exporting proceeds and replaces it). |
| "Export bundle (.zip)" | Enabled once counting has finished, the scope is non-empty, and a destination is chosen. "Nothing to export in this scope." appears when the scope is empty; either way, hovering the button always shows a tooltip naming the same reason it's disabled (or "Write the bundle to the chosen destination." once it's ready) — a disabled press can never look like it silently did nothing. |
| "Cancel" | Visible only while an export is running; stops it. |
| "Last export: …" | Appears after the first successful export this session; names the exact path written and how long ago, and stays until the next successful export replaces it. |

## Common tasks

1. **Import one file** — Click "Import…", press "Browse…", pick the
   file, wait for the type breakdown, then press "Start import". When the
   row reads "✓ done", press "Open in Library" to view it.
2. **Import a whole folder** — Type or paste the folder's path into the
   path field (the "Browse…" picker selects single files only). Review the
   breakdown and size estimate — folder scans stop at 1,000 files and note
   " · more files not shown" — then press "Start import".
3. **Import from a URL** — Paste the address into the path field. A link to
   a video site imports as audio/video; a PDF link as a PDF; other pages
   under "Web pages", where "What to fetch" / "Maximum pages" / "Maximum
   depth" control how much gets scraped. Press "Start import".
4. **Fix a "may fail to import" warning** — Press "Copy install command"
   right under the "⚠" warning in the pre-check summary (the same button
   also appears in the "Some files may fail to import:" dialog after
   Start). Quit the app, run the copied command in the environment the app
   is installed in, relaunch, and start the import again — the warning is
   gone.
5. **Export your notes as a bundle** — In the rail click Browse ▸ Notes,
   press "Export…" above the list. On the "Export bundle (.zip)" form
   confirm the scope line says "Notes · N items", adjust the name, press
   "Choose destination…", pick where the `.zip` goes, then press "Export
   bundle (.zip)".
6. **Retry a failed job** — Find the "✗ failed" row in the Queue and press
   "Retry"; the new attempt shows a " · retry 1" suffix. No Retry button
   means the failure is permanent (unsupported type or missing file) — fix
   the source and start a fresh import, and use "Dismiss" to drop the row.

## Keyboard & commands

**Import media** is keyboard-first: **i** opens it from anywhere on the
Library screen (not just the landing — though never while you're typing in
a text field, where `i` stays a letter), and entering the form always
parks the caret in the path field, so you can type or paste a path
immediately. **Enter** in the path field starts the import once the gate
line clears; **Escape** returns you to the Library landing (a half-filled
form is kept, same as switching rail rows). The footer and F1 list the
same set while the form is open: `enter start import`, `esc back to hub`.

The Export form has no screen-specific shortcuts. **Escape** also closes
the "Some files may fail to import:" and Parakeet install dialogs. Global
keys live in the [guide index](../index.md).

## Related settings & docs

`config.toml` keys, all under `[library]`:

| Key | What it remembers |
|---|---|
| `ingest.backend` | Whether imports target this machine or the server. |
| `ingest.last_directory` | The folder "Browse…" opens in next time. |
| `ingest_options.<group>.<field>` | Every per-type option, saved when you start an import (e.g. `ingest_options.generic.chunk_size`). |
| `ingest_directory_scan_limit` | Folder scan cap (default 1000). |

"Chunk size" is kept between 100 and 5000 words — values outside that
range are pulled back to the nearest bound when the import starts. An
untouched form submits the defaults the panel displays (size 1000,
overlap 100), identically on the local and server paths.

Deep dives: [TRANSCRIPTION.md](../../Features/TRANSCRIPTION.md) covers the
audio/video transcription providers and their optional extras. See also
[the Library overview](../library.md) and
[Media & Conversations](media-and-conversations.md) for what happens to
imported items afterwards.

## Quirks & troubleshooting

- **Unsupported files become ✗ rows on purpose.** Anything the pre-check
  can't classify is still queued and "recorded as a failure", so a big
  folder import tells you exactly what was skipped instead of silently
  dropping files. These rows offer "Dismiss" but not "Retry".
- **The Queue remembers past sessions.** Finished and failed jobs are kept
  between launches. A job still running when you quit comes back as
  "✗ failed · … · Interrupted by app restart" and stays retryable; work
  that was mid-parse is not resumed automatically.
- **"Analyze after import" is off, "Chunk content" is on — by design.**
  Analysis costs an LLM call per document, which a folder import shouldn't
  trigger unasked. Chunking is local and cheap, and without it imported
  documents never show up properly in search and RAG — leave it on.
- **Transcription may need optional extras.** The default audio/video
  provider needs its packages installed; the pre-check warns you and the
  guardrail dialog hands you the install command. The curated Parakeet
  model is a separate one-press download from the options fold.
- **Export is local-only.** In server mode the rail's Export row is
  disabled ("Export packages local content only.") — switch the Library
  back to local to package content.
- **"Show details" is your first stop on a confusing failure** — it opens
  the full error behind the shortened reason on the row.

—
*Verified against dev @ 4acb17a0b — 2026-08-07 (TASK-2857: the rail
button/canvas title/Start button/completion toast all read "Import…" /
"Import media" / "Start import" / "Import finished" — was "Add content…"
/ "Import media" / "Start ingest" / "Ingest finished"; the Export form
reads "Export bundle (.zip)" everywhere it used to say "Export chatbook";
option labels and dialogs say "import" instead of "ingest" throughout)*

*Verified against dev @ 6b38a13b8 — 2026-08-07 (task-2858 AC#3:
the Export button's disabled state now always carries an explaining
tooltip, and a successful export leaves the durable "Last export: …"
line described above — previously an empty-scope/no-destination press
had no tooltip and a successful export left the canvas unchanged)*

*Verified against dev @ 023a04a48 — 2026-08-07 (task-3300: the "Some
files may fail to import:" dialog now renders every warning line and its
"Copy install command" button in a compact themed dialog — previously the
warning rows collapsed to an empty full-height column; counts read
"1 file" / "N files"; "Start import anyway" fits on one line; Cancel is
no longer styled red)*

*Verified against dev @ 023a04a48 — 2026-08-07 (task-3302: the Keyboard
& commands section above is new behavior — entering Import media now
focuses the path field, Esc returns to the landing, `i` works from any
Library canvas, the footer/F1 advertise `enter start import` / `esc back
to hub`, and focused fields/compact buttons show a heavy structural focus
edge instead of a color-only change)*

*Verified against feat/media-ingest-ux-parity @ 7c451678e — 2026-08-07
(task-3301: "Analyze after import", "Chunk content" OFF, and "Encoding"
are wired for real on the local path — analysis resolves the
`[analysis_defaults]` provider (readiness hint above Start, "analysis
skipped: …" on done rows when unconfigured), Chunk OFF stores no chunks
for any type while Chunk ON now chunks plain text/HTML/documents too,
the Encoding select governs text decoding, the chunk size/overlap unit
hints read "words", and an untouched form submits overlap 100 on both
the local and server paths — previously all three options were silent
no-ops locally and the local overlap fallback was 50)*

*Verified against feat/media-ingest-ux-parity @ e28d31d76 — 2026-08-07
(task-3303: Word/Office documents (.doc/.docx/.odt/.rtf) get their own
options fold — "Processing method", "Enable OCR", "OCR language" — and an
honest pre-check noun ("1 Word/Office document", previously "1 plain text
file"); the PDF fold adds the docext engine plus "OCR language"/"OCR
backend", and Enable OCR is inert with its reason under engines that
cannot OCR (previously a silent no-op); e-books gain a "Chunking method"
select whose "chapters" default reaches the processor; Audio & video gain
"Translate to English (via faster-whisper)" and a VAD filter toggle; a
local import with a multi-page "What to fetch" selection now says
"Multi-page fetch runs on the server — this local import fetches one
page." under the control — previously it silently imported one page)*

*Verified against feat/media-ingest-ux-parity @ 4839d7ce2 — 2026-08-07
(task-3304: schema-disabled options now read as inert AND say why at the
control ("— needs the parakeet-onnx provider", "— single-page fetch
selected", "— needs Chunk content on"), with app-tier Legible Disabled
styling instead of the invisible 0.7 fade; Start scrolls the Queue heading
into view and a pinned "▼ more — scroll for the rest" row marks overflow;
the Browse picker gained Name/Size/Modified headers, humanized sizes, no
size on folder rows, and a "File name:" label; "Copy install command" now
also sits under the pre-check "⚠" warnings, not only in the guardrail
dialog)*

*Verified against feat/media-ingest-ux-parity @ 0ba5bf44c — 2026-08-08
(task-3305: every option dropdown shows plain-language choices ("PyMuPDF4LLM
(Markdown)", "Auto (faster-whisper)", "This page only") while the internal
value still persists; the supported-formats list and the Start gate name
URLs ("Enter a file path or URL to start."); URL pre-check failures read as
plain reasons ("URL unreachable — the server name could not be found."),
never a raw exception; a URL pre-check reads "1 web page" with no bogus
"1 file · 0 B" estimate; bracketed filenames render clean in Recent
imports; a finished queue's tally drops the "— in queue" suffix; collapsed
option-fold titles cap at the three most salient settings (changed values
first) and never render a dangling empty value; the empty Parakeet folder
field shows an example path instead of repeating its label; a failed row
no longer repeats its own filename inside the reason; and the "N will
import" commit line hides while the fix-your-options gate is blocking
Start)*

*Verified against feat/media-ingest-ux-parity — 2026-08-08 (task-3301
xhigh review round: "Analyze after import" now actually reaches a
provider — the ingest analysis dispatches through the same `chat_api_call`
path the Media analysis panel uses, carrying the full `[analysis_defaults]`
call shape (model/temperature/top_p/min_p/max_tokens/system_prompt);
provider spellings like "MistralAI"/"KoboldCpp" resolve to dispatchable
names or the pre-Start hint says "provider 'X' is not supported for ingest
analysis"; an in-band "Error: …" result is never stored as an analysis —
done rows read "Imported name — analysis failed: <reason>" instead)*

*Verified against feat/media-ingest-ux-parity — 2026-08-08 (xhigh review
round 2: e-book chunking now actually executes — the "Chunking method"
select (chapters/sentences/words/paragraphs) governs real chunk output
instead of silently degrading to one full-text chunk; the chunk size/
overlap "words" unit is now true for PDF and audio/video too (the
pipeline sends an explicit word method where the processors used to
default to sentences); the "Some files may fail to import:" dialog
scrolls its warning list so Cancel / "Start import anyway" stay reachable
at any warning count; a stale "Translate to English" tick left over from
a translating provider no longer fails transcribe-cpp/parakeet batches
at dispatch; and an unknown explicit Encoding value degrades to
replacement characters plus a visible warning instead of failing the
import)*
