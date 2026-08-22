# Library Import & Export — getting content in, packaging it out

## What this screen is for

These are the Library's two doorways. **Import media** turns files, folders,
and URLs into Library media — checked before they run, queued while they
run, and searchable afterwards. **Export bundle (.zip)** packages your local
media, conversations, notes, and Prompts/Recipes into a single portable `.zip`
you can archive or share. Come here when you want the app to know about a
document, a recording, or a web page — or when you want to carry your
content somewhere else.

## Getting there

Press **Ctrl+3** for Library (see [the Library overview](../library.md)),
then either:

- Click **"Import…"** — the primary button at the very top of the
  left rail — to land on Import media directly.
- In the rail's **Import / Export** section, click **"Import…"** or
  **"Export"**.

Scoped exports also arrive here on their own: use **"Export…"** in the Media,
Notes, Conversations, or Prompts panels (or **"Export selected"** where a list
offers it). Each opens the same Export bundle (.zip) form, pre-limited to that
content (see [Media & Conversations](media-and-conversations.md)).

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
  PDF documents, Word/Office documents, audio/video files, e-books,
  images, plain text files, web pages." and "Imported items are
  searchable in your Library and can be used as context in chat."
- **Pre-check summary** — as soon as you enter a path the form shows
  "Checking…", then replaces it with a type breakdown ("1 PDF document,
  2 audio/video files"; a URL reads "1 web page"), a size estimate
  ("3 files · 1.2 MB" — omitted for URLs, whose size isn't known ahead of
  time), any "⚠" warnings about missing tooling, and — if some files
  can't be handled — "2 unsupported files will be skipped: …" (importing
  on the server, the same line reads "will fail", because that backend
  records a failure row rather than skipping quietly). The pre-check does
  **not** contact a URL: it reads the address, not the site, so nothing is
  fetched while you are still typing one. A URL that turns out not to be
  fetchable is reported by the import job itself, where the failure carries
  a real reason. (You can turn a link check back on with `[library]
  ingest_url_preflight_probe = true` in `config.toml`; it then runs only
  when you leave the field, press Enter, pick with Browse…, or press
  "Retry check" — never while you type — and an address it cannot check
  reports one plain "The link could not be checked ahead of time. The
  import will still be attempted." A link that answers but says the page
  does not exist still reports "URL unreachable — the server says this page
  does not exist (HTTP 404).") When the import is
  aimed at the server, the missing-tooling block is replaced by a single
  quiet note ("1 local component isn't installed — that affects imports
  on this machine only; this one runs on the server."): those extras
  belong to a machine that is not doing the work, so installing them
  would not change the run.
- **Options** — "Expand all" / "Collapse all", then one fold per detected
  content type. An untouched fold uses only its short type name; after you
  change a setting, its title becomes a concise receipt of those changes.
  Word/Office
  documents (.doc/.docx/.odt/.rtf) get their own fold; the Plain text &
  HTML fold's Analyze/Chunk/Encoding options still apply to them as the
  always-present base. Each fold ends with "Reset to defaults".
- **Metadata** — three persistently labeled fields: "Title (optional)",
  "Author (optional)", and "Keywords (optional)". Example/default guidance
  remains in the placeholders while the labels survive entered values.
  These apply to everything in the import.
- **Start** — a forecast line, a quiet gate line ("Enter a file path or
  URL to start.") and the "Start import" button, kept together in a pinned
  review bar so a long pre-check cannot push the decision below the fold.
  After submission the blank review bar hides so Queue activity gets the
  viewport. The forecast is one
  sentence of counts for the staged selection — "2 will import · 1 will
  skip · 4 will fail (3 need tooling, 1 empty)" — and it counts a file
  whose type needs tooling you don't have as a failure, not an import,
  because the pre-check has already warned that the tooling is missing.
  The failure segment names its reasons whenever tooling is one of them,
  so you can tell which part of the number an install would change. The
  line stays on screen while a gate blocks Start (a bad option value, a
  selection with nothing importable), so the numbers you were reasoning
  about don't vanish at the moment you need them — it goes quiet only
  when this runtime has no import path at all (no job queue, no media
  database), where a count would promise something nothing can deliver.
  Two hedges are carried rather than rounded off: when the duplicate
  check hit its candidate cap the match count is a floor and the import
  count is therefore a ceiling, so the line reads "at most 5 will
  import · at least 20 will match"; and when the import is aimed at a
  **server**, the local tooling inventory says nothing about it, so the
  line reads "5 will be sent to the server · server tooling isn't
  checked from here" rather than pretending to know what the server has
  installed. What the forecast *can* know about a server import is which
  files that backend will not take at all — images, and anything with no
  recognised format — and those are counted as failures with the reason
  named: "3 will be sent to the server · 2 will fail (unsupported by the
  server)". "Unsupported by the server" is not "unreadable": an image
  imports fine on this machine, it simply has no place in the server's
  import API, so switching the target back to this machine changes the
  number. An empty (0 B) file is counted as a failure on both targets,
  and on both targets that is this app's own doing: a server import
  refuses a 0-byte file here rather than uploading it ("empty.txt is
  empty; there was nothing to send."), so the count never depends on
  what a server would have made of an empty upload.
- **Queue** — the "Queue" heading, a per-state count line while jobs
  exist ("This queue: 1 parsing · 2 queued · 1 done" — task-2859: the
  "This queue:" prefix replaced a trailing "— in queue" suffix that
  self-contradicted whenever every listed job was already done or
  failed), one line per job with action buttons underneath,
  "Clear finished", and a collapsed "Recent imports" fold listing the
  last finished jobs. Batch headers use `active` plus exact state counts
  (`1 queued · 1 done · 1 failed`) rather than adding a contradictory
  `running` synonym. Empty state: "No import jobs yet."
  Pressing "Start import" scrolls the Queue heading into view, so the
  freshly queued rows are the first thing you see after a submit.
- **Fold indicator** — while the form is taller than the pane, a pinned
  "▼ more — scroll for the rest" row holds the bottom edge; it disappears
  once everything fits.

**Export bundle (.zip)** is a single form: the "Export bundle (.zip)"
header, a scope line ("Everything: 128 media · 542 conversations · 87 notes · 34
prompts", or "Prompts · 34 items" when you arrived scoped; "Counting…" while
it tallies), the "Export name" and "Description (optional)" fields, a
"quality: …" chooser with a helper line matching whichever option is
actually selected (task-2859: "keeps a small preview image instead of the
full file" / "shrinks media files before adding them to the zip" /
"copies full media files into the zip" for thumbnail/compressed/original
respectively — a single fixed "original copies full media files…" caption
used to show regardless of the selected option) (shown only when media is
in scope), "Choose destination…"
above "No destination chosen", and the "Export bundle (.zip)" submit
button. A "Cancel" button appears while an export is running. Once an
export finishes, a "Last export: <path> · <relative time>" line appears
above the submit button and stays there — it updates in place after each
further export and survives switching to another rail row and back, for
the rest of the session.

## Features & controls

### Conversation and `.chatbook` portability

Current `.chatbook` packages preserve a conversation's complete included
message graph: stable export-local message and parent identities, branches,
assistant variants and their order, the selected leaf, deleted eligibility,
and each assistant owner's validated provider-continuation checkpoint. Import
validates and remaps the complete graph before attaching any checkpoint, so an
ID collision cannot move private state to a sibling variant. Opening or
importing a package never starts or resumes tools.

Ordinary conversation JSON is intentionally narrower. It exports only the
active path with an explicit public message projection and, when present, an
`_private.provider_continuation` field attached to that assistant item. It does
not serialize whole database rows, off-path variants, transient UI fields, or
metadata bags. Older flat `.chatbook` packages remain readable, but when they
cannot prove the private field's exact owner, visible messages import and exact
tool continuation is discarded with a safe warning.

Both formats can therefore contain private model state, tool arguments, and
provider-bound tool results. Protect them as you would the full conversation;
Chatbook adds no separate file encryption. Text, Markdown, rendered transcript,
clipboard-message, search/FTS, summary, title, log, error, and usage exports do
not include this private field. Invalid, unknown-version, contradictory, or
oversized private data is dropped while usable visible messages still import,
and the warning never quotes the rejected data.

| Import control | What it does |
|---|---|
| "Browse…" | Opens the "Import media" file picker (remembers your last folder). The listing shows Name / Size / Modified column headers, human-readable sizes ("512 B", "2.4 MB", never a bare byte count), no size on folder rows (including ".."), and a labeled "File name:" input at the bottom. Folders and URLs are typed or pasted into the path field instead. |
| Pre-check warnings ("⚠ …") | Name a missing optional package, what it's needed for, and the install command that fixes it. A compact "Copy install command" button sits right under the warnings (with several distinct commands, each button names its extra, e.g. "Copy install command (audio)", "Copy install command (video)"). |
| "Choose a file…" / "Retry" | Offered under pre-check errors — pick a different path, or re-run the check after a network hiccup. |
| Per-type options | Every dropdown shows a plain-language choice (the internal value still travels to the pipeline). PDF documents: "PDF engine" ("PyMuPDF (plain text)" / "PyMuPDF4LLM (Markdown)" / "Docling (layout-aware · OCR-capable)" / "Docext (vision-model OCR)"), "Enable OCR (docling or docext engines only)", "OCR language", "OCR backend" ("Auto (let Docext choose)" / Docext / Tesseract / EasyOCR / PaddleOCR / Docling — docext engine only). Word/Office documents: "Processing method" ("Auto (Docling when installed)" / Docling / "Native per-format parser"), "Enable OCR (docling method only)", "OCR language". Audio & video: "Transcription provider" ("Auto (faster-whisper)" / "Parakeet (ONNX)" / "Faster Whisper" / "transcribe.cpp (GGUF)"), "Local Parakeet model folder", "Transcription model" (the full faster-whisper catalog — Tiny through Large v3 including the English-only ".en" variants, the distilled "Distil" family, and the community "Large v3 Turbo" / "CrisperWhisper" builds), "Language", "Translate to English (via faster-whisper)", "Include timestamps", "Speaker diarization", "Voice activity detection (VAD) filter", "Start at" / "Stop at" (trim bounds, HH:MM:SS or seconds — blank means unbounded; "Stop at" is an absolute position in the recording, not a length measured from "Start at", and means the same thing for audio and video files), "Cookies file for gated URLs" (a Netscape cookies.txt path for yt-dlp; video URLs only — the file must exist when the job runs, otherwise the import proceeds without cookies and the queue row says "cookies ignored: …"), "Recursive summary (map-reduce)" (with Analyze after import + chunking: summarizes each chunk, then combines the summaries). E-books: "Extraction method" ("Filtered (skips covers & front matter)" / "Markdown (keeps headings & structure)" / "Basic (every section · plain text)"), "Chunking method" ("By chapter" / "By sentence" / "By word count" / "By paragraph"), "Include table of contents". Images (.png/.jpg/.jpeg/.gif/.webp/.bmp/.tiff/.tif): "Extract text (OCR)" (on by default — the extracted text is what gets imported), "OCR language", "OCR backend" ("Auto (best installed backend)" / "Docext (vision model)" / Docling / Tesseract / EasyOCR / PaddleOCR). Plain text & HTML: "Analyze after import", "Chunk content", "Chunk size", "Chunk overlap", "Encoding". Web pages (URLs): "What to fetch" ("This page only" / "Site map" / "Pages under this URL" / "Follow links (recursive)"), "Maximum pages", "Maximum depth". |
| PDF / document OCR | The OCR checkbox is inert under engines that cannot OCR — the label names the capable ones. "OCR language" rides the OCR toggle; the PDF "OCR backend" applies to the docext engine only. |
| Inert (grayed) options | Any option whose precondition isn't met renders dimmed on a darker field AND says why at its label — "Local Parakeet model folder — needs the parakeet-onnx provider", "Maximum pages — single-page fetch selected", "Chunk size — needs Chunk content on", "OCR language — needs Enable OCR on"; a missing optional package reads "— needs <package> installed". Flip the named gate and the field wakes up. |
| "Translate to English" | Transcribes audio/video into English regardless of the spoken language. Runs via faster-whisper — the toggle is inert under parakeet-onnx and transcribe-cpp, which cannot translate. |
| Image imports | An image's imported content is the text OCR finds in it, so OCR needs a backend installed (any one of docling, tesseract, easyocr, paddleocr, or docext — the pre-check "⚠" warning carries an install command when none is present). An image with OCR off, or in which OCR finds no text, fails its row honestly ("No text was found in …") instead of storing an empty, unsearchable entry. Images import locally only — a server-mode submission refuses them, since the server's ingest API has no image type. The Images fold applies to image FILES: a link to an image (`https://…/chart.png`) is pre-checked and imported as a web page, because the URL pipeline fetches and clips pages and has no image-download step. |
| E-book "Chunking method" | "chapters" (the default) stores one retrieval chunk per chapter; sentences / words / paragraphs chunk by that unit using the Chunk size/overlap values. Chapter chunking now works for PDFs and other documents too, not just e-book files — the same engine method underlies both, so a PDF imported with the chapter scheme is split by its headings/chapters rather than rejected or silently re-split. |
| Web "What to fetch" on a local import | The multi-page methods (sitemap / url_level / recursive_scraping) run only on the server. Selecting one while importing on this machine shows "Multi-page fetch runs on the server — this local import fetches one page." right under the control. |
| "Analyze after import" | Runs an LLM summary of each imported item, stored alongside it (visible from the media viewer's analysis panel). The whole `[analysis_defaults]` section travels — provider, model, temperature, top_p, min_p, max_tokens, system_prompt — so the stored analysis matches what the Media analysis panel would produce under the same config; the key comes from `[api_settings.<provider>]` or the provider's usual environment variable. When the option is on but no provider is callable — including a configured provider the analysis pipeline cannot dispatch ("provider 'X' is not supported for ingest analysis") — a line above Start says so ("Analyze after import is on, but … Imports will run without analysis.") and finished rows read "Imported name — analysis skipped: <reason>". If the analysis call itself fails (provider error), the import still succeeds and the row reads "Imported name — analysis failed: <reason>" instead of silently storing nothing (or worse, the error text). |
| "Chunk content" | Governs every type: off means no retrieval chunks are stored at all; on chunks plain text / documents / HTML too (not just PDF/e-book/audio), using "Chunk size" and "Chunk overlap" — both measured in words. (The import forms expose the handful of methods that make sense per type; the chunking engine underneath implements the full roster — words, sentences, paragraphs, tokens, semantic, json, xml, ebook_chapters, rolling_summarize, fixed_size, code, code_ast, structure_aware — for anything that calls it directly, including `ebook_chapters` for PDFs and documents.) |
| "Chunking template" | In the Import behavior fold, above the queue: pick one of your chunking templates (the live rows from RAG Admin) instead of setting method/size/overlap by hand. Defaults to "None (manual settings)" — today's behavior exactly. A picked template's chunking scheme beats the form's untouched size/overlap defaults; only a value you explicitly changed in the form overrides the template. The choice is remembered per imported item (it drives a later re-chunk of that item), and the import form's list loads from the database when the canvas is shown — newly created or renamed templates appear the next time you enter the Import view. The control is hidden in server mode (a server import never carries a template), and it is inert with "Chunk content" off ("— needs Chunk content on"). A template that no longer resolves (deleted or renamed after the choice was made) fails that import row with a named error instead of silently chunking a different way. |
| "Chunking template" — "Auto" | The picker's other choice, listed right after "None (manual settings)": let the app pick the chunking scheme per item. Auto decides in three steps and always lands somewhere: (1) if one of your templates carries a **classifier block** that matches the item (its `media_types`, plus optional `filename_regex` / `title_regex` / `url_regex`, cleared its `min_score`), the highest-scoring match wins and runs in full — preprocessing, chunking and postprocessing, indistinguishable from picking it by hand; (2) otherwise a media-type-aware plan from the same auto planner the server uses derives the method/size/overlap; (3) if that declines too, today's plain defaults apply — Auto never fails an import, it can only explain why it declined. Templates opt in by carrying a classifier block: a template without one is never auto-picked (the built-ins ship without blocks, so nothing changes until you author one), and blocks are added through the template authoring path in RAG Admin / the service layer. The name "auto" is reserved — you cannot create or rename a template to it — so the choice can never be shadowed by a row. The decision is recorded per item and **re-decided, not replayed**: a later re-chunk re-runs the selection against the current template store (add, delete or re-score classifier blocks and the re-chunk follows the new outcome), and the item's stored record is re-stamped to say what the re-chunk actually used. Auto is exclusively this picker choice — the `[chunking] default_template` config key never triggers it (a configured default is an ordinary template name). |
| "Encoding" | How plain text and HTML files are decoded: "Auto-detect (UTF-8 first)" (strict UTF-8, then detection) or an explicit UTF-8 / UTF-16 / "Latin-1 (ISO-8859-1)" / "Windows-1252 (Western)". A wrong explicit choice shows up as replacement characters rather than failing the import. |
| "Install verified Parakeet v2 INT8 (630.6 MiB)…" | In the Audio & video fold, enabled when the provider is parakeet-onnx (under any other provider the button is inert and its label ends "— needs the parakeet-onnx provider"). Opens a consent dialog listing Source, Revision, License, Download size, and Destination, ending "All four files are checked against pinned sizes and SHA-256 digests before the bundle becomes usable." Buttons: "Cancel" / "Install". |
| "Start import" | Queues everything the pre-check found. If "⚠" tooling warnings are outstanding, the first press doesn't submit — the line beside Start turns into "⚠ Press Start again to import anyway — N files will fail without more tooling." (or "… N files may fail." when the missing package is only an optional enhancement) and a second press (or a second Enter in the path field) starts the import. See "Consent for risky imports" below. Start is unavailable, with the reason stated at the button, when the selection has nothing importable: "This folder is empty — there's nothing to import. Choose a folder with files, or a single file." for a folder that really is empty, "Nothing in this folder could be scanned — 2 entries were skipped: folder imports pass over hidden files, links, and folders they can't read. Import a file directly, or choose another folder." for a folder whose entries the scan passed over, and "Nothing in this selection can be imported — N unsupported files." when nothing in it has a handler. Importing on the server adds one more: a selection this machine reads perfectly well but that backend will not take at all (a folder of nothing but images) gates Start with "Nothing in this selection can be sent to the server — 3 files unsupported by the server. Switch to importing on this machine, or choose video, audio, document, PDF or e-book files." — a different sentence from the one above, because the files are fine and the destination is the problem. None of these leaves a failed row behind: the import never starts. |
| Queue rows | "● queued / parsing / writing · name" while working, "✓ done · name · 4s" on success, "✗ failed · name · reason" (plus " · retry 1" after a retry) on failure, "⊘ cancelled · name" when stopped on purpose. Server jobs carry an " · on server" suffix. |
| Row actions | "Open in Library" (done, local) jumps to the new media item; "View on server" (done, server); "Show details" shows the full error; "Retry" re-queues a failed job; "Cancel" stops an in-flight server job; "Dismiss" removes a failed row. |
| "Show details" | Opens inline under the row: a plain-language reason ("Reason: No text could be extracted." / "The file couldn't be read." / "The file is empty." / "The Library couldn't be written to."), the full message when it says more than the row line, the underlying tool output once (never repeated between the message and the chain), and — only when a retry could actually change the outcome — one line of advice derived from that same reason. A deterministic failure whose text actually named a remedy says so ("Retrying now will fail the same way — install the tooling named above first, then Retry."); when nothing on screen named one, the advice states the determinism without inventing a remedy ("Retrying now will fail the same way — this file's content, or the tooling for it, has to change first."); a named missing package is named ("Missing dependency: pymupdf. Install it, then Retry."); and a cause we can't classify says nothing rather than encouraging a retry that would repeat itself. |
| "Clear finished" | Removes all done and failed rows at once (two presses: the first arms and renames the button "Press again to clear N finished…"). |
| "Retry this batch" | Below the queue, once your last import of the session has settled (while a job is still queued/parsing/writing it is hidden, and `r` is inert too — re-staging mid-run invites a duplicate batch): one press puts that submission's source, options, title, author, and keywords back into the form and re-runs the pre-check from scratch — install the package a warning named, press it, and the fresh forecast reflects the fix. If the form currently holds work the re-stage would overwrite (a different path, a title you started typing, an option you flipped), it takes two presses: the first renames the button "Press again to replace form" and changes nothing. It stages, not submits: review the forecast and press "Start import" again. Keyboard: `r` (anywhere on the Import canvas outside a text field). |

**Consent for risky imports** — starting with "⚠" tooling warnings
outstanding takes two presses, right at the Start button (task-3314
retired the old "Some files may fail to import:" dialog). The first press
converts the line beside Start into "⚠ Press Start again to import anyway
— N files will fail without more tooling." naming how many staged files
cannot be handled at all without it ("1 file" when only one), or "…
N files may fail." when the missing package only degrades the result.
That count is the same number the forecast line above it reports — both
are renderings of one computation, so they cannot disagree. The second
press — button or Enter,
they behave identically — starts the import. You can arm with Enter in
the path field and confirm with the Start button, or the other way
round; moving focus between the two presses does not cancel anything.
Starting anyway is safe: affected files simply fail individually and show
up as ✗ rows you can retry after installing. Esc backs out of the pending
confirm (and stays on the form); editing the path, changing an option,
picking a different file with "Browse…", or a pre-check that comes back
different also cancels it. The "Copy install command" buttons stay under
the warnings the whole time.

If the same source is already queued, parsing, or being written for the chosen
Local or Server destination, the first Start press queues nothing and the line
beside Start says the import is active. Press Start again after the brief
double-press guard to deliberately queue one duplicate. Local and Server are
separate, finished jobs do not block another import, and a folder is admitted
as one batch: the first press queues none of its files, while the confirmed
second press queues the whole unchanged selection. When tooling and active-source
warnings occur together, one confirmation accepts both, so the complete action
still takes two presses rather than three. Editing the request, changing its
destination, or leaving the Import canvas cancels pending consent.

| Export control | What it does |
|---|---|
| "Export name" | Pre-filled "Library export 2026-07-31" (today's date); becomes the bundle's display name. |
| "quality: thumbnail" | Press to open a one-row strip of thumbnail / compressed / original (✓ on the active one) right under the button; pick one directly, or press the button again / Escape to close without changing. The helper line underneath always describes the option currently showing. Only "original" copies full media files into the zip; the others keep the package small. |
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
3. **Import from a URL** — Paste the address into the path field. Pasting
   it does not contact the site; nothing is fetched until you press "Start
   import". A link to a video site imports as audio/video; a PDF link as a
   PDF; other pages under "Web pages", where "What to fetch" / "Maximum
   pages" / "Maximum depth" control how much gets scraped. Press "Start
   import".
4. **Fix a "may fail to import" warning** — Press "Copy install command"
   right under the "⚠" warning in the pre-check summary. Quit the app, run
   the copied command in the environment the app is installed in,
   relaunch, then press "Retry this batch" (or `r`) below the queue — the
   same source and options come back staged, the pre-check re-runs against
   the fixed environment, and the warning is gone.
5. **Export your notes as a bundle** — In the rail click Browse ▸ Notes,
   press "Export…" above the list. On the "Export bundle (.zip)" form
   confirm the scope line says "Notes · N items", adjust the name, press
   "Choose destination…", pick where the `.zip` goes, then press "Export
   bundle (.zip)".
6. **Export all Prompts and Recipes** — In the rail click Browse ▸ Prompts,
   press **Export…** in the list toolbar, confirm the scope says `Prompts · N
   items`, choose a destination, then press **Export bundle (.zip)**. The count
   covers every active local Prompt/Recipe, not only the visible page or selected
   collection. Choose the rail's **Export** row for `Everything` when the bundle
   should also include media, conversations, and notes.
7. **Retry a failed job** — Find the "✗ failed" row in the Queue and press
   "Retry"; the new attempt shows a " · retry 1" suffix. No Retry button
   means the failure is permanent (unsupported type or missing file) — fix
   the source and start a fresh import, and use "Dismiss" to drop the row.
   A URL your web-security settings refuse fails with a plain receipt:
   "URL blocked — your web-security settings don't allow fetching this
   address. To allow it, add the host to allowed_hosts under web_security
   in config.toml."

## Keyboard & commands

**Import media** is keyboard-first: **i** opens it from anywhere on the
Library screen (not just the landing — though never while you're typing in
a text field, where `i` stays a letter), and entering the form always
parks the caret in the path field, so you can type or paste a path
immediately. **Enter** in the path field starts the import once the gate
line clears — with "⚠" warnings outstanding, Enter,Enter carries the same
two-press consent as the Start button. **r** re-stages your last import
of the session ("Retry this batch") when the queue has settled — inside a
text field it stays a letter. **Escape** first backs out of a pending
"Press Start again" confirm (staying on the form), otherwise returns you
to the Library landing (a half-filled form is kept, same as switching
rail rows). At narrow widths the navigation rail collapses to its reachable
**Nav** handle so the form keeps working width. The footer preserves primary
and recovery actions first, and F1 lists the same state-derived set:
`enter start`, `esc back`, and, when available, `r retry`.

The Export form has no screen-specific shortcuts. **Escape** also closes
the Parakeet install dialog. Global keys live in the
[guide index](../index.md).

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

One thing to know about what chunking does to your text: imported text is
**sanitized before it's chunked and stored** — null bytes and unusual
control characters are turned into spaces, Unicode is normalized to NFC
where that doesn't shift character positions, and bidirectional text
override characters (a known display-spoofing trick) are neutralized. The
chunked, searchable text can therefore differ slightly from the raw file
you fed in. Every stored chunk is also stamped with the chunking engine
version that produced it (visible to the RAG Admin diagnostics report), and
carries richer metadata — offsets into the source, word counts, and the
method used — which is what makes precise citations possible. `tiktoken`
and `defusedxml` now ship as core dependencies, so the `tokens` chunking
method counts real tokens (and says plainly to install tiktoken if it's
missing, instead of silently approximating by word count) and the `xml`
method parses safely by default.

If you use a **chunking template** (the "Chunking template" picker above),
two more things apply. First, a default can be set for every import that
didn't pick one: `[chunking] default_template = "<name>"` in config.toml
(empty by default — no default template). Second, a **chunk's offsets can
be relative to preprocessed text**: most templates run a preprocessing step
(normalizing whitespace, cleaning markdown) before chunking, so a chunk's
start/end positions count into that transformed text, not necessarily the
stored source. Each chunk says which basis it used in its metadata
(`offset_basis`: "source" when nothing rewrote the text, otherwise the
preprocessing operation named) — consumers that need source-relative spans
(navigation, citations) can check that one key instead of guessing. The
item's stored chunk rows also record which template chunked them, alongside
the engine-version stamp.

If you pick **Auto** in that picker, a template's **classifier block** is
how it volunteers for automatic selection: `media_types` matches the
import's type, each of `filename_regex` / `title_regex` / `url_regex` that
matches adds score, and the block's `min_score` gates the result (a block
with no `min_score` selects at any positive score). Highest score wins;
ties break by the block's `priority` and then by name. The stored decision
(`mode: "auto"`, the winning tier and rationale) rides the same
`Media.chunking_config` column a named pick does — template-tier Auto wins
record the winner's name there too, so RAG Admin's usage counts and
"documents using this template" treat them exactly like manual picks, and
stop counting the moment a re-chunk moves the item off that template.

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
  provider needs its packages installed; the pre-check warns you and its
  "Copy install command" button hands you the fix. The curated Parakeet
  model is a separate one-press download from the options fold.
- **Export is local-only.** In server mode the rail's Export row is
  disabled ("Export packages local content only.") — switch the Library
  back to local to package content.
- **Prompt records are portable, not database backups.** They preserve current
  Prompt/Recipe content, separate System/User lanes, keywords, type/format/schema,
  and stored definition. They exclude source IDs/UUIDs, versions, source
  timestamps, deleted rows, retained history, collections, and usage state;
  import assigns ordinary destination-owned identity and lifecycle state. Legacy
  single-`content` Prompt records remain accepted.
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

*Verified against dev @ 023a04a48 — 2026-08-07 (task-2859: the export
quality helper line now matches the selected option instead of always
describing "original"; the queue count line reads "This queue: …" instead
of the self-contradicting "… — in queue" suffix)*

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

*Verified against feat/media-ingest-followups — 2026-08-09 (tasks
3311/3312/3308: "Clear" now always returns the caret to the path field,
even mid-relayout, so the next keystrokes build a path instead of running
a Library search; F1 lists one escape row while the form is open; an
egress-blocked URL's queue receipt reads in plain language with the
allowed_hosts remedy intact (no markup artifacts, no clipped sentence);
the "Some files may fail to import:" dialog never repeats a feature name
as its own explanation; focused option-fold headers show a structural
(heavy side-rail) focus cue with no size change; `.xml` files are
declared unsupported at pre-check and skipped — XML import remains
deferred)*

*Verified against feat/media-ingest-followups — 2026-08-09 (task-3306:
Audio & video gains "Start at" / "Stop at" trim bounds (format-checked in
the form: HH:MM:SS or seconds — and applied exactly once, the video path
used to be able to double-cut), "Cookies file for gated URLs" (a
cookies.txt PATH for yt-dlp video downloads — raw cookie text is never
accepted, since these options persist to config), and "Recursive summary
(map-reduce)"; the Transcription model list now offers the full
faster-whisper catalog (.en variants, Large v1-v3, Distil family,
community Turbo/CrisperWhisper builds) instead of five sizes; the
adaptive / multi-level chunking and chunk-language processor knobs were
audited and deliberately NOT exposed — the audio/video chunker ignores
them end-to-end, so the controls would lie)*

*Verified against feat/media-ingest-followups — 2026-08-09 (task-3307,
ship ruling in task-3310: raster images (.png/.jpg/.jpeg/.gif/.webp/
.bmp/.tiff/.tif) are now a supported import type with their own Images
option fold — "Extract text (OCR)" on by default, "OCR language", and an
"OCR backend" select; the OCR text is the imported content and is chunked
and analyzable like any text import; a no-text image fails its queue row
honestly instead of storing an empty entry; images stay local-only (the
server ingest API accepts no image type); .svg/.ico/.heic/.heif remain
honestly unsupported)*

*Verified against feat/media-ingest-followups — 2026-08-09 (tasks
3313/3314, owner rulings in task-3310: consent is now one grammar — the
"Some files may fail to import:" dialog is retired, and starting with "⚠"
warnings outstanding is an inline two-press at the Start button ("⚠ Press
Start again to import anyway — N files may fail.", second press or second
Enter submits; Esc/edits/a changed pre-check cancel the pending confirm;
no-warning starts stay single-press); "Copy install command" lives under
the warnings; a "Retry this batch" button below the queue (key: `r`)
re-stages the session's last submission — source, options, and metadata
— and re-runs the pre-check fresh, so tooling installed since the last
run changes the forecast)*

*Verified against feat/media-ingest-followups — 2026-08-09 (tasks
3306/3307 xhigh review round: "Stop at" is now absolute on BOTH media
paths — a video trimmed 0:30-1:00 used to yield 0:30-1:30 while the same
pair on an audio file yielded 0:30-1:00; a cookies path that does not
exist is refused at the option boundary and annotated on the queue row
("cookies ignored: …") instead of being silently parsed as cookie JSON,
and a cookies file the user owns is never deleted by the downloader's
cleanup; cookies now also authenticate the pre-download size probe and
metadata lookup, so a gated URL no longer fails before the cookied
download runs; an image URL pre-checks as a web page, matching what the
pipeline actually does with it, instead of promising OCR that never ran;
image OCR text is chunked by the form's chunk size like any other text
import; and the "no OCR backend" warning now follows the OCR manager's
real rules — paddleocr alone, or docext without one of
gradio_client/transformers/openai, counts as no backend)*

*Verified against feat/media-ingest-followups — 2026-08-09 (xhigh review
+ live-verify round): the two-press Start confirm now survives moving
focus out of the path field, so "Press Start again" can be answered with
the Start button after arming with Enter (previously that click cancelled
the consent and merely re-armed, and nothing could submit); picking a
different file with "Browse…" cancels a pending confirm instead of
letting it cover the new file; "Retry this batch" takes two presses when
re-staging would overwrite form content you entered (the button renames
itself "Press again to replace form"), and `r` is now inert exactly while
the button is hidden mid-run; each "Copy install command" button names
its extra in plain text ("(audio)") — the bracketed spelling was eaten by
the renderer and every button read "Copy install command (.)"; a gated
text option keeps its format hint beside the reason it is inert (e.g.
"Cookies file for gated URLs (Netscape cookies.txt · video URLs only) —
needs yt-dlp installed"); a blocked-URL failure row now names the address
it refused; and characters typed immediately after "Clear" are no longer
swallowed by the relayout.*

*Verified against feat/media-ingest-forecast-truth — 2026-08-10 (tasks
14820/14821/14823): the commit-point forecast and the two-press consent
line are now two renderings of ONE computation, so they can no longer
state different numbers for the same selection (live saw "15 will
import" two rows above "7 files may fail", delivering 8 imported / 5
skipped / 8 failed); files whose type group needs tooling this install
lacks are forecast as failures with the reason named ("3 need tooling,
1 empty"), verified against the real import receipt for a mixed folder;
the forecast stays visible while a gate blocks Start; a failure's
"Show details" now states a plain-language reason instead of a raw
category token ("write error" for a file that never reached a write),
prints the underlying tool output once instead of twice, and derives its
retry advice from that reason — the optimistic "a retry can succeed if
the failure was transient" no longer appears under a deterministic
missing-tooling failure, and an unclassifiable cause says nothing at
all; and a selection with nothing importable (an empty folder, or one
whose files are all unsupported) gates Start with its own stated reason
instead of manufacturing a permanent "✗ failed · <folder>" receipt.*

*Verified against feat/media-ingest-forecast-truth — 2026-08-10 (xhigh
review of tasks 14820/14821/14823): the forecast now knows WHICH backend
it is forecasting. Local tooling gaps are local facts, so a server import
is no longer condemned by this machine's inventory — five recordings with
no local audio extra read "0 will import · 5 will fail (need tooling)"
for a batch the server would have transcribed in full, and now read "5
will be sent to the server · server tooling isn't checked from here",
which is the most this app can honestly claim about a machine it cannot
inspect. A capped duplicate check now hedges the import count as well as
the match count. The forecast goes quiet when the runtime has no import
path at all, rather than promising imports beside a permanently dead
Start. A folder whose entries were all skipped (symlinks, hidden files,
unreadable subfolders) is no longer told it is empty — it gets its own
sentence and its own recovery. And the retry advice stopped claiming
tooling was "named above" when nothing named any: a transient executor
teardown is no longer sentenced to "retrying now will fail the same
way", and the generic "no text could be extracted" refusal states the
determinism without pointing at a remedy that was never given.*

*Verified against feat/media-ingest-forecast-truth — 2026-08-10 (tasks
14822/14824/14825/14826): missing optional tooling no longer prints a
wall. A folder that needs eleven optional components now shows one line
("⚠ 3 of 21 files need optional tooling — those imports may fail"), one
**Copy install command** button that copies a SINGLE pip command
installing every missing extra at once, and a **What's missing** fold
holding the per-component detail and its per-extra copy buttons — so the
type breakdown, the options and **Start import** are all on screen at
once, which eleven warnings and nine stacked buttons used to prevent.
The lines that describe your selection ("5 unsupported files will be
skipped", "1 empty file will fail") are now bold, distinct from the
muted environment note above them. Accessibility: the encoding (and
every other) dropdown now shows a heavy border on focus instead of a
colour change only; the path field carries a persistent "File, folder or
URL to import" label that survives being filled in; placeholder text was
raised to meet the AA contrast floor in both enabled and disabled
fields; and an options group whose controls are all disabled — which
Textual removes from the tab order entirely — states the reason on its
collapsible title, which IS a tab stop ("Audio & video — 13 options
unavailable — needs faster-whisper installed"). A collapsed options
panel holding an invalid value is marked in its title ("⚠ Chunk size
needs fixing"), and a collapsed title no longer advertises settings for
controls you cannot edit. In the file picker, the Name/Size/Modified
headers now line up with their own columns whether or not the listing is
long enough to show a scrollbar.*

*Verified against fix/ingest-test-health-and-server-forecast — 2026-08-10
(task-14827): the forecast now asks the backend it is actually aimed at.
Importing on the server, a file that backend refuses — an image, or a
format nothing recognises — is counted as a failure with its reason named
("2 will fail (unsupported by the server)") instead of the local
pipeline's "will skip", which is what the queue really recorded: the
submission raised before it ever left this machine and the row landed as
"✗ failed". The two backends genuinely refuse different sets — an image
imports locally via OCR and a web page is clipped rather than refused —
so the wording says which one is refusing. Missing local extras stop
being presented as blockers during a server-targeted import: the ⚠ block
and its "Copy install command" button are replaced by one quiet note
saying the gap affects imports on this machine only.*

*Verified against dev @ 642567627 — 2026-08-10 (task-4023 AC#1, RC-07:
while its gate is closed the submit button reads "○ Export bundle (.zip)"
— the Library's non-colour disabled marker — and renders at 7.25:1
(measured live; it was 1.44:1), with its F-018 reason tooltip unchanged).*
*Verified for TASK-197 — 2026-08-12 (ADR-057: the widest export scope is
`Everything` across media, conversations, notes, and Prompts/Recipes. Prompt
scope uses uncapped active local IDs, while Skills and Prompt collection/history
lifecycle state remain excluded. Escape returns to the canvas whose Export…
opened the form, including Prompts, or to the hub when entered from the rail.)*
*Verified against fix/ui-background-signal-bounds — 2026-08-10
(task-14910): a 0-byte file is no longer uploaded to the server. The
forecast has always counted one as a certain failure, which was true
locally (the parse chain refuses an empty source before any write) and
merely assumed on the server, where the app sent the file and only the
server decided. The client now refuses it with the reason it already
knows — the row reads "✗ failed · empty.txt · empty.txt is empty; there
was nothing to send." — so the forecast's count is a statement about
this app's own behaviour on both targets, and no round trip is spent on
a file that is almost certainly a mistake.*

*Verified against fix/ui-background-signal-bounds — 2026-08-10
(task-14911): the Start gate now asks the backend the import is aimed at.
A folder of nothing but images used to forecast "0 will be sent to the
server · 3 will fail (unsupported by the server)" with **Start import**
still live, and pressing it queued three rows that could only land as
permanent failures — the guaranteed-failure submit the gate already
prevented on this machine, one backend over. The gate reads the same
forecast the commit line does, so the two cannot state different
numbers, and it keeps its two vocabularies apart: "Nothing in this
selection can be imported" for files nothing here can read, "Nothing in
this selection can be sent to the server" for files that only this
destination refuses. The refusal is enforced at the submit itself, not
just on the button, so Enter in the path field cannot route around it.*

*Verified against feat/library-queue-batch @ 0662e09f5 — 2026-08-11
(task-14902: the quality control converged on the Library chooser-strip
pattern — pressing "quality: thumbnail" opens a one-row strip of all
three values under the still-visible button with ✓ on the active one; a
pick applies directly and updates the helper line, a second press or
Escape cancels (Escape never leaves the form while the strip is open),
and the footer/F1 read "enter choose quality / esc cancel" meanwhile.)*

*Verified against the chunking-engine-parity worktree — 2026-08-19
(chunking-engine-parity, doc-only on this page): the chunking engine behind
every import path is now the server's engine (vendored at dev@385afa95
behind a compatibility shim). The sanitization behavior (null bytes →
spaces, length-preserving NFC normalization, control-character and
bidirectional-override neutralization) is the engine's `_sanitize_input`
(`tldw_chatbook/Chunking/engine/chunker.py`), the full method roster is the
shim's `_LEGACY_METHOD_MAP` plus engine-native pass-throughs
(`tldw_chatbook/Chunking/Chunk_Lib.py`), `tiktoken`/`defusedxml` are core
dependencies (`pyproject.toml`), and the engine-version stamp
(`parity-1@385afa95`, media DB schema v6) plus the RAG Admin legacy-chunk
report are Phase C — pinned by
`Tests/DB/test_media_db_schema_v6.py`,
`Tests/Local_Ingestion/test_engine_version_stamp.py`, and
`Tests/Chunking/test_callsite_characterization.py`. The import form's own
controls are unchanged; only what the chunking layer does underneath
moved.*

*Verified against feat/chunking-template-parity — 2026-08-21
(chunking-template-parity task 11: the Import behavior fold gains the
"Chunking template" picker documented above — default "None (manual
settings)", hidden in server mode, markup-escaped labels, populated from
the template store off the mount path; `[chunking] default_template` ships
in the config template; template imports fill the `chunking_template` /
`chunking_params` chunk columns and `Media.chunking_config` in a shape both
existing readers (`get_documents_using_template`'s LIKE,
`get_template_statistics`' `json_extract`) round-trip; and the
offset-basis caveat for template chunks is the paragraph above. Pinned by
`Tests/UI/test_library_ingest_template_picker.py`,
`Tests/Local_Ingestion/test_ingest_template_persistence.py`, and
`Tests/test_config_chunking_defaults.py`.)*

*Verified against feat/chunking-auto-selection — 2026-08-22
(chunking-auto-selection, tasks 1-5: the "Chunking template" picker gains
the "Auto" option documented above — value the reserved sentinel name
`"auto"` (`Chunking/auto_selection.AUTO_SENTINEL`), None still the
default, the sentinel stripped from server-mode ingest kwargs. The
three-tier decision is `Chunking/auto_selection.resolve_auto` over the
vendored planner (`Chunking/engine/auto_planner.py`, manifest-moved from
excluded), with the media-type vocabulary pinned by
`Tests/Chunking/test_media_type_vocabulary.py` and planner parity by
byte-pinned fixtures (`Tests/Chunking/test_auto_planner_parity.py`).
Persistence (`mode`/`auto_tier`/`auto_rationale`, `template` key only on
a template-tier win) and re-chunk re-resolution — including the re-stamp
of `Media.chunking_config` with the re-resolved outcome, so a tier flip
on re-chunk never leaves a stale template name for the readers to count —
are pinned by `Tests/Local_Ingestion/test_ingest_template_resolution.py`,
`Tests/UI/test_library_ingest_template_picker.py`, and
`Tests/Library/test_library_rechunk_service.py`. Template CRUD refuses
the reserved name `auto` on create and rename.)*

*Verified against task/19556-burn @ f12bb21ad — 2026-08-22 (TASK-19556 (a)):
the import pre-check no longer contacts a URL. It used to fetch the address's
headers 0.8 s after you stopped typing — before you had asked for anything to
be imported — and the three answers it could get back (refused / answered
with a status / clean) were each rendered differently in the summary, so
pasting a link read out the state of whatever the address pointed at,
including hosts on your own network. Pasting is now inert; the address is
classified by name, exactly as a local path is classified by its extension.
A link check remains available behind `[library]
ingest_url_preflight_probe = true`, and in that mode it runs only from the
deliberate triggers (leaving the field, Enter, Browse…, "Retry check"
— note Textual also reports the field as left when the terminal itself
loses focus), is
routed through the `[web_security]` egress policy, follows no redirects, and
reports one identical "could not be checked" note for every address the
policy declines. Pinned by `Tests/Library/test_ingest_preflight_egress.py`
and `Tests/Library/test_ingest_preflight.py`.*
