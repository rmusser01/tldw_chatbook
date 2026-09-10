---
target: "Library screen + all subscreens (critique #9, whole-Library scope, post critique-8 fix wave)"
total_score: 22
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 7
timestamp: 2026-09-10T14-50-24Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated on separate scratch profiles and tmux sockets, fresh-eyes, no prior snapshot; parent reconciled the two and re-tested the disputed P0 against the app log and both export artifacts). Target `tldw_chatbook/UI/Screens/library_screen.py` at dev 02374bf66a (2026-09-09, the tip after the critique-8 fix wave landed: PRs #2517 #2519 #2523 #2524 #2525 #2528 #2531 #2533). Scope = the whole Library. Live at 235x52, 100x30 and 60x24 on a brand-new profile and a seeded profile (11 media, 6 conversations, 7 notes, 5 prompts, 2 skills); no LLM provider configured. The local Import success path was NOT exercisable: this Mac's POSIX semaphores are exhausted, so every parse pool fails at start (host condition, recorded, not a defect). 143 captures.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | A selected-media export writes a bundle with zero content and reports success (P0, proven in the app log) |
| 2 | Match System / Real World | 2 | The blocked RAG Answer panel tells a first-timer to set `OPENAI_API_KEY` or edit `[api_settings.openai]`; one feature carries three names (Collections / Quick Capture / captures) |
| 3 | User Control and Freedom | 3 | Undo, Trash, Cancel and Dismiss remain strong; Escape is advertised and inert on the Notes list and in the Notes filter box |
| 4 | Consistency and Standards | 2 | `○` means disabled, unchecked and skipped on three canvases of one screen; Escape behaves differently on Media and Notes; every list has its own toolbar dialect |
| 5 | Error Prevention | 2 | Nothing stops or flags an archive that contains none of the items the scope line promised |
| 6 | Recognition Rather Than Recall | 2 | The Media type chooser marks focus with a 1.09:1 background difference and no glyph; the destination picker has no path field |
| 7 | Flexibility and Efficiency | 3 | The accelerator set is rich and the review-set walk is excellent; the footer drops half the keys at 100 columns with no cue |
| 8 | Aesthetic and Minimalist Design | 2 | Five canvases are about 90% empty while the blocked RAG panel prints one sentence four times |
| 9 | Recognize/Diagnose/Recover from Errors | 2 | The import queue's failure grammar is the best in the app; the export failure is silent and the RAG remedy is config-speak |
| 10 | Help and Documentation | 2 | 41 guide claims exercised: 24 hold, 12 contradicted, 5 partial, within 48 hours of a docs sweep |
| **Total** | | **22/40** | **Acceptable** |

Assessment A scored 20; B filed 17 defects with no P0. I reconciled to 22: A's 1s on Consistency and Aesthetics describe details diverging around main flows that do match (the rubric's 2), and I confirmed A's P0 rather than B's absence of one, because B only exported the "Everything" scope, which takes a different code path.

## Design Specificity Verdict

Mostly authored for this product, wrapped in a category-interchangeable frame (both assessors, independently). The Media reader's three stable panes, the review-set walk with its agreeing banner and footer, the two-line item rows, the delete → receipt → focused Undo → Trash chain, and the import pre-flight that names its skips are unmistakably this product's. The landing hub (ten lines in a 190x44 canvas), the Study staging canvas, Prompts and Skills (a 48-cell list beside a 145-cell "select something" pane) and File Notes (two lines on a full screen) would fit any left-rail TUI. The clearest evidence: Media earned a rule for large canvases ("with no item open, Items takes the empty Reader's width") and no sibling canvas got it. The detector cannot read Python or TCSS (both targets return an empty scan with zero files read), so every claim here is live-measured.

## What the critique-8 wave fixed, confirmed by fresh eyes

Stored notes open again (the P0 of critique #8 is gone on both profiles). Escape leaves the Media filter box and lands where the footer says. `s` enters select mode from a focused row with the Reader open. Focused list rows carry the `█` bar; focused grips invert. Blocked Generate and Analyze name their next step inline. The import row's failure copy is now the best error grammar in the app (`✗ failed · cause · remedy`, raw errno behind Show details, unsupported files `○ skipped` with no Retry). The `‹ Library` return exists below 64 columns. The Chunking Lab strip is gone from the canvas header. Undo restores the database row (verified).

## Overall Impression

The screen's authored half is genuinely good and got better; its frame is still generic, and a fresh whole-screen pass finds as much as the last wave closed. The single biggest finding is not a UX polish item at all: the archive path a local-first product exists for silently writes an empty bundle for the scope a power user is most likely to use. Fix that first; then make Notes honour the same Escape contract Media now honours; then unify the glyph and blocked-state grammar so the good patterns stop contradicting each other.

## What's Working

- **Delete → receipt → focused Undo → Trash** is the template for every destructive action, and it now verifiably restores the row (B checked the DB).
- **Import's failure grammar**: `✗ failed` (retryable, three actions) versus `○ skipped · Unsupported file type: .json.` (not retryable, one action), with the pre-flight naming the skips before anything runs.
- **The Media reader's reading discipline**: five rows of chrome above the first line, prose at ~90 columns, a byline only when there is an author, the 98-character CJK title intact at 235 and cleanly truncated at 100.
- **Inline blocked reasons** (`○ Generate · No analysis provider is configured · Set one in Settings ▸ Providers & Models.`) are readable without hover and are the pattern the RAG panel should adopt.

## Priority Issues

- **[P0] A selected-media export writes an empty bundle and reports success.** Alex selects two media items, the canvas says "Selected media · 2 items", a destination is chosen, "Last export: …" appears; the zip holds only README and a manifest with `content_items: []`. PROVEN: the app log reads `Error collecting media local:media:10: invalid literal for int() with base 10: 'local:media:10'` for both items. Select mode carries the canonical display id (`local:media:<n>`), the selected-scope path in `library_export_scope.py` passes those ids through untouched (`if scope.ids: return {…: list(scope.ids)}`) while the "everything" path normalises with `str(int(id))`, and `ChatbookCreator._collect_media` does `int(media_id)` inside a broad `except Exception` that logs and continues. The "Everything" scope works (B's bundle holds 12 media items). Pre-existing, not a wave regression: the canonical id dates from 2026-08-16 and the passthrough from 2026-07-12; the pinning tests feed bare "1","2","3" and `str(selected_id)`, so they never see the real id shape. Fix: normalise ids at the scope seam with the existing backing-id coercion (`library_media_state.py:202`), make the creator raise rather than swallow when a non-empty selection collects nothing, and render the receipt from the artifact (`✓ exported · N items · X KB · path`, `✗` with Retry on zero items). Suggested command: /impeccable harden.

- **[P1] Notes does not honour the Escape contract Media now does.** In the Notes filter box Escape is a no-op and the next letter is typed into it (B D2, both profiles); on the plain Notes list Escape never reaches the rail's search box although the footer promises "esc focus rail", and typed keys are swallowed until Enter reopens the note (B D3, A cap 15). The critique-8 pin (`test_library_crit8_keyboard.py::test_escape_from_a_list_filter_box_still_goes_where_the_footer_says`) drives Media only; the `_library_list_focus_rail_target()` helper returns the rail box for Notes but the hop does not take effect on that surface. Fix: extend the gate and the hop to the Notes canvas (and the folder-tree layout), pin both cases. Suggested command: /impeccable harden.

- **[P1] Info states an untruth about the item you are reading.** A `document` whose stored text starts `# Roadmap sync` shows literal Markdown with no Rendered|Raw toggle, and Info says "No Markdown formatting to render — showing the stored text." PROVEN: `_MARKDOWN_MEDIA_TYPES` in `library_media_viewer_state.py` allowlists plaintext/markdown/obsidian_note/video/audio and returns False for every other type before the content sniff runs. Fix: sniff content for every type, or say the real rule ("Rendered view is available for markdown, transcripts and plain-text items — this is a document"). Suggested command: /impeccable clarify.

- **[P1] `○` carries three meanings on one screen.** Disabled (`○ Export selected`), unchecked (`○ Media (0)` in the Sources panel) and a settled outcome (`○ skipped · weird.xyz`); meanwhile a leading `▸` marks both the selected rail row and an expandable node. The guide codifies `○` as "the disabled marker" and `▸` as "the selected row"; neither holds. Fix: one legend, `▸/▾` trailing for disclosure, `█` leading for the cursor, `☐/☑` for selection, `✓/✗/–` for outcomes; blocked actions keep their inline reason and drop the glyph. Suggested command: /impeccable polish.

- **[P1] The blocked RAG Answer panel prints one sentence four times and speaks in config-file.** `Blocked.` / `Unavailable: …` / `Why: … Set OPENAI_API_KEY or add api_key under [api_settings.openai].` / `Next: …` / `Recovery: <same sentence>` / `Owner: LLM provider credential.` Two rows away, the Media reader's identical condition says "Set one in Settings ▸ Providers & Models." One missing key, two remedies, one of them TOML. Fix: collapse to the Media pattern plus an "Open Settings ▸ Providers" action; keep the structured record in the log. Suggested command: /impeccable clarify.

- **[P1] The Media type chooser marks focus by colour alone.** ANSI decode: the focused row is background 30,30,30 against 39,39,39 elsewhere, a 1.09:1 ratio, darker than its neighbours, no glyph; only the active value has a `✓`. The footer has committed the user to a keyboard interaction whose cursor they cannot see. Fix: the `█` left-edge bar the lists already use. Suggested command: /impeccable harden.

- **[P1] File Notes drops the rail at 235 columns and ignores the configured notes folder.** Selecting Notes ▸ Folder files replaces the whole Library frame with a full-bleed pane and a breadcrumb although the guide says the rail stays beside it at 120 columns and wider; the pane says "No folder selected" while `[notes] sync_directory` holds three markdown files. Measured at 02374bf66a, BEFORE PR #2543 (fix/library-notes-file-notes) landed on dev an hour later: re-verify on the new tip before filing a fix. Suggested command: /impeccable adapt.

## Full issue register

| # | Sev | Where | Issue | Evidence | Already tracked |
|---|---|---|---|---|---|
| 1 | P0 | Export | Selected-media scope writes zero content; success reported; `int('local:media:N')` swallowed | A 43–46 + zip + app log | no |
| 2 | P1 | Notes | Escape inert in the filter box (next key typed) and on the list (footer promises the rail) | B D2/D3, A 13/15 | no |
| 3 | P1 | Media reader | "No Markdown formatting to render" false for `document`; type allowlist precedes the sniff | A 37/38 | no |
| 4 | P1 | All canvases | `○` = disabled / unchecked / skipped; leading `▸` = selected row and disclosure | A 16/21/41/60 | no |
| 5 | P1 | Search/RAG | Blocked panel: four repeats, `Owner:`, env-var remedy contradicting Media's | A 18 vs 40/41 | no |
| 6 | P1 | Media | Type chooser focus is a 1.09:1 background, no glyph | A 33 ansi | no |
| 7 | P1 | File Notes | Full-screen takeover at 235 contra guide; configured sync folder not offered (pre-#2543) | A 57 | re-verify after #2543 |
| 8 | P1→rider | Conversations | Blocked state reads as two "Open in Console" controls; `c` refuses silently | A 54/55 | rider 32101 (update) |
| 9 | P2 | Rail | "Search Library…" row is 2 cells too wide and clips the canvas frame on every canvas (likely the new clear button) | B D1, A 04/06 | no — wave regression to confirm |
| 10 | P2 | Media | 0-result filter hides the whole toolbar incl. the active type facet | B D5 | no |
| 11 | P2 | Media | Entering the list focused the type chooser, not row 0 (B's repro had prior keypresses; a pin asserts the opposite) | B D4 | no — needs clean repro |
| 12 | P2 | Notes | Sort chooser present only on the empty list; folder verbs ungated | A 10/13/56 | no (peer Notes wave may touch) |
| 13 | P2 | Import | "Show details" throws focus to the Keywords field 25 rows up | A 08/09 | no |
| 14 | P2 | Landing, Study, Prompts, Skills, File Notes | No density rule: five canvases ~90% empty; Analysis actions pinned to the pane floor | A 11/22/23/30/40/57/58 | no |
| 15 | P2 | Rail/Notes | Vocabulary drift: Collections/Quick Capture/captures; five names for Notes | A 20/21/56/57 | 32057 (Collections); Notes naming new |
| 16 | P2 docs | library.md | Still documents the Chunking Lab strip under every header; Actions group is below the rail fold at 52 rows | B D8 | no |
| 17 | P3 | Rail | Heading truncates to "Navigati" at 100 columns | B D6 | no |
| 18 | P3 | Export | "1 notes" pluralisation | B D7 | no |
| 19 | P3 | Media viewer | Three-way mismatch on the Escape chip vs two guide claims | B D9 | no |
| 20 | P3 | Skills | List cannot distinguish an approved skill from an unapproved one | B D10 | no |
| 21 | P3 | Collections | Filtered-empty wording for an empty source; disabled pager chrome | B D11 | 32057 nearby |
| 22 | P3 | Media | Undo of a bulk delete re-dates the items to now and reorders (guide: "restore never rewrites") | B D12, A 51 | no |
| 23 | P3 | Notes | List title does not live-update while the editor is open | B D13 | 32106 nearby |
| 24 | P3 | <64 cols | Escape on a list is a no-op; `‹ Library` never advertised | B D14 | no |
| 25 | P3 | Rail | Unsubmitted search text carries across canvases into the RAG query box | B D15 | 32069 claimed emptying — regression check |
| 26 | P3 | Media | "2 selected┃ Select all…": no gap before the focused button | B D16 | no |
| 27 | P3 | Conversations | Footer offers neither "esc focus rail" nor a `/` hint | B D17 | no |
| 28 | P3 | Export | Destination picker has no path field; Ctrl+A is move-to-start | A 44 | no |
| 29 | P3 | Rail | DB-sizes line wraps mid-value; "Handoff · 0 eligible, ● 1 blocked" is unactionable | A 24/25 | no |
| 30 | P3 | Import | Four identical failed rows × three buttons, no dismiss-all | A 60 | no |

## Persona Red Flags

**Jordan (first-timer):** the first arrival offers "Back to Get started" for a view never seen; Collections opens a canvas headed Quick Capture whose empty state blames a filter that is not set; RAG Answer's block tells them to set an environment variable; Escape on the Notes list swallows the next three keystrokes silently. Abandons at the first blocked feature or the first swallowed keys.

**Alex (power user):** the archive is empty and nobody said so (P0); the destination picker has no path field; the footer drops half the accelerators at 100 columns with no cue; `c` on a gated conversation does nothing; Undo re-dates restored items; no Sort on a populated Notes list. Keeps Media, distrusts the rest.

**Sam (keyboard-only, low vision):** the type chooser's cursor is invisible in plain text; "esc focus rail" is false on Notes and inert at 100x30 on Media; at 100x30 the only route back to the collapsed rail is an unlabelled `--->` glyph in the list gutter; `○` is the screen's shape-not-colour promise and it means three things. What works: the `█` row bar, the heavy focused-Input frame, the inline text reasons.

**Riley (stress):** a resize to 60 columns mid-export silently discards the canvas and the bundle name being typed; the rail scrolls past its own search box while the footer still says "/ focus search". CJK titles, `weird.xyz`, `export.json` and `nested/deep-note.md` are all handled correctly.

**Solo builder/operator:** source authority is well labelled ("Library | Local", "Source · Local", "local only"); the honest archive they need most is the P0.

**Researcher/student:** the Study hand-offs are honest and clear; the middle of their pipeline is blocked with two contradictory remedies; highlights exist per item but there is no cross-item highlights view.

## Docs vs Live

41 concrete claims exercised by B: 24 hold, 12 contradicted, 5 partial. A found 11 divergences. The ones that matter: the rail beside Folder files at ≥120 columns (measured before #2543); the Collections canvas title; the Rendered|Raw rule for non-allowlisted types; "2 of 5 · type: pdf" status line; "restore never rewrites the item"; the single-row Media toolbar (three rows live); rail glosses "drop first on narrow terminals" (Collections has its gloss only at 60); Escape on the plain list; note folders and their four verbs undocumented; the Chunking Lab strip still documented under every header.

## Minor Observations

Undo of a bulk delete re-sorts restored items to the top. The import queue shows four identical failure rows with no dismiss-all. The `--->` grip in the item gutter reads as a row pointer at 100x30. "Handoff · 0 eligible, ● 1 blocked" names a count and a colour dot but not what is blocked. Select mode puts ten rows of chrome above the first item. The rail's Details ▸ Actions sits below the fold at 52 rows with no scroll cue.

## Improvement opportunities beyond the fixes

1. **One state-glyph system with a legend** (`▸/▾` trailing disclosure, `█` leading cursor, `☐/☑` selection, `✓/✗/–` outcomes; blocked actions keep the reason, lose the glyph). Sam, Jordan. M.
2. **One shared list-canvas component** (`filter · type · sort · scope actions · select`) for Media, Conversations, Notes, Prompts, Skills, Collections; fixes the Notes sort gap, the Conversations footer gap and the widening asymmetry at once, and lets the footer be generated. Alex, Sam. L.
3. **Promote "an empty pane gives its columns away"** from Media to a screen-wide rule. Alex, researcher. S.
4. **An export receipt read back from the artifact**: reopen the zip, count content items, `✓ exported · N items · X KB · path` with Reveal/Dismiss; zero items becomes `✗` with Retry. The structural fix for the P0. Solo operator. M.
5. **A cross-item Highlights view** (`Highlights (N)` Browse row listing every saved quote with its source). Researcher. M.
6. **Grouped import queue rows** (`✗ failed · 4 files · reason` with Show the 4 files / Retry all / Dismiss all). Alex, Riley. S.
7. **A path field in every file dialog** with `~` and tab completion. Alex, solo operator. S.
8. **A "what is this?" line on every empty destination**, sourced from the same string as the rail gloss so vocabulary cannot drift. Jordan. S.

## Questions to Consider

- What is the landing hub for, if the rail is always present? Should Library land on the last-used destination with Continue and Needs attention folded into the rail?
- Why does every list get its own toolbar dialect? What would one shared component with per-canvas slots cost?
- Should a blocked capability ever explain itself twice, differently? What would it take for every gate to resolve reason and remedy from one place?
- Is Collections a Library destination at all, or a filter on Media?
- What does Library owe the user after an export? If the answer is "a receipt you can trust", the receipt must be read back from the artifact.
- Two findings are mine to own: the rail search row that now overruns the pane by two cells arrived with the clear button I specified in task 32069, and the "esc focus rail" pin I accepted in the keyboard branch covered Media only. Proposals: size the clear button inside the row's existing width, and extend the Escape pin to every list canvas rather than one.

## Environment note

Every local import failed at parse-pool start (`[Errno 28]`) because this Mac's POSIX semaphores are exhausted; the user has decided not to reboot. Import's success path, "Open in Library" and cancel-mid-queue remain unverified live; the failure presentation was assessed and is good.
