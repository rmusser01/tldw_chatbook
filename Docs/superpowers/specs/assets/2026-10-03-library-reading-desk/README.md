# Library reading desk: mockups (2026-10-03, revision 3: the working note)

These mockups are for owner approval of
[`2026-10-03-library-reading-desk-design.md`](../../2026-10-03-library-reading-desk-design.md), spec revision 2,
§10. Each state has three files: `.png` (for viewing), `.svg` (Textual's own screenshot) and `.txt` (a plain-text
frame).

**What changed in revision 3.** The owner ruled (§3 Q2) that the desk holds your **working note**: one note that
stays open while you move between documents and cites each document it quotes. It replaces the per-document
companion note.

- The note here is **"Retrieval practice — lit review"**. It has a short `## Key claims` section and two
  attributed quotes:
  - `paper-retrieval-practice`, ¶12
  - `Spaced repetition, explained`, ¶3
- Links show as `title ↗` while editing (§3 Q6 / §5.6, slice D7). State 12 shows how the raw Markdown appears when
  the caret is inside a link.

## How they were made

- **The running app.** The real app (this worktree at `200202cdcf`, Textual 8.2.8) was started headless with
  `run_test`. It used an isolated copy of the 2026-10-02 review profile. Library › Media › the document was opened
  through real clicks.
- **Grafted desk pieces.** Only the desk-specific parts were added to the live widget tree:
  - the companion pane
  - the grip label `Note`
  - the toolbar **Note** button
  - the `¶n of N · Quote` status
  - the paragraph cursor
  - the stacked-faces header

  Everything else comes from the running app: nav bar, header, shell, grips, reader, tabs, footer and stylesheet.
- **The companion reuses the note editor's own ids.** These are `#library-note-body`, `#library-note-chrome-facts`
  and `#library-note-context-*`. The Library stylesheet therefore styles the companion the way it styles the note
  editor today.
- **Labels and rows come from the app's own helpers:**
  - `‹ Items` uses `back_cue_label()`.
  - The compact mode strip and the current-note mark use `LIBRARY_CHOICE_ACTIVE_MARKER` (`✓`).
  - Picker rows are built with `library_row_button` and the Media list's row classes.
- **The footer uses the real footer API** (`set_workbench_shortcuts`).
- **Pane widths come from the spec formula** `clamp(round(0.40 × available), 40, 72)`. They are applied through
  the shell's own `sync_layout` and measured from the rendered regions.
- **PNG files** are the SVGs rendered by headless Chromium and reduced to a 256-colour palette. The window title
  bar ("tldw chatbook") comes from Textual's SVG template.
- **Fixture data**, in the profile copy only:
  - The paper's stored path is the neutral `file:///Users/you/Papers/paper-retrieval-practice.pdf` (privacy).
  - The paper's text is the capture's abstract split into blank-line blocks (31 blocks, so ¶12 is the Results
    paragraph).
  - The seeded article "Spaced repetition, explained" (plain text, Raw view only) had its filler sentences
    replaced with seven short paragraphs, so the quoted ¶3 reads naturally.
  - The picker shows the seeded notes with the ages the review captures show (Ideas inbox 1h, Reading list 2h,
    Thesis outline 3h).
  - "Exam prep — memory techniques" was invented as the one older note that cites the paper.
  - The list's "updated" ages were shifted so they match the capture's clock.

## Files

| State | Terminal | What it shows | Focus |
|---|---|---|---|
| `00-before-media-reader-160x45` | 160x45 | **Today, unchanged** (apart from the stored path). The Media reader with the PDF open: Navigation, Items and Reader | Navigation rail |
| `01-desk-235x52` | 235x52 | The desk side by side. Header row 1: `‹ Items  Working note · Retrieval practice — lit review  cites 2 ▾`. The note has Key claims plus two attributed quotes from two documents. Status: `Saved 20:12 · ctrl+s save` | Note body |
| `02-desk-160x45` | 160x45 | Just after `q` on ¶12. Status: `Quoted ¶12 of paper-retrieval-practice`, with the caret after that quote. The reader's ¶12 bar is muted because the reader is not focused | Note body |
| `03-desk-120x36` | 120x36 | The smallest side-by-side size. Header: `‹ Items  Retrieval pra… ·  cites 2 ▾`. Compact mode strip: `✓ Edit  Preview  Info … Save`. The reader is focused (blue bar on ¶12) | Reader |
| `04-stacked-note-100x30` | 100x30 | Stacked faces, Note face. Header: `‹ Items  Note · Retrieval practice… (saved 20:12) ⇄ Reading · paper-retrieval-practice`. The `cites 2 ▾` chooser sits on the status row | Note body |
| `05-stacked-reading-100x30` | 100x30 | Stacked faces, Reading face, with the header order reversed and the ¶12 cursor | Reader |
| `06-stacked-60x24` | 60x24 | Stacked faces at 60x24, Note face. Header: `‹ Items  Note · Retrieval… (saved 20:12) ⇄ Reading`. Save and `cites 2 ▾` are both still visible | Note body |
| `07-write-in-picker-160x45` | 160x45 | **No working note yet, so `n` opens Write in…** (§4.1). It has a filter field and three groups. `Notes that cite this document (1)` has one row, highlighted `▸` because Enter picks it. `Recent notes` has three seeded notes with ages. `New` has `+ New note` with its Title field pre-filled `paper-retrieval-practice`. Footer: `enter pick \| esc cancel` | Filter field |
| `08-swap-working-note-160x45` | 160x45 | **The `cites 2 ▴` picker reopened to swap the working note.** The current note is listed first and marked `▸ ✓ Retrieval practice — lit review · now`. The groups are the same as in 07 | Filter field |
| `09-note-info-sources-trash-160x45` | 160x45 | The note's Info tab (header `cites 3 ▾`). `Sources (3)` lists `paper-retrieval-practice ↗ · pdf`, `Spaced repetition, explained ↗ · article` and `Make It Stick — chapter 2… · In Trash · Restore` | Restore |
| `10-reader-paragraph-cursor-160x45` | 160x45 | The paragraph cursor moved with `j` to ¶13 (`¶13 of 31`). Footer: `n note beside \| q quote \| j/k paragraph \| ]/[ next item \| esc close desk` | Reader |
| `11-second-document-160x45` | 160x45 | **Research across items.** The reader now shows "Spaced repetition, explained" (byline N. Ahmed; plain text, Raw view, cursor on ¶3, `¶3 of 7`). The **same** working note stays on the left with both quotes. Status: `Quoted ¶3 of Spaced repetition, explained` | Reader |
| `12-caret-in-link-160x45` | 160x45 | Same as 02, but the caret is inside the ¶12 attribution. Only that link shows its raw Markdown, `[paper-retrieval-practice, ¶12](media://2e9fe89e-…#p12)`, which wraps over three rows at this width. The ¶3 link stays tidy (`… ¶3 ↗`). Caret position: `7:32` | Note body |

**Measured pane widths** (cells; unchanged since revision 1):

| Terminal | Nav grip | Companion | Note grip | Reader |
|---|---|---|---|---|
| 235x52 | 5 | 72 | 5 | 149 |
| 160x45 | 5 | 58 | 5 | 88 |
| 120x36 | 5 | 42 | 5 | 64 |
| 100x30 | 5 | stacked | — | 95 |
| 60x24 | 5 | stacked | — | 55 |

**Spec table note.** The formula gives 58/88 at 160 columns and 42/64 at 120. The §6.2 table says ~60/~85 and
~45/~60, so the table should be updated to match the formula.

## Fidelity check

`00-before` was compared with the review capture `j7-researcher-loop/07-pdf-reader-settled-160x45.ansi`. Each
cell's character, foreground, background, bold and underline were compared. Row 2 was left out in both runs: in
the review capture, a stray huggingface `WARNING` log line was printed over the nav bar.

- **Control render, with the fixture's original path: 0 cells differ.**
- **The render shipped here: 1167 cells differ, all inside the reader pane** (columns 101–160, rows 7–34). The
  neutral path fits on one byline row instead of four, so the reader body moves up three rows. Everything outside
  the reader is identical.

## Rulings applied

- **§3 Q2, the working note.** One note across documents, shown with `cites N ▾`.
  - When no note is set, `n` opens Write in… (07).
  - `▾` reopens the picker to swap the working note (08).
  - `]` or opening another item keeps the note on screen (11).
- **§3 Q6 / §5.6, tidy links.** Links show as `title ↗` in Edit, and the raw Markdown appears only while the caret
  is inside the link (12).
- **§4.4a, Save never drops off.** Below 46 content cells the mode strip becomes `✓ Edit  Preview  Info … Save`.
- **§4.4a, `‹ Items`.** It is on header row 1, in the Write in… picker, and at the start of the stacked-faces
  header. The reader's `‹ Back` is hidden while the desk is open.
- **§5.2, paragraphs.** Paragraphs are the blank-line blocks of the stored text, and the heading counts as ¶1. The
  cursor and `¶n of N` also work on Raw-only items (11).

## Mockup choices the spec does not state yet (please confirm)

1. **Header row 1 compaction.** The full row `‹ Items  Working note · <title> · cites 2 ▾` needs 70 cells:
   - At 235 columns it fits once the trailing `·` is dropped.
   - At 160 (54 content cells) the `Working note · ` prefix goes first, so the note's full title stays readable.
   - At 120 the title is shortened (`Retrieval pra… ·`).
2. **Stacked faces.** The face header follows the §6.3 shape, `Note · <title…> (saved 20:12)`. The `cites N ▾`
   chooser moves to the status row so the swap stays reachable.
3. **Picker marks.** `▸` marks the row Enter picks (the Media list's marker). `✓` marks the current working note
   (the Library choice marker). Recent notes leave out notes already shown in the cite group.
4. **Same focus border in both panes** (§7, review S-14). When the reader is focused it uses the heavy blue edge the
   note body uses. Today's reader uses an amber border (`$accent`).
5. **Wide mode strip** uses the reader's `(selected)` wording.
6. **Paragraph cursor.** It uses the app's row-focus edge (a thick `█` in the focus colour) when the reader has
   focus, and a muted edge when the note has focus.
7. **Raw view gutter.** In Raw the cursor needs a two-cell gutter, so text there starts two cells further right
   than in today's Raw view (11).

## Spec wording to reconcile

- **Receipt wording.** §4.3 gives the receipt as `Quoted ¶12 of <document>`, and the mockups follow it. §4.4a row 2
  still says `Quoted ¶12 into …`.
- **Stacked header example.** §6.3's example still reads `Note · Notes — paper… (saved 20:12)`, from the
  per-document design. The mockups use the working note's title.

## Known render artifacts

These come from the renderer, not from the app:

- Chromium's font draws `⇄`, `↗` and `✓` small or as look-alikes.
- Chromium's font squeezes the CJK row (`学習メモ`) in `00`.
- Heavy box lines (`┏━┓`) are drawn as thin lines.
