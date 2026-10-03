# Library reading desk: mockups (2026-10-03, revision 2)

These mockups are for owner approval of
[`2026-10-03-library-reading-desk-design.md`](../../2026-10-03-library-reading-desk-design.md) (§10).
Each state has three files: `.png` (for viewing), `.svg` (Textual's own screenshot) and `.txt` (a plain-text frame).

Revision 2 applies the coordinator's rulings of 2026-10-03:

- §4.4a: Save never drops off the mode strip.
- §4.4a: the companion header gains `‹ Items`.
- §5.2: paragraphs are the blank-line blocks of the stored text.
- Privacy: the document's stored path now shows a neutral `file:///Users/you/…` path.

## How they were made

- **The running app.** The real app (this worktree at `200202cdcf`, Textual 8.2.8) was started headless with
  `run_test`. It used an isolated copy of the 2026-10-02 review profile, the j7 run that had already imported the
  paper. Library › Media › `paper-retrieval-practice` was opened through real clicks.
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
- **Labels come from the app's own helpers.** `‹ Items` uses the app's `back_cue_label()`. The compact mode strip
  uses `LIBRARY_CHOICE_ACTIVE_MARKER` (`✓`).
- **The footer uses the real footer API** (`set_workbench_shortcuts`). It shrinks with the window the same way the
  real footer does.
- **Pane widths come from the spec formula** `clamp(round(0.40 × available), 40, 72)`. They are applied through
  the shell's own `sync_layout`. The widths below were measured from the rendered regions.
- **PNG files** are the SVGs rendered by headless Chromium and reduced to a 256-colour palette. The window title
  bar ("tldw chatbook") comes from Textual's SVG template.
- **Fixture data**, in the profile copy only:
  - The paper's stored source path was replaced with `file:///Users/you/Papers/paper-retrieval-practice.pdf`. The
    review fixture's path contained a scratch directory and the host user name.
  - Paragraphs (§5.2) are the blank-line blocks of the stored text, and the heading counts as ¶1. In the paper
    (31 blocks), ¶12 is the Results paragraph.
  - The list's "updated" ages were shifted so they match the capture's clock.

## Files

| State | Terminal | What it shows | Focus |
|---|---|---|---|
| `00-before-media-reader-160x45` | 160x45 | **Today, unchanged** (apart from the stored path, see Fidelity check). The Media reader with the PDF open: Navigation, Items and Reader | Navigation rail |
| `01-desk-235x52` | 235x52 | The desk side by side. Navigation is collapsed to its grip, the companion is 72 cells and the reader 149. Header row: `‹ Items  Note · Notes — paper-retrieval-practice · 1 of 2 ▾`. The note has a source line, `## Key claims`, a bullet and the ¶12 quote | Note body |
| `02-desk-160x45` | 160x45 | The desk just after a quote. The note status shows the receipt `Quoted ¶12 into Notes — paper-retrieval-practice`. The reader's ¶12 bar is muted because the reader is not focused | Note body |
| `03-desk-120x36` | 120x36 | The smallest side-by-side size: companion 42, reader 64. The **compact mode strip** reads `✓ Edit  Preview  Info … Save`, so Save stays visible. The header drops its `Note · ` prefix to keep the title legible. The reader is focused (blue bar on ¶12) | Reader |
| `04-stacked-note-100x30` | 100x30 | Stacked faces, Note face active. Header: `‹ Items  Note · Notes — paper… (saved 20:12) ⇄ Reading · paper-retrieval-practice` | Note body |
| `05-stacked-reading-100x30` | 100x30 | Stacked faces, Reading face active, with the header order reversed and the ¶12 cursor | Reader |
| `06-stacked-60x24` | 60x24 | Stacked faces at 60x24, Note face. Header: `‹ Items  Note · paper… (saved 20:12) ⇄ Reading`. The mode strip still shows Save | Note body |
| `07-desk-empty-note-160x45` | 160x45 | After `]` to the next document, a **plain-text (Raw-only) item**. The note slot reads `‹ Items  Note` / `No note yet · n to start one` with a **Start a note** button. Per §5.2 the paragraph cursor works in Raw: `¶1 of 5 · Quote` sits on its own row because a Raw-only item has no Rendered/Raw strip | Reader |
| `08-desk-chooser-160x45` | 160x45 | The `1 of 2 ▴` chooser open. It lists the two linked notes with their ages, the current one marked `▸` | Chooser row |
| `09-note-info-sources-trash-160x45` | 160x45 | The note's Info tab. `Sources (2)` lists one live source (`paper-retrieval-practice ↗ · pdf`) and one in Trash (`Spaced repetition, explained · In Trash · Restore`) | Restore |
| `10-reader-paragraph-cursor-160x45` | 160x45 | The paragraph cursor moved with `j` to ¶13 (`¶13 of 31`). Footer: `n note beside \| q quote \| j/k paragraph \| ]/[ next item \| esc close desk` | Reader |

**Measured pane widths** (cells; unchanged from revision 1):

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
  neutral path fits on one byline row instead of four, so the reader body moves up three rows. Everything left of
  the reader is identical, including the navigation rail, the Items list, the grips, the header and the footer.

## Rulings applied

- **§4.4a Save never drops off: applied.** Below 46 content cells (120x36), the strip drops the `(selected)` word
  and marks the active mode with the Library choice marker (`✓ Edit`).
- **§4.4a `‹ Items`: applied.** It appears on companion header row 1, in the empty-note state, and at the start of
  the stacked-faces header. Without it, the stacked Reading face would have no visible way out. The reader's
  `‹ Back` stays hidden while the desk is open.
- **§5.2 blank-line blocks: no numbering change.** The paper's blocks already matched. State 07 now shows a real
  Raw-only item instead of revision 1's workaround (a Markdown heading added to the fixture).

## Open owner decision

**Links while editing.** As the brief asked, the note body shows `Source: paper-retrieval-practice ↗` and
`— paper-retrieval-practice, ¶12 ↗`. Today's note editor is a plain TextArea and shows the raw Markdown
(`[title](media://uuid)`). Showing `title ↗` in Edit would be a new editor capability. Otherwise the shipped Edit
view shows the raw link and only Preview shows `title ↗`.

## Mockup choices the spec does not state yet (please confirm)

1. **Same focus border in both panes** (§7, review S-14). When the reader is focused it uses the heavy blue edge the
   note body uses. Today's reader uses an amber border (`$accent`).
2. **Wide-pane mode strip** uses the reader's `(selected)` wording. The compact form is described under Rulings
   applied.
3. **Header row 1 below 46 content cells** drops the `Note · ` prefix, because the grip already says `Note`.
4. **Paragraph cursor** is the app's row-focus edge: a thick `█` in the focus colour when the reader has focus,
   muted when the note has focus.
5. **Raw view gutter.** In Raw the paragraph cursor needs a two-cell gutter, so text there starts two cells further
   right than in today's Raw view (07).
6. **Stacked layout.** The list region's grip is hidden, and the inactive face is a button that flips to it. At
   60x24 the active face shortens its title (`Note · paper…`) and the inactive face shows only `Reading`.

## Known render artifacts

These come from the renderer, not from the app:

- Chromium's font draws `⇄`, `↗` and `✓` small or as look-alikes.
- Chromium's font squeezes the CJK row (`学習メモ`) in `00`.
- Heavy box lines (`┏━┓`) are drawn as thin lines.
