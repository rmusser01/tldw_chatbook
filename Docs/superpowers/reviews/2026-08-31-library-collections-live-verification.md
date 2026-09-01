# Library Collections Capture Reader Live Verification

**Task:** TASK-18919

**Date:** 2026-09-01

**Branch:** `codex/task-18919-collections-reader`

**ADR:** `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`

## Outcome

- **Local authority: PASS.** The production-shaped mounted walkthrough used a disposable test
  profile/data root and a real on-disk Collections database seeded with 45 captures. Its resolved
  database-path fingerprint matched before and after the walkthrough; no private path is retained
  in this record.
- **Server authority: BLOCKED.** The configured Server profile and credential were present, but the
  docs-info prerequisite was unreachable with `APIConnectionError`. No Server data was read or
  mutated, no capability bypass was attempted, and `hasReadingSnapshotPagesV1: true` could not be
  attested. TASK-18919 therefore remains **In Progress**.

## Local walkthrough evidence

The live test `Tests/Live/test_library_collections_capture_walkthrough.py` exercises the actual
Library route, production stylesheet, capture repository/service, and mounted Textual widgets.
Condition-based waits are used for asynchronous state changes.

| Terminal | Pinned pure geometry (Library / Items / Work) | Mounted result |
| --- | --- | --- |
| 160×50 | 30 / 40 / 80 | PASS: all panes, both grips, Work focus via F6, no descendant overflow |
| 120×35 | 0 / 56 / 54 | PASS: Items reclaims Library width, Work remains mounted, no overflow |
| 100×30 | 0 / 42 / 48 | PASS: compact Items and Work, keyboard focus and containment preserved |
| 80×24 | 0 / 0 / 70 | PASS: Work-only posture, both grips remain accounted for, no overflow |

The run also verified:

- untouched Library startup does not pre-mount the capture reader;
- route activation, every Library/Items collapse combination, wide resize restoration, and F6 Work
  traversal;
- exact pages of 20, 20, and 5 rows from a coherent total of 45;
- closing Library expands Items so longer titles receive the reclaimed width;
- Quick Capture commits the URL, tags, and freeform note before extraction completes;
- a controlled extraction failure persists as `failed`, same-item re-selection refreshes it, and an
  explicit Retry reaches `ready` with authoritative recovered text;
- Read, Highlights, Notes, and Info modes;
- archive and Undo, offline-copy cleanup through confirmed hard delete, and complete recovery export
  of all 45 legacy Collections rows plus all 45 memberships;
- simulated unknown Server-save UI behavior: the draft remains visible, no automatic retry occurs,
  the user is told to refresh first, and only the explicit warning confirmation issues one retry.

## Findings corrected during the walkthrough

1. The Items toolbar could overflow a compact pane even while the shell's child widths summed
   correctly. Sort now owns a separate toolbar row, and the live assertion checks every visible
   descendant against its pane.
2. The Work action toolbar could similarly push Open Original outside the pane. Primary and
   secondary actions now use separate rows.
3. Re-selecting a capture whose background extraction changed did not refresh the loaded detail.
   Same-item selection and Retry now reload authoritative detail.
4. A late summary/audio/offline result could paint onto a newly selected capture. Content,
   highlight, capture-note, and linked-Note completions now verify identity after awaits.
5. An indeterminate Server save could lose its draft or invite an accidental retry. The reader now
   retains the draft, exposes Refresh first, and requires a second explicit confirmation with the
   current Server default-reapplication warning.

## Automated evidence

- Focused controller, mounted reader, and Local live walkthrough: **29 passed**.
- Complete capture feature and Local live gate: **206 passed**.
- Production-shaped cross-reader closeout after the final save-recovery hardening: **490 passed**.
- Static validation covers the edited Python modules, CSS regeneration, bytecode compilation, and
  whitespace/error checks.

## Remaining completion gate

Run the enabled-Server walkthrough only when docs-info is reachable and advertises exact
`hasReadingSnapshotPagesV1: true`. Then seed/use an isolated principal with more than 40 captures,
repeat all four sizes and pages 1–3, verify authoritative capture content, capability reasons,
archive/Undo, source replacement, workspace non-effect, confirmed and controlled-unknown saves,
and switch back to Local. Until that succeeds, acceptance criteria 4, 9, and 10 are not closed and
the task must not be marked Done.
