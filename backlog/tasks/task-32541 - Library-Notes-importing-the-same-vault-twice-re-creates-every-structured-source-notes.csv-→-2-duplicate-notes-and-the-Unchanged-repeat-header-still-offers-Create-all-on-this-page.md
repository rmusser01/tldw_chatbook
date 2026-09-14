---
id: TASK-32541
title: >-
  Library Notes: importing the same vault twice re-creates every structured
  source (notes.csv → 2 duplicate notes), and the Unchanged repeat header still
  offers "Create all on this page"
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:46'
updated_date: '2026-09-14 14:45'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, personas Riley and Jordan, Obsidian import workflow. A P2 #7 / B D5.

**What happened.** Second Import once of the unchanged vault → folder-collision block with "Create a unique sibling" pre-selected ("vault (2)") → every `.md` is "Unchanged repeat · Content: no change · Folder placement: no change" with Skip pre-selected — but `notes.csv` is under New again: "create 2 new notes: CSV note one, CSV note two · keywords csv · Create in vault (2)" (A 61; B 36). `meta.yaml` and `scratch.txt` follow the same path. The Unchanged repeat group's header also carries "Create all on this page", which would re-create 56 notes in one click (A 61). Captures: A 61; B 36.

**Cause.** INFERRED: structured sources (CSV rows, YAML) are not fingerprinted per produced record, so repeat detection — which works for Markdown — never classifies them. No test in Tests/Notes or Tests/UI names CSV or structured repeats (B grep). Task-32130 / 32176 fixed the first-import CSV count only. Docs contradicted: notes.md's Unchanged repeat rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A second Import once of an unchanged vault lists notes.csv, meta.yaml and scratch.txt under Unchanged repeat with Skip pre-selected
- [x] #2 The Unchanged repeat group header offers no "Create all" action
- [x] #3 A test imports a CSV source twice through the review and asserts the second review classifies it as an unchanged repeat
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce: import the vault twice on a fresh profile; capture notes.csv/meta.yaml/scratch.txt under New and the Unchanged-repeat Create-all button.
2. Trace: receipts turn a row into a PriorImportObservation only when outcome_count == 1 and payload_count == 1; the planner degrades any multi-payload source to UNCERTAIN.
3. RED tests: planner (two-row CSV twice -> UNCHANGED_REPEAT, default SKIP), receipts (multi-payload source -> one source-level observation), canvas (Unchanged-repeat header has no Create-all).
4. Fix: source-level fingerprint derived from the per-payload digests the ledger already stores (no schema change); planner compares it when the payload counts agree, keeps UNCERTAIN on a count mismatch; canvas drops Create-all for unchanged_repeat.
5. GREEN, live second import, guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A source that parses into several notes left no prior observation behind, so
the planner saw `notes.csv`, `meta.yaml` and `scratch.txt` as new on every
import: the receipt reader turned a ledger row into a `PriorImportObservation`
only when `outcome_count == 1 and payload_count == 1`, and the planner
degraded any multi-payload source to `UNCERTAIN`.

Receipts now read **every** APPLIED payload row of the latest item for a
source and emit ONE source-level observation carrying `payload_count` and a
fingerprint combined from the per-record digests the ledger has always
stored — no schema change, and receipts written before this fix are
recognised (proved live: the ledger holds one completed session, written
before the fix, and a later review classified all three structured sources
from it). The planner compares that fingerprint when the source still parses
into the same number of records and keeps the `UNCERTAIN` degrade when the
count moved. A multi-record repeat offers **Skip** and **Create new** only —
"Update authorization requires one payload" already forbade the third action,
so the classification contract no longer demands it. The **Unchanged repeat**
group header drops **Create all on this page**: there is nothing to create,
and one press re-created 56 notes.

Trade-off: the fingerprint is rebuilt from stored digests rather than added
as a new ledger column, which keeps the device-state schema version fixed and
makes old receipts count.

Live (235x52): a repeat **Import once** of the 66-source vault read
"Unchanged repeat (58)" with `vault/meta.yaml`, `vault/notes.csv` and
`vault/scratch.txt` on it, each "Content: no change · Folder placement: no
change" with **Skip** pre-selected, the header carrying only **Skip all on
this page**, and the multi-record rows offering **Skip** / **Create new**
(`wave4-caps/data-truth/data-25-third-import-unchanged-repeat`).

AC#1 honesty note: in the seeded verification vault only `notes.csv` is
multi-record; `meta.yaml` and `scratch.txt` parse to a single payload there
(the capture shows them offering **Update existing**, which only a one-payload
repeat may offer), and the old code already classified single-payload sources.
So the live capture proves the CSV case for this fix; the other two rows are a
negative control, not evidence. The critique's vault had multi-record YAML/text
sources, and the mechanism is general in `payload_count` — the headless pins
(`Tests/Notes/test_note_import_planner.py`,
`Tests/Notes/test_note_import_receipts.py`) carry the multi-payload contract.

Files: `tldw_chatbook/Notes/note_import_receipts.py`,
`tldw_chatbook/Notes/note_import_planner.py`,
`tldw_chatbook/Notes/note_import_plan_models.py`,
`tldw_chatbook/Widgets/Library/library_note_import_canvas.py`,
`Tests/Notes/test_note_import_{planner,receipts,executor}.py`,
`Tests/UI/test_library_notes_wave_import_ux.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
