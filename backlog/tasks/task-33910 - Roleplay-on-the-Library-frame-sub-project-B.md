---
id: TASK-33910
title: 'Roleplay on the Library frame (sub-project B)'
status: To Do
assignee: []
created_date: '2026-10-02 18:47'
labels:
  - roleplay
  - ux-review-2026-10-01
  - layout
dependencies: []
references:
  - Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-10-01 Roleplay NN/g + HCI layout review (findings RP-001..RP-097, report `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md`, landing with PR #2957) scored Roleplay 11 of 40 against the Library's 27. At 120x36 Roleplay spends 17 rows on chrome, shows 5 of 28 sample characters, gives the character editor 58 columns, and leaves the test chat unusable. Sub-project A (TASK-33781..TASK-33791 for Roleplay; TASK-33792 and TASK-33793 for Console) fixes the defects first. This task is **sub-project B**: rebuild Roleplay (Ctrl+4) on the Library's frame, a Roleplay navigation rail, a full-height items list and a permanent work pane with views and one action bar, with the right-hand Inspector folded into the work pane. The four jobs (J1 import and chat, J2 author characters, J3 build world info, J4 manage personas and "you") are equally important. The design centre is 120x36 to 160x45 and 200+ columns; 80x24 and below 64 columns only degrade gracefully.

**Spec:** `Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md` (approved 2026-10-02; PR #2960, branch `docs/roleplay-library-layout-ux`). Owner rulings Q1-Q15 are in its §8, design rulings R1-R41 in §0.6.

**Delivery** (spec §5 "Delivery: slices, testing, documentation, governance, risks"): the subtasks are the slices B0, B1, B2a, B2b, B3, B4, B5a, B5b-1, B5b-2, B5c, B6, B7, B8, B9a, B9b, B10, B11 and B12 in §5.1's dependency order, then the fresh-lens review follow-ups FU-1..FU-5 (§5.13 item 2). Each slice gets its own implementation plan under `Docs/superpowers/plans/` (§5.13 item 1) and meets the §5.4 definition of done. Rules that bind the whole programme:
- One `personas_screen.py`-touching slice in review at a time; sub-project A's tasks count against this rule. B0 and B8 touch no `personas_screen.py`, so B8 may be in review beside B9a-B11 once B7 has merged (§5.1).
- B6 and B7 land in one release window: no release is cut between them, because B6 narrows the work pane until B7 (§5.3).
- Interim affordances are tracked in the §5.3 register; each is removed by its named slice.
- Hard prerequisites on sub-project A are listed in §5.5; B5c alone waits for Console's TASK-33621.2 (D13, R29).
- Merge PR #2957 before, or together with, the spec's PR #2960; B0's plan starts only once both are on dev (§5.13 item 3).

**Out of scope** (§0.5): sub-project A's fixes, world-info semantics (sub-project C), persona semantics (sub-project D), Console defects, and the shell and design-system defects listed there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Roleplay's chrome above and below the content is 7 rows (5 + 2) at 120x36, 160x45 and 220x55, and the first list item sits on row 7 (with today's 3-row nav), against 17 rows and row 23 today.
- [ ] #2 With the 28-character sample, 28 / 18 / 23 characters are visible at 120x36 / 160x45 / 220x55 (today 5 / 9 / 14).
- [ ] #3 In a work session the editor has 110 columns at 120x36 and 108 at 160x45; at 220x55 it has 129 with the Try column off, or a 107-column body beside a 60-column Try column by default (Q2). Today it has 58 / 78 / 108.
- [ ] #4 A 250-entry lore book shows 23 / 33 / 43 entry rows at 120x36 / 160x45 / 220x55 (today 4 / 10 / 10).
- [ ] #5 The test chat transcript is full height with at least 20 / 29 / 39 rows at 120x36 / 160x45 / 220x55.
- [ ] #6 The right-hand Inspector column no longer exists; its contents live in the work pane's title row, view strip, More actions and Info.
- [ ] #7 Every slice subtask (B0, B1, B2a, B2b, B3, B4, B5a, B5b-1, B5b-2, B5c, B6, B7, B8, B9a, B9b, B10, B11, B12) is Done, each having met the spec's §5.4 definition of done.
- [ ] #8 The User Guide is consolidated (B12): the Roleplay pages are retitled "Roleplay" with their filenames kept, the drift listed in spec §5.8 is fixed, and every `scripts/check_guide_claim_strings.py` candidate on the four pages is fixed or named as a reviewed exception.
- [ ] #9 The governance changes of spec §5.9 have landed on dev: the new shared-pane-shell ADR (G1-G1e) with its ADR-086 and ADR-084 pointers, and the amendments to ADR-046, ADR-031, ADR-120 (mirrored in ADR-004), ADR-115, ADR-152, ADR-015 and DESIGN.md (G11, G12, G14).
- [ ] #10 Boot parsed CSS is no larger after B12 than at the programme's base, and every slice PR recorded its §5.10 budget-ledger rows.
- [ ] #11 Follow-ups FU-1..FU-5 are each Done or carry a recorded owner decision.
<!-- AC:END -->
