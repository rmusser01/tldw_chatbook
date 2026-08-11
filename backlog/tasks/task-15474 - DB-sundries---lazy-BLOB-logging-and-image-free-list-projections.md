---
id: TASK-15474
title: DB sundries: lazy BLOB logging and image-free list projections
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two small verified DB items from the audit. (1) Logging regression: `DB/ChaChaNotes_DB.py:5916` `logger.debug(f"Params: {final_params}")` is a plain eager f-string and `image` is in the updatable fields — saving a character card with an image builds a multi-MB repr of the raw BLOB on every update regardless of log level. This is exactly the pattern `DB/sql_logging.py` exists to prevent; the correct lazy form is 3k lines up in the same file (`:3014-3019`). Smaller eager stragglers: `DB/Client_Media_DB_v2.py:2184/:2245/:5819`. (2) Query shape: `character_cards.image` is a BLOB, and `list_character_cards` (`:5626`) / `list_character_cards_page` (`:5713/:5719`) select `*` — so the Console character picker deserializes up to 500 images to build a NAME list (`chat_screen.py:9000`), and personas paging/evals pickers drag the same BLOBs.

Fix direction: `logger.opt(lazy=True)` + `preview_params` for all four sites; explicit column projections excluding `image` for list/picker paths (detail views keep the full row). Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No eager params stringification remains in ChaChaNotes or Client_Media execute paths (grep + unit evidence)
- [ ] #2 Character list/picker paths fetch no image column (evidence); card save/list behavior unchanged (tests)
<!-- AC:END -->
