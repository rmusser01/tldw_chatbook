---
id: TASK-41112
title: "test_actor_pack_export.py's golden-bytes cases [character] and [persona] are red on dev and run on no lane"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-15 20:07'
created_date: 2026-11-15 02:20
labels:
- follow-up
- actor-packs
- tests
- dev-red
priority: low
references:
- Tests/Actor_Packs/test_actor_pack_export.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The test pins exported Actor Pack archives byte for byte against independent golden bytes. Red, it no longer tells anyone whether an export change was intended.

**Evidence.** Through run.sh at the stack top: `test_archive_matches_independent_golden_bytes[character]` and `[persona]` (`:580`) fail with `assert output.getvalue() == golden`, first difference at byte 14 (the first local header's CRC-32, so the first member's content changed). TASK-40304 recorded both red on its base too. No workflow names the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The commit that changed the first member's bytes is named, and the case is classified as intended format drift or an export defect
- [x] #2 Drift: the golden bytes are regenerated from an independent writer with the change explained; defect: the export is fixed with the test red first
- [x] #3 The file runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Find the commit that changed the first member (actor-pack.json) of the golden archive and classify it.
2. If drift: stop the golden depending on the release version, regenerate the goldens with a stdlib-only writer (no product import), and explain the change; if a defect: fix the export with the test red first.
3. Run the file on a PR lane (recovered-suites, sandboxed block) and update the Tests/CI pin.

## Implementation Notes

Commits aae3234008 (test) and 7d9c3988b1 (lane).

- **Cause (AC#1).** Only `actor-pack.json` differed: `producer.version` is `tldw_chatbook.__version__` (`Actor_Packs/export.py`). The goldens were cut at 0.1.8.0 (80e0cbb055); the first commit to move them is 2c893cfc8d (0.1.8.0 -> 0.1.8.1), then every release bump to 0.2.3. Intended format drift, not an export defect: the export is correct, the golden pinned the release.
- **Fix (AC#2).** The golden case pins the writer's version to `GOLDEN_PRODUCER_VERSION` ("0.1.8.0", `Tests/Actor_Packs/conftest.py`); the deterministic case now asserts `producer == {name, __version__}`, so the app-version link is still pinned. `generate.py` was rewritten as an independent writer: stdlib `zipfile` plus the conftest oracle, no `tldw_chatbook` import (checked with `-X importtime`); it reproduces both committed archives byte for byte (sha256 4895990d... / d2ca4192... unchanged), so no binary changed.
- **Red/green.** `pytest Tests/Actor_Packs/test_actor_pack_export.py -n 4`: 2 failed (`[character]`, `[persona]`, first difference at byte 14) at load ~25; after: 17 passed.
- **Lane (AC#3).** recovered-suites (PR lane), sandboxed block: 17 passed in 83 s serially with the lane's flags (`--storage-lease-guard --timeout=180`) at load 104. Pinned in `RECOVERED_SUITES_TARGETS`.
- Files: `Tests/Actor_Packs/{conftest.py,test_actor_pack_export.py,fixtures/export-golden/generate.py}`, workflow, Tests/CI, ADR-103.
