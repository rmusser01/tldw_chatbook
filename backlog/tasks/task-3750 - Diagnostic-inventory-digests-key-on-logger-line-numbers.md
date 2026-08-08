---
id: TASK-3750
title: Diagnostic inventory digests key on logger line numbers
status: Done
assignee:
  - '@claude'
created_date: '2026-08-08 21:06'
updated_date: '2026-08-08 23:14'
labels:
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Docs/security/production-diagnostic-inventory.json stores a per-file digest that changes when logger call LINE NUMBERS move, not only when the diagnostics themselves change. Any refactor that shifts lines in a file containing logging therefore fails Tests/Architecture/test_persistent_diagnostic_inventory.py and needs a review-and-regenerate cycle -- this cost one cycle per task across decomposition waves 4 and 5, every time with call_count unchanged and the sink topology byte-identical. Keying the digest on diagnostic CONTENT (message, level, owner) rather than position would keep the security signal the file exists for while removing a per-refactor chore that trains people to regenerate it without reading the diff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Moving a logger call within a file without changing it does not change its digest
- [x] #2 Adding, removing, or editing a diagnostic still changes the digest and fails the test
- [x] #3 The existing reviewed inventory is migrated to the new keying in one deliberate commit
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the script and identify exactly what feeds diagnostic_digest
2. Re-key the digest on content (log method + statement source text), dropping line numbers
3. Decide the sink question on evidence, not taste
4. Write unit tests for AC1/AC2 incl. multiplicity, plus a real-file end-to-end
5. Migrate the checked-in inventory in a separate deliberate commit
6. Prove both arms by mutating real production source
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-keyed the per-file diagnostic_digest on CONTENT instead of position.

**Approach.** The digest hashed a list of {line, method, digest} per logger call, so
line numbers were a hash input and any refactor that shifted lines failed the gate.
It is now a sorted list of (log method, statement source-text digest). Sorted, so
reordering or relocating calls is invisible; a LIST and not a set, so multiplicity
counts and deleting one of two identical calls is still caught.

**Sinks: line: dropped, replaced by scope:.** Decided on evidence found while doing
this task, not on taste. The entire sink drift sitting on dev was pure line movement
-- all 5 drifted entries had byte-identical digest and kind, with only line: changed
(app.py 6063 -> 6656). Sinks live in app.py and config.py, two of the churniest files
in the repo, so the same false-positive engine runs there. Entries now carry the
enclosing qualified scope (TldwCli._setup_buffered_logging rather than 6063), which
navigates better than a line number and does not go stale. The rejected middle option
-- print line: but exclude it from the comparison -- would let the number rot and send
a reviewer to the wrong place, which is worse than a chore.

**Diagnostics deliberately do NOT carry scope,** though sinks do. Moving a logger call
into an extracted method is the single most common refactor in this repo (decomposition
waves 4/5) and must stay silent; sinks are 19 entries that essentially never move
between functions, and there the scope is the only navigation handle a reviewer has.
Owner entries never had a line number to lose -- they are only path/count/digest.

**schema_version 1 -> 2.** Every digest changes because the keying changed, not because
code did; the version is the only signal that a mismatch across the bump is a semantics
change rather than a new diagnostic.

**Sensitivity is unchanged.** Proven on real production source, six arms, each run
against the real checker: a moved logger call passes (exit 0); reworded message fails;
level debug -> error fails; a moved sink passes; a retargeted sink fails; full restore
passes. Plus unit tests for add/remove/reword/re-level/duplicate-multiplicity.

**The migration commit absorbs pre-existing dev drift, reviewed rather than rubber-
stamped.** The inventory was last regenerated in f990464ed (HEAD) and was ALREADY
stale there -- a branch that regenerates before merge is re-staled by whatever merges
ahead of it. Of 47 drifted owner entries, 28 were pure line movement and 19 were real
content changes (3 files new). The added diagnostics in the high-risk TASK-492 paths
were read: a Library factory warning, an MCP backend-unavailable exception, and 7
robots.txt debug lines in web_tool_impls.py that log cache_key -- which is
scheme://host[:port] only, no path or query, so no token can leak.

**Files:** scripts/check_persistent_diagnostic_inventory.py,
Tests/Architecture/test_persistent_diagnostic_inventory.py,
Docs/security/production-diagnostic-inventory.json (migration in its own commit).
<!-- SECTION:NOTES:END -->
