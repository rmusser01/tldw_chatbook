---
id: TASK-19873
title: Decide the fate of two CCP handlers that have never been able to run
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-29 16:39'
labels:
  - dead-code
  - owner-decision
  - personas
dependencies:
  - TASK-19559
  - TASK-19563
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: **TASK-19559**'s reviewer, correcting that task's blast-radius claim.
Re-verified at `3605bd52d`.

`CCPConversationHandler` (`UI/CCP_Modules/ccp_conversation_handler.py:39`) and
`CCPDictionaryHandler` (`ccp_dictionary_handler.py:16`) are exported from
`UI/CCP_Modules/__init__.py` and **never constructed in production**.
`PersonasScreen` builds only `character_handler` and `persona_handler`
(`UI/Screens/personas_screen.py:1026-1027`). Nothing else instantiates either
class.

That they are dead is not a guess — it is provable from the code, because
neither could ever have executed. Both contain the same two independent
`run_worker` defects, either of which raises on the first call:

1. `self.window.run_worker(self._search_conversations_sync, search_term,
   search_type, thread=True, exclusive=True, name="conversation_search")`
   (`ccp_conversation_handler.py:111-118`; same shape at
   `ccp_dictionary_handler.py:79-85`). The extra positional arguments bind onto
   `run_worker`'s **own** `name` and `group` parameters, not onto the callable
   → `TypeError`.
2. The target methods carry `@work(thread=True)`
   (`ccp_conversation_handler.py:120`, `:276`;
   `ccp_dictionary_handler.py:87`, `:430`, `:459`), whose decorator asserts
   `isinstance(self, DOMNode)` — and these handlers are plain objects, not
   widgets → `AssertionError`.

So conversation search, dictionary load, conversation refresh and conversation
load in these two handlers have never worked and cannot have. This is the
corpse of a feature, not a regression.

Five further dispatches of the same shape sit in
`UI/Tools_Settings_Window.py` (`:6685`, `:6747`, `:6894`, `:7000`, `:7325`).
That surface is nav-unreachable: `UI/Navigation/screen_registry.py:122-123`
routes `tools_settings` to `MCPScreen`, so the window is never mounted.

**This wants a decision, not a silent repair.** Quietly fixing the `run_worker`
misuse would resurrect four code paths that no one has exercised, tested, or
reviewed for correctness against the current schema and screen structure — a
larger and riskier change than it looks, dressed as a bug fix. Deleting them is
equally a decision: it removes the last trace of whatever these were meant to
become. The corpse trap runs in both directions, and the owner should choose.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An explicit decision is recorded for `CCPConversationHandler` and
      `CCPDictionaryHandler`: delete, or wire up and fix
- [x] #2 Whichever is chosen is carried out completely — if deleted, the exports in
      `UI/CCP_Modules/__init__.py` go with them; if wired up, each restored
      path has a test that would have caught the original `TypeError` and
      `AssertionError`
- [x] #3 The five `UI/Tools_Settings_Window.py` dispatches get the same explicit
      decision, taken together with the standing question of whether that
      nav-unreachable surface survives at all (TASK-3240)
- [x] #4 Nothing is left in a state where a `run_worker` call is known-broken and
      merely unreachable
- [x] #5 The evidence that these paths never ran is preserved in the
      implementation notes, so a future reader does not mistake the deletion
      for a feature removal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add positive retirement guards, then delete the two unconstructed CCP handler modules and exports. 2. Replace the deprecated Tools Settings UI contract, then remove the five broken operation families and orphan helpers. 3. Remove orphan private-SQLite owner policies while preserving generic seam coverage with retained owners. 4. Correct current architecture documentation and regenerate only affected diagnostic and pre-import artifacts. 5. Run the focused tests, Ruff, inventory, boot-budget, reference, and diff checks; self-review and record closeout evidence. ADR required: no. ADR path: N/A. Reason: dead-code removal enforcing existing navigation and ownership boundaries.
<!-- SECTION:PLAN:END -->

## Notes

Recorded because it corrects a claim made in flight: TASK-19559 reported the
CCP conversation-search fix as a headline find. It is a correct fix to dead
code. The only genuinely live CCP defect that task touched is the character
load, which it rated as the lesser of the two.

## Implementation Notes

- Decision: deleted the unconstructed `CCPConversationHandler` and
  `CCPDictionaryHandler`, their package exports, and the five unrouted/broken
  Tools Settings operation families (individual vacuum, backup, restore,
  integrity check, and legacy Chatbook import). The rest of Tools Settings was
  retained.
- Evidence preserved: `PersonasScreen` constructed only the character and
  persona handlers. TASK-19563 repaired only dead dispatch spelling, and no
  production construction or routing path for the deleted handlers or
  operations existed through deletion.
- Retained boundaries: live character/persona behavior, canonical Chatbooks
  workflows, bulk database maintenance, and the shared private-SQLite seam.
  Orphan Settings SQLite owner policies were removed while generic backup and
  restore tests were retargeted to retained owners.
- Current architecture docs and the affected diagnostic/pre-import generated
  artifacts were updated. Canonical writers absorbed reviewed deletion effects
  as well as pre-existing stale upstream diagnostic/snapshot drift. The known
  invalid baseline dangerous-restore test was removed with the retired
  single-restore contract.
- Focused final gate: 438 passed, 1 skipped, 7 summary warnings in 364.79s,
  exit 0. The separate pre-import gate passed 1 test with 2 warnings, exit 0;
  capacity remained non-blocking at 491/500 modules and 379,358/380,000 LOC.
  The diagnostic inventory verified 540 owners, 1,270 TASK-492 calls, 7,325
  TASK-494 calls, and 8 sink files, exit 0.
- Static evidence: aggregate Ruff check reported only the pre-existing E402 at
  `Tests/UI/test_tools_settings_window.py:430`; aggregate format-check reported
  only the three pre-existing drift files
  (`Tools_Settings_Window.py`, `test_ccp_handlers.py`, and
  `test_tools_settings_window.py`). Direct `origin/dev` checks reproduced the
  same E402 (line 429 there) and all three format failures. The other eight
  modified Python files passed Ruff check and format-check, and all three
  baseline-drift files passed `ruff check --select F`, all exit 0. Ruff was
  therefore not claimed globally clean.
- Reference/diff evidence: all four narrow current-owner/current-doc/generated
  retirement searches produced no matches; deleted names remain only in
  positive absence assertions. Both `git diff --check` and
  `git diff origin/dev...HEAD --check` exited 0. Stat, name-status, production
  hunk, and full-boundary review found no unrelated production removal or
  formatting/historical-document churn. No stray `task-task- - .md` exists.
  The full repository suite was intentionally out of scope.
- Plan corrections: review hardened the export/dispatcher absence guards and
  pruned stale claims from current docs. No implementation scope was added.
- ADR required: no. ADR path: N/A. Reason: this dead-code removal enforces
  existing navigation and ownership boundaries without changing storage,
  service, security, runtime, or cross-module architecture.
