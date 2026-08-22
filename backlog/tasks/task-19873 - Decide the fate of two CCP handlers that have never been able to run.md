---
id: TASK-19873
title: >-
  Decide the fate of two CCP handlers that have never been able to run
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - dead-code
  - owner-decision
  - personas
priority: medium
dependencies:
  - TASK-19559
  - TASK-19563
---

## Description

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

## Acceptance Criteria

- [ ] An explicit decision is recorded for `CCPConversationHandler` and
      `CCPDictionaryHandler`: delete, or wire up and fix
- [ ] Whichever is chosen is carried out completely — if deleted, the exports in
      `UI/CCP_Modules/__init__.py` go with them; if wired up, each restored
      path has a test that would have caught the original `TypeError` and
      `AssertionError`
- [ ] The five `UI/Tools_Settings_Window.py` dispatches get the same explicit
      decision, taken together with the standing question of whether that
      nav-unreachable surface survives at all (TASK-3240)
- [ ] Nothing is left in a state where a `run_worker` call is known-broken and
      merely unreachable
- [ ] The evidence that these paths never ran is preserved in the
      implementation notes, so a future reader does not mistake the deletion
      for a feature removal

## Notes

Recorded because it corrects a claim made in flight: TASK-19559 reported the
CCP conversation-search fix as a headline find. It is a correct fix to dead
code. The only genuinely live CCP defect that task touched is the character
load, which it rated as the lesser of the two.
