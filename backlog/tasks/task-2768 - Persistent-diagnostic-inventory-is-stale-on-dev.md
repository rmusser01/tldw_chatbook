---
id: TASK-2768
title: Persistent-diagnostic inventory is stale on dev
status: Done
assignee: []
created_date: '2026-08-07 06:42'
updated_date: '2026-08-07 19:45'
labels:
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
scripts/check_persistent_diagnostic_inventory.py fails on clean dev (verified at 22c08f958), so Tests/Architecture/test_persistent_diagnostic_inventory.py is red there. The regeneration diff is ~199 insertions spanning new diagnostic owners in Agents/local_tool_provider.py and Chat/prompt_history.py plus six UI/Console_Modules entries that the decomposition waves moved. Wave 3 deliberately did NOT run --write: the checker prints 'review the diff before running --write' because this is a security artifact, and signing off on two unrelated modules' new diagnostic owners is exactly the rubber-stamp that gate exists to prevent. Someone who owns those diagnostics should review and regenerate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each new diagnostic owner in the regeneration diff has been reviewed by someone who can vouch for it
- [x] #2 The inventory is regenerated and Tests/Architecture/test_persistent_diagnostic_inventory.py passes on dev
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Regenerated after reviewing the diff, which is what the checker's "review the
diff before running --write" gate asks for. The delta reconciles exactly:
`chat_screen.py` 145 -> 142 (wave-4 extractions), `Console_Modules/agent.py`
0 -> 2 (where they landed), `Console_Modules/workspace.py` 24 -> 25 (a routed
button branch), `watchlists_collections_screen.py` 75 -> 76 (not mine). Net +1
across 7,986 diagnostics; my waves are net zero.

The one genuinely new diagnostic is `"Watchlist tree write could not start."`
from `d625b9429` -- a static string with no interpolated data, using
`opt(exception=True)` rather than `diagnose=True`, which is the variant that
dumps frame locals and has leaked an API key in this repo before.
`persistent_sink_topology` is byte-identical: no new disk sink.
<!-- SECTION:NOTES:END -->
