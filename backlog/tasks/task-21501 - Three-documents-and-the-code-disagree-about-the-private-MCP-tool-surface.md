---
id: TASK-21501
title: Three documents and the code disagree about the private MCP tool surface
status: Done
assignee: []
created_date: ''
updated_date: '2026-08-24 05:11'
labels:
  - testing
  - test-integrity
  - documentation
  - mcp
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The contract test that exists to stop the MCP documentation drifting from the code had itself
drifted, and while it was red nobody could see what it was hiding.

The set of Library tools withheld from the standalone server is declared once in the code, in
a descriptor table the local capability manifest is generated from — the manifest builder's
own docstring says the table is never to be hand-maintained. Three documents restate that set
for readers, and the contract test restates it a fourth time in order to check them.

Every restatement had fallen behind, and by different amounts, so the three documents and the
test disagreed with the code and with each other. The failure the suite reported was the one
document that disagreed with the test; the two that agreed with the test were equally wrong
and passed. Most of the reported failures were the contract's own self-checks, which cannot
demonstrate that mutations are rejected while the text they mutate is already non-conforming.

Worth stating separately, because it is the part that matters to a user rather than to the
suite: the tool missing from all three documents is the only **write** tool in the set. The
page that enumerates what the private surface contains did not mention it at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The inventory the contract checks against is obtained from the code that defines it, not restated
- [x] #2 Every document the contract governs agrees with the code, and the count each one states matches the list it prints
- [x] #3 The code's own prose description of the surface is corrected where it is stale
- [x] #4 Any tool absent from the documents is identified and reported rather than quietly added, and its significance is stated
- [x] #5 The contract is shown to reject a document that falls behind the code again
- [x] #6 Prose assertions in this module cannot be switched off by reflowing a paragraph
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish what the code actually declares, from the table the manifest is generated from.
2. Compare all four restatements against it and record how each differs.
3. Replace the test's restatement with a derivation.
4. Correct the documents and the code's stale prose from the derived truth.
5. Mutation-prove both the doc-versus-code link and the prose assertions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The withheld-from-standalone Library tool set is declared once, in
`LIBRARY_TOOL_DESCRIPTORS` — the table `MCP/server.py`'s `_describe_local_library_tools`
iterates **unfiltered**, whose docstring states it is "never hand-maintained here". Four
restatements existed and all four were behind, by different amounts:

| restatement | count |
|---|---|
| code (`LIBRARY_TOOL_DESCRIPTORS`) | **24** |
| `Docs/User_Guide/mcp.md` | 23 |
| `Docs/Design/MCP.md` | 18 |
| `Docs/Development/release-recovery-setup.md` | 18 |
| this test's `PRIVATE_LIBRARY_TOOLS` | 18 |

**Why only part of it was visible.** The test agreed with two of the three documents, so only
`mcp.md`'s parametrizations failed; the two documents that matched the test were equally
wrong and passed. Eleven of the fourteen failures were the contract's own mutation
self-checks, which cannot demonstrate that extra/missing/duplicate entries are rejected while
the text they mutate is already non-conforming. A red contract test was concealing two further
wrong documents.

**AC#1** — `PRIVATE_LIBRARY_TOOLS = tuple(LIBRARY_TOOL_DESCRIPTORS)`. Derived, not restated,
so the drift class is removed rather than reset by six. Kept a tuple because
`_mutate_inventory` indexes element 0.

**AC#4 — the finding, stated rather than absorbed.** The tool missing from **all three**
documents is `library_save_note`: the only **write** tool in the set, carrying its own
`library.notes/save` policy. It was added by `6e04d3199`, after `b8592cfa4` wrote
`server.py`'s "23", and `grep` found it **nowhere** in `Docs/User_Guide/mcp.md`. A page whose
job is to enumerate what the private surface contains omitted its only write capability. Had
this been closed by editing the test's 18 up to 23 to match `mcp.md`, that omission would
have survived and the suite would have gone green over it.

**AC#3** — `server.py`'s own prose said 23; corrected to 24 and it now names the write tool
explicitly rather than leaving it inside an "and siblings" gloss.

**AC#6** — `test_expanded_local_tool_group_copy_names_watchlists_everywhere` read **raw** file
text while every neighbouring prose assertion in the module normalizes whitespace. It reported
the three-group label missing from `agent-runs-and-tools.md` when the label is present and
correct, merely wrapped across a newline. The half that matters is the *stale*-label check:
demonstrated directly that with raw text the retired label "Local workspace + web tools",
reintroduced wrapped, is **not** detected (`False`) and is detected once normalized (`True`).
Reflowing a paragraph could switch that guard off.

**Evidence.** 69 passed / 0 failed, up from 55 passed / 14 failed — and now checking 24
entries across three documents instead of 18. Mutation-proven three ways: removing one tool
from `Docs/Design/MCP.md` → 13 failed; making a document's stated count disagree with its own
list → 13 failed; reintroducing the retired label wrapped → 1 failed. Restored: 69 passed.
Regression sweep `Tests/MCP` + `Tests/QA` + `Tests/CI` + `test_library_tool_contract.py`:
**1,431 passed, 1 failed**, and that one
(`test_console_notes_workspace_uat.py::test_console_agent_reads_then_updates_configured_workspace_note`)
fails identically on a clean `dev` checkout.

Modified: `Tests/MCP/test_mcp_documentation_contract.py`, `Docs/User_Guide/mcp.md`,
`Docs/Design/MCP.md`, `Docs/Development/release-recovery-setup.md`,
`tldw_chatbook/MCP/server.py` (docstring only).
<!-- SECTION:NOTES:END -->
