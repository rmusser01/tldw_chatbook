# PR2726 conflict choices and visual review

The owner approved this visual/conflict snapshot. [Current closeout evidence](CLOSEOUT.md) records the inherited selector-budget repair, 203 passing targeted checks and pixel-identical modal/Audit replays. Earlier pending-approval and unchanged-source statements below describe the approved snapshot; final-head CI/review still gates merge.

The saved PR at `9684e7a98f93abbb82f95446e0c765a0780f70e2` was rebased onto
merged dev `cccf0acdad8e939a55cb003588ff1406cef5d1f4` (PR2724). The rebase
produced local commit `4c4b4176c3`; the follow-up preserves profile checks and
refreshes tests and native QA on this combined source.

## Conflict choices

| Conflicted file | Combined result |
| --- | --- |
| `mcp_workbench.py` | Kept dev's publication lock through row selection and rendering, boolean row-selection failure handling and warning/clearing behavior, and post-selection profile validation. Added PR2726's profile validation after lock acquisition and current same-ID lookup; row selection, effective permissions, cascade and argument rules all use that current definition. |
| Completion audit | Retained merged revocation, rule-action, permission-navigation and PR2724 closeout history, then retained PR2726's original catalog-freshness notes. Added a current checkpoint that marks older pending statements historical. |
| MCP ledger | Retained both histories and added the current merged/draft status and remaining scope. |

No wholesale application-side selection discarded the other branch's behavior.
PR2724's retired/unavailable-control ownership checks in the inspector remain
unchanged. No CSS or token values changed. The refreshed native bootstrap and
two extra profile-switch cases were deliberate follow-up edits, not conflict
choices. The profile-switch negative control demonstrates why the retained
post-selection validation is necessary.

## Reviewable evidence

[168 targeted cases, seven guards and independent review](README.md) pass.
[Eight current captures](GALLERY.md) show the refreshed identity in both Tools
and Permissions at compact/wide sizes in both themes. Tools also displays the
new description. Definition/schema/availability and removed-target behavior
are checked independently of the screenshot.

Please include the [first-run header anomaly](HEADER-FOLLOWUP.md) in visual
review. The unchanged-source replay has aligned headers; this intermittent
rendering issue remains a separate follow-up and is not claimed fixed here.
Current-head CI/review and this PR's own final visual approval gate merging.

## PR2724 closeout receipt

PR2724 merged at `cccf0acdad` after owner approval, final-head CI, Qodo's zero
remaining findings and resolution of all five accumulated threads. The merge
also incorporated concurrent PR2742 wide-modal changes, so its tree differs
from the reviewed PR head. All MCP modules were unchanged. On the actual merged
state, 32 Audit tests passed and all 24 native terminal captures matched the
approved views except synthetic timestamps; lifecycle checks passed.
[Merge receipt](current-dev/pr2724-closeout.json), [comparison](current-dev/pr2724-merged-visual-comparison.json)
and [lifecycle](current-dev/pr2724-merged-lifecycle.json) preserve that distinction.
