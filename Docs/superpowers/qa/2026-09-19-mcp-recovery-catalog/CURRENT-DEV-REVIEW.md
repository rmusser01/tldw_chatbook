# PR2728 integrated catalog and guidance review

PR2727 merged at `ebee42fab8a7f55def6a03fc7b3301935940eebf` after its owner
visual approval, current-head CI and accumulated review. PR2751 landed during
that merge, changing eight files outside MCP. The actual merged app passed all
85 affected tests and a fresh 16-capture native journey, with normal shutdown
and unchanged user defaults. MCP code and styles match the approved parent.
[Actual merge receipt](integration/pr2727-closeout.json).

This existing follow-up is rebased onto that verified merged state. The original
draft remains preserved on `codex/mcp-recovery-catalog-saved-dcdeb2`. Its final
product scope is unchanged: repopulate the reviewed local catalog through the
existing passive reader, publish only while the originating view owns the
result, preserve successful approval if the read fails, and clarify that
historical rules and grants remain inactive.

## Conflict choices

- **MCP review ledger:** retain every dev line in order and the complete saved
  catalog checkpoint. A new top checkpoint identifies the current state.
- **Diagnostic inventory:** start from dev, review the sole new warning through
  the existing bounded secret/path redactor, and regenerate from the combined
  source. The correct aggregate is 7,761 calls, not either stale conflict count.
- **Product code:** no conflicts. The merged same-mode Permissions row/profile
  invalidation, permission navigation and Tools header repairs remain intact.

[Resolution record](integration/rebase-conflicts.json) and
[diagnostic delta](integration/diagnostic-review.txt) preserve the exact choices.

[Qodo findings and dispositions](QODO-REVIEW.md): isolated coverage added;
the reported malformed environment mapping is already normalized by the existing
storage model. Real store/service regressions verify both profiles survive.
The app and runner are unchanged.

## Current qualification

- [182 distinct targeted cases](integration/qualified-cases.json) pass:
  91 product, parent and adjacent cases, 77 native-runner boundary cases,
  and 14 isolated/storage cases added after Qodo review.
  Every one of the original draft's 49 selected cases is covered again.
  [Product selection](integration/selected-cases.txt); no full suite ran.
- All [seven derived-artifact guards](integration/preflight.txt) pass.
  [Static analysis](integration/static-analysis.json) adds no diagnostics;
  [changed ranges and new files are formatted](integration/formatting.txt).
- [Independent review](integration/independent-review.txt) found no blockers in
  the product integration or the runner follow-up.
- The saved runner's seven malformed-input cases all failed before reusing the
  shared CLI boundary. [Red](integration/tests/runner-red-001.txt) and
  [77 green cases](integration/tests/runner-green-001.txt) verify rejection before
  environment changes, application imports, restore setup or output writes.
  The app's supported image warm-up replaces the obsolete private import.
- [Eight current captures](GALLERY.md), all rendered and individually inspected,
  cover dark/light at 120×40 and 170×48. Native keyboard cancellation preserves
  activation; confirmation creates fresh Ask/local defaults and retains history.
  The repopulated `demo` row opens the disconnected server detail. No connection,
  discovery, execution or permission grant occurs. The first cell activates the
  generation; later cells repeat review of that same unchanged generation.
- [Native lifecycle](integration/lifecycle.json): App.run returns normally,
  exit 0, process absent, instance lock released, ten healthy private databases,
  zero conversations/messages and three unchanged user-default fingerprints.
  All source and runner hashes match current files. The known unrelated restored
  Evals `storage_scope_not_enrolled` diagnostic remains the only app error;
  the faulthandler log is empty.

[Source verification](integration/verification.json) and
[export hashes](integration/export-manifest.json) identify this evidence.
The original packet and [historical gallery](HISTORICAL-GALLERY.md) remain as
evidence for their older source, not as current qualification.

ADR required: no. Existing ADR-126 and ADR-150/161 apply; no ownership, storage,
service, token or stylesheet boundary changes. Long-path review presentation,
80×24, inspector refresh, connected-runtime journeys and saved Console approval
draft PR2730 remain separate follow-ups. PR2707's heartbeat remains paused.

PR2728 requires its own current-head CI, accumulated review and final owner visual
approval before merge. PR2727's approval does not authorize merging this PR.
