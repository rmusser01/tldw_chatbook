# Restored MCP review completion — TASK-32839

An accepted review cleared its view token before the native write finished.
Its later receipt could reset a newer selection or notify another screen.
A current success refreshed only Permissions, leaving historical catalog rows
in the other canvases. The workbench now retains receipt ownership through the
write and rendering, rejects superseded views (including screen round trips),
and synchronizes all passive canvases after clearing their old catalog caches.
Busy remains set until publication finishes; retained native cancellation and
the actual recovery owner's source checks remain unchanged.

ADR required: no. Existing ADR-126 governs restored authority; ADR-150/161 govern
UI composition. This is a caller lifetime and passive display repair, without
new authority, persistence, service contracts, tokens or CSS.

## Verification

[45 distinct targeted cases pass](qualified-cases.json): 18 new real-owner cases,
17 existing mounted recovery controls, three owner approval/lifetime cases,
five adjacent synchronization/inspector cases and two real destination footer
cases. [Selected cases](selected-cases.txt), [adjacent cases](adjacent-cases.txt)
and individual [test receipts](tests/final-001.json) retain the exact boundaries.
No full suite ran. The two footer checks initially failed before UI creation
with `raw_source_selection_changed`; applying the repository's existing
`private_profile_test` wrapper made their unchanged assertions pass in
[footer-001](tests/footer-001.txt). No production profile guard was relaxed.

The initial [ten-case baseline](tests/red-001.txt) had eight meaningful failures:
old Servers rows survived review, late success overwrote navigation, and late
failure notified another view. [The first repair passed 27 checks](tests/green-001.txt).
Independent review identified rendering-phase continuation and screen round-trip
ownership. The corrected held-render fixture has [six meaningful red cases](tests/render-red-002.txt).
Its first version raced the ordinary mode-change clear; the first screen fixture
queried the default screen before its host was pushed. Those harness failures
remain in the earlier receipts, not as product evidence. The final current-owner,
superseded-owner, retained cancellation and held-render assertions all pass.

All [seven preflight guards](preflight.txt) pass. [Static analysis](static-analysis.json)
introduces no diagnostics; existing files retain their baseline counts, and new
files are clean. [Changed ranges and new files are formatted](formatting.txt).
[Independent review](independent-review.txt) found no remaining concrete blockers.
Two pytest warnings concern unrelated old temporary-directory cleanup.

## Native visual evidence

The real TldwCli ran with LinuxDriver and TTY streams using a fresh private
HOME/USERPROFILE/XDG/TLDW environment. A repository-owned fixture creates and
selects an actual isolated restore, then the real app mounts its normal MCP
service. Onboarding and catalog refresh are disabled in this disposable fixture.
The native MCP client connection, discovery and execution methods are guarded
against accidental calls; the fixture's network guard records no attempts.
No MCP runtime service or approval writer is replaced.

[Sixteen rendered and inspected captures](GALLERY.md) cover dark/light themes at
120×40 and 170×48. Each cell reads the review, reaches both complete action
labels by real Tab/focus navigation, cancels, reopens, and confirms. The first
cell activates the fresh roots; later cells repeat review of the same unchanged
generation. Assertions verify fresh Ask/local defaults, retained historical
bytes, no old server/Audit rows, and activation of only the MCP owners. The
other restored owners, including config and skills, remain unapproved.

[The native receipt](native/result.json) and [lifecycle check](lifecycle.json)
record normal App.run return, exit 0, absent process, released instance lock,
ten healthy private SQLite databases, zero conversations/messages and unchanged
user default files. Twelve source hashes and the runner hash match final source.
The fresh restored app logs one Evals `storage_scope_not_enrolled` diagnostic;
that unrelated owner's recovery is not qualified here. There are no other
recorded app errors and the faulthandler log is empty.

Two earlier native attempts are explicitly unqualified. [001](unqualified-native/001-result.json)
reached onboarding instead of MCP; [002](unqualified-native/002-result.json)
scrolled the already-focused Cancel control offscreen, then used a no-op focus
request. The corrected journey uses Tab to reveal it again. Both apps exited
normally with qualification status 1; the second required Escape to close its
modal before Ctrl+Q. The final runner includes that failure cleanup.
[Export hashes](export-manifest.json) distinguish originals and normalized copies.

## Bounds and continuation

This independent branch starts from dev `cef6bd2a3e3f0b8de0e166146acb4900ca7ea2b6`.
Earlier Audit and inspector repairs remain separate follow-up PRs. This slice
qualifies receipt ownership and removal of old passive projections: it does not
reload catalog/discovery, connect a server, grant a tool, or qualify later runtime
use of the reviewed definitions. Catalog repopulation and the persistent restored
review guidance after success remain follow-up UX review items. The 80×24 case,
bulk permission actions, approval workflows and connected-runtime journeys remain
outside this qualification.

Current-head CI and final visual approval are still required before merging.
[Task allocation](allocation-census.json), [fresh reference census](precommit-census.json)
and [sole ownership](allocation-owner-check.json) cover TASK-32839.
