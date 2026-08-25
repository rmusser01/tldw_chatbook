# Task 6 review package — Research round trip and closeout

Review base:

`ebf80f954923b33cc2397148d4eedec8bcf704eb`

Expected commit subject:

`fix: close Research Sources review gaps`

## Review order

1. `tldw_chatbook/UI/Screens/library_screen.py` and the mounted canvas tests:
   persisted-owner display, pending-intent sequencing, serialized persistence,
   UI-loop completion repaint, latest-generation fencing, and failure recovery.
2. `Tests/integration/test_research_source_round_trip.py`: the Server case now
   traverses production app dispatch, registry reconciliation, terminal
   listener, scheduler, coordinator, generated My Media catalog, and exact
   workspace-source association with observable zero Local calls.
3. The UTC cleanup and lint evidence: task-owned `UP017` is green while the 11
   upgrade-style findings on pre-existing lines remain explicitly baseline.
4. User guide, Task 6 report, and TASK-21508: exact canonical Server result
   ownership, fix-round inverse evidence, targeted gates, and Done status.

## High-risk invariants

- Intake always lands in the selected authority's canonical general catalog
  before an idempotent association; a later stage cannot roll that owner back.
- A captured qualified workspace, not current visible UI state or a keyword,
  owns association and resume.
- Server IDs remain Server-only. Profile/principal mismatch mutates neither the
  remote workspace nor any Local owner.
- Unlink removes membership and matching retrieval scope without deleting
  canonical Library/Media content.
- Desired selection does not impersonate readiness; Hybrid requires both FTS
  and vector readiness.
- The private overlay cannot hold note bodies, source content, paths, tokens,
  or secrets.
- An installed distribution contains every runtime migration needed to reach
  schema v43, not only the newest Task 5 artifact.
- A background preference write may update the mounted Library canvas only on
  the UI loop and only for the latest selected generation. Stale completions
  cannot repaint or become the final persisted owner.
- A Server round-trip test must derive its canonical item from production
  submission/reconciliation; a constant fake result cannot satisfy the guard.

## Verification snapshot

Fix-round gates include 138 mounted ingest-canvas, 3 exact backend-worker, 12
Research Sources, 20 Server/remote runner, 7 round-trip integration, and 43
installed-packaging passing checks. Four new review mutations (missing repaint,
missing generation fence, Server ID copied into Local media, and constant fake
catalog) each failed their named guard and were restored. The full Library
screen file retains one exact-base failure for a stale top-button selector; its
four changed backend/owner tests pass. The prior isolated F10 smoke remains
valid; no test Server API is available, so live Server behavior is not claimed.
Full pytest was not run by repository policy.
