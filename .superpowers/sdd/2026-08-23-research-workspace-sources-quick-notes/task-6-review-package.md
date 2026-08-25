# Task 6 review package — Research round trip and closeout

Review base:

`bddc90a4ab276de31feb4899b65b15385e85132b`

Expected commit subject:

`docs: complete Research Sources and Quick Notes`

## Review order

1. `Tests/integration/test_research_source_round_trip.py`: canonical Local
   catalog and membership lifecycle, restart/resume, association failure,
   unlink retention, tag projection, qualified Server ownership, and no blend.
2. `MANIFEST.in`, `pyproject.toml`, `Packaging/check_manifest.py`, and
   `Tests/Packaging/test_installed_distribution.py`: complete v40-to-v43
   installed runtime migration chain and artifact mutation detection.
3. `Docs/User_Guide/research_workspace.md`: exact separate-screen navigation,
   owner selection, ASCII pane controls, shipped Sources/Quick Notes behavior,
   and honest limitations.
4. Focused test maintenance: rolling UTC date evidence, mounted Library worker
   seam, and changed-file lint cleanup.
5. TASK-21508 and Task 6 report: AC-to-evidence reconciliation, targeted-gate
   counts, inverse checks, and isolated live-verification limitation.

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

## Verification snapshot

Focused gates total 1,742 passing checks plus one Windows-only skip across the
recorded split boundaries; overlapping files are intentionally counted only as
their command outputs, not as a full-suite claim. All seven required inverse
mutations failed and were restored. Isolated F10 live smoke passed; no test
Server API was available, so live Server behavior is not claimed. Full pytest
was not run by repository policy.
