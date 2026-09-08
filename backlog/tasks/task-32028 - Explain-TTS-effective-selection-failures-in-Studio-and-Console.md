---
id: TASK-32028
title: Explain TTS effective-selection failures in Studio and Console
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 04:56'
updated_date: '2026-09-08 15:50'
labels:
  - tts
  - bug
  - diagnostics
dependencies: []
references:
  - backlog/decisions/039-global-and-studio-tts-settings-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TTS selection failures currently collapse to an exception class and generic configuration guidance, preventing users from identifying invalid settings or stale state and encouraging ineffective retries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Studio and Console resolution failures identify the affected setting scope and appropriate recovery action.
- [x] #2 Failure logs preserve bounded resolution code, axis, and source without text, credentials, raw identifiers, or exception payloads.
- [x] #3 Targeted tests distinguish invalid selections, missing choices, unavailable catalogs, and stale preferences and retain existing non-resolution failure handling.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/039-global-and-studio-tts-settings-ownership.md
Reason: Repair bounded, value-free diagnostics and recovery guidance using the existing resolution error contract; no new stored data or trust boundary.

1. Add failing tests for actionable scoped resolution copy and bounded diagnostics in Studio and Console.
2. Reuse a shared fixed-copy mapping for resolution failures and retain code, axis and source at existing log sites.
3. Verify typed error branches, malicious/raw payload exclusion and unrelated failure handling; refresh the diagnostic inventory if required and document the recovery behavior.
4. PR #2512 Qodo follow-up: document the public recovery formatter return and privacy contract; verify its existing diagnostic regressions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added shared value-free recovery copy to TTSEffectiveResolutionError and used it at the Studio generation, Console generation and trusted automatic-speech preflight boundaries. Invalid settings identify their axis/owner, catalog failures offer refresh and inconsistent Studio state offers reopening Speech Lab. Existing unrelated typed failure handling remains intact. The three diagnostic sites retain only constructor-validated resolution code, axis and source plus the existing Console outcome; raw exception strings, text and identifiers remain excluded.
All 19 new diagnostics cases failed before implementation and pass afterward. They exercise all three error boundaries with private payload canaries and distinguish invalid/missing/unsupported/catalog/stale settings. Combined TTS focused checks: 216 passed; diagnostic inventory and metadata-only gates: 2 passed. Regenerated only the two reviewed diagnostic owner rows and their count, preserving unrelated inventory work and sink topology. New tests lint/format clean; production changed ranges formatted with zero new Ruff findings, diff check clean and independent privacy review approved.
No new ADR: existing ADR-039 settings ownership and ADR-029 metadata-only diagnostics apply. Updated TTS_MODULE_GUIDE.md and Console recovery guidance. Production changes are effective_settings.py, stts_events.py and tts_events.py; regression test_resolution_failure_diagnostics.py. Scoped changes also applied to the isolated newer dev worktree.

The newer dev schema-v3 diagnostic inventory also tracks heuristic path candidates. Reviewed the added typed-resolution outcome_code branch: TTSEffectiveResolutionError maps to the fixed configuration_invalid outcome; resolution code/axis/source are constructor-validated metadata. Preserved the existing generic-outcome branch and legacy file_path candidate, and refreshed only this owner candidate row/count alongside the two already-reviewed owner digests. No unrelated candidate status or sink topology changed.

Final combined dev verification: all six derived-artifact preflight gates passed, including the schema-v3 diagnostic inventory. The integrated TTS/settings cohort passed 200 tests under Python 3.12.11. Baseline-relative Ruff found zero new findings across all 23 changed Python files in the combined fix.

PR #2512 Qodo follow-up: documented recovery_message with a Google-style Returns section and its fixed, value-free UI contract. The encoded-audio limit added for TASK-32027.1 now has literal Console recovery guidance that excludes upstream private payloads and avoids suggesting retries. The added malicious-payload regression failed before the change and passes afterward; the combined TTS gate passed 68 tests and the adjacent admission/provenance gate passed 225. Public diagnostic calls and inventory are unchanged; formatting, baseline-relative Ruff (zero introduced findings), and all six derived checks pass. No new ADR is required; ADR-039 and the existing typed error boundary remain applicable.
<!-- SECTION:NOTES:END -->
