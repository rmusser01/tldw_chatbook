# Library L3b QA evidence — Ingest canvas + job registry + Home feed + legacy cleanup

Plan: `Docs/superpowers/plans/2026-07-10-library-l3b-ingest-jobs-home.md`
Captures: textual-serve + playwright chromium, viewport 2050x1240, device_scale_factor 1,
isolated HOME `/private/tmp/tldw-l3b-qa`, seeded 4 notes / 4 conversations / 3 media / 3 decks /
7 due flashcards / 2 quizzes, plus sample ingest files (incl. a bracket-hostile filename and an
unsupported `.xyz` extension).

## Captures

| Capture | What it shows |
|---|---|
| `l3b-library-landing.png` | Post-L3b rail: Ingest section has ONE row (`Import media · in Library`); the placeholder Import/Export row is gone; all counts intact. |
| `l3b-ingest-idle.png` | The ingest canvas: `Import media` header, path input + `Browse…`, Title/Author/Keywords, `Advanced options` collapsible, disabled `Start ingest`, `Queue` with per-state counts + `No ingest jobs yet.` |
| `l3b-ingest-queue-mixed.png` | Live queue after three submissions: `✓ done · notes [draft].txt · 0s` (bracket filename renders safely) + `✓ done · tidal-bores.txt · 0s`, each with `Open in Library`; `✗ failed · report.xyz · Unsupported file type: .xyz…` with `Retry`; counts line `0 queued · 0 running · 2 done · 1 failed`; rail `Media` count grew live (3→5) from the completion poke. |
| `l3b-open-in-library-viewer.png` | `Open in Library` on a done job lands straight in the in-Library media viewer on the ingested item (`neap-tides`, Type: plaintext). |
| `l3b-home-failed-ingest.png` | Home Needs Attention (2): the failed ingest job `report.xyz · Library` + the L3a flashcards-due row; canvas shows source/status and the control row. |
| `l3b-home-retry-pressed.png` | Home `Retry` on the failed ingest job → `Retry queued for report.xyz.` toast (wired to the real requeue seam). |
| `l3b-home-open-details-landed.png` | Home `Open details` on the failed ingest job → lands on the Library ingest canvas with the failed job visible + confirmation toast. |
| `l3b-handoff-flashcards.png` | Post-cleanup `handoff` canvas kind (flashcards): trio + WIP/snapshot callouts + open button — byte-identical rendering to the retired mode canvas. |

## Bugs found and fixed BY this live QA (all committed with RED-first regression tests)

1. **Re-ingesting an unchanged file lost its `Open in Library` action** — the DB's update path
   returns `media_id=None`; the runner now resolves the id via `get_media_by_url` (`780269cf`).
2. **Home `Open details` on an ingest item silently did not navigate** — `NavigateToScreen` to the
   CACHEABLE `library` route reused a previously-unmounted cached screen instance that never
   repaints; fix invalidates the Library screen cache first (the `open_notes_workspace` pattern).
   The L3a flashcards path only worked because `study` is not cacheable. Any future deep link into
   a cacheable route must invalidate its cache the same way (`7d7ccd42`).
3. **Home `Retry` on a failed ingest job showed the generic "not connected" toast** — now wired to
   `retry_library_ingest_job` with an honest queued/no-longer-retryable toast (`321cb473`).

## Accepted v1 limits (stated in module docstrings)

- In-memory registry: queue history dies with the app; quitting waits for the in-flight file to
  finish its DB write, queued jobs are lost; serial queue (parallelism is a follow-up).
- Home rows refresh on mount/interaction (matches the watchlist/chatbook feed model — no live push).
- Ingest job rows carry no age label (`updated_at` deviation: registry timestamps are monotonic,
  not wall-clock).
- One file per submission — the queue provides batching (TAB_INGEST's batch-select equivalent).
- No `type:` control and no URL input (inventory: `ingest_local_file` has no type override and
  TAB_INGEST ships no local URL ingest).

## UX wave (user-approved sr UX/HCI review findings, commits a72acafa + 32b1971d)

Findings A1–A6, B1–B2, C1–C2, D1–D3 implemented (opus review: Approved, zero issues);
B3 (seam error copy claims XML support) and C3 (no age on ingest rows) logged as follow-ups.

| Capture | What it shows |
|---|---|
| `l3b-ux-ingest-idle.png` | `(optional)` metadata placeholders, `Enter a file path to start.` quiet gate line, `Chunk size (words)` labeling. |
| `l3b-ux-ingest-queue.png` | Non-zero-only counts line (`2 done · 1 failed`), indented row actions (association fixed), `Dismiss` on failed rows, `Clear finished`. |
| `l3b-ux-ingest-after-dismiss.png` | Dismissed failure removed from the queue (and from Home — pilots assert the ripple). |
| `l3b-ux-handoff.png` | Rebuilt handoff canvas: header → one purpose line → `Carries forward: a, b, c and N more.` → `Generation and review run in Study.` → primary open button; WIP roadmap callout gone; ready state muted. |
| `l3b-ux-home-canvas.png` | Merged status line (`● failed · Library`), selection-aware primary (`Retry` accented for a failed item; `Review flashcards` accented only when its row is selected). |

Registry lifecycle additions: retry SUPERSEDES the failed original (one row per file);
`dismiss(job_id)` (failed-only) and `clear_finished()` with exactly-once listener semantics;
title clears with the path after submit (author/keywords stay as batch metadata).

## GATE DECISION: TAB_INGEST deprecation — RESOLVED (user, 2026-07-10): option (a) KEEP

The top-nav `Ingest` tab STAYS routed (user decision, option (a)): its server-mode Sources and
Web Clipper panels call real server seams and remain reachable there; the Library canvas is the
local ingest path. Deprecation is revisited when server ingest ships end-to-end (the Server Jobs
tab's mis-wiring — `submit_media_ingest_jobs` et al. do not exist on the scope service, masked by
mocked tests — is a pre-existing follow-up below).

## Follow-ups (logged)

- TAB_INGEST Server Jobs tab mis-wiring (`submit_media_ingest_jobs` vs `submit_ingest_jobs` —
  pre-existing, server-mode only).
- Bulk Library export (no seam exists; the removed Import/Export row's build-out).
- Ingest parallelism + persistent job history; URL/web local ingest.
- Open-in-Library on a media row deleted between done and click lands on an empty viewer
  (pre-existing L3a asymmetry with the conversations branch).
- `notes_create` nav-context branch shares the (now-fixed-for-ingest) stale-form gap.
- Home `Route: library` line exposes internal route ids (Home-wide copy follow-up, pre-L3a).
- Handoff-canvas copy polish (dense legacy wording retained verbatim by the behavior-preserving
  cleanup).
