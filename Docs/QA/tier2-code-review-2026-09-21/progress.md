# progress — tier-2 code review 2026-09-21

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review-t2` @ `origin/dev` = `3722a857480b94b30fd4755f3f8e3002bd163ec3`.

## Phase log
- Phase 0 started 2026-09-21.

## Slice coverage

| Slice | Area | Files | Lines | State | Read in full | Sampled | Mechanical only | Findings |
|---|---|---:|---:|---|---:|---:|---:|---:|
| S01 | Notes A - sync engine, conflict resolution, file services | 20 | 40126 | done | 19 | 1 | 0 | 12 |
| S02 | Notes B - templates, importers, remainder | 21 | 26458 | done | 6 | 12 | 3 | 13 |
| S03 | TTS A - backends | 29 | 26521 | done | 3 | 17 | 9 | 14 (S03+S04 joint) |
| S04 | TTS B - remainder | 62 | 39395 | done | 5 | 18 | 39 | 14 (S03+S04 joint) |
| S05 | Library | 57 | 45150 | done | 16 | 16 | 25 | 14 |
| S06 | API client | 61 | 38737 | done | 14 | 21 | 26 | 14 |
| S07 | Speech in | 55 | 39378 | done | 3 | 26 | 26 | 11 |
| S08 | Scheduling | 87 | 41236 | done | 13 | 25 | 49 | 8 |
| S09 | Characters | 66 | 35065 | done | 5 | 22 | 39 | 10 |
| S10 | Chunking | 70 | 26734 | done | 1 | 11 | 58 | 13 (S10+S26 joint) |
| S11 | Evals+ingest | 74 | 44016 | done | 9 | 18 | 47 | 17 |
| S12 | Media | 52 | 26474 | done | 11 | 17 | 24 | 13 |
| S13 | Canvas | 53 | 35510 | done | 6 | 28 | 19 | 11 |
| S14 | Web | 50 | 27083 | done | 9 | 16 | 25 | 12 |
| S15 | Models | 39 | 20758 | done | 3 | 10 | 26 | 14 (S15+S16 joint) |
| S16 | Persona | 50 | 23660 | done | 0 | 21 | 29 | 14 (S15+S16 joint) |
| S17 | Small pkgs + css python | 80 | 38316 | done | 8 | 24 | 48 | 15 |
| S18 | Screens A | 11 | 39343 | done | 1 | 8 | 2 | 11 |
| S19 | Screens B | 68 | 40757 | done | 35 | 8 | 25 | 15 |
| S20 | UI root (UI/*.py top level) | 33 | 31100 | done | 9 | 13 | 11 | 8 |
| S21 | UI modules A | 55 | 45614 | done | 5 | 23 | 27 | 12 |
| S22 | UI modules B | 77 | 29039 | done | 2 | 22 | 53 | 13 (S22+S23 joint) |
| S23 | Widgets rest | 39 | 10709 | done | 4 | 6 | 29 | 13 (S22+S23 joint) |
| S24 | Interop cluster | 178 | 68546 | done | 2 | 9 | 167 | 2 |
| S25 | Backup_Recovery (added by this run) | 80 | 42955 | done | 10 | 28 | 42 | 9 |
| S26 | Workflows + strays (added by this run) | 22 | 7676 | done | 18 | 3 | 1 | 13 (S10+S26 joint) |

## Phase log (detail)

### Phase 0 — provenance and baseline (complete)
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review-t2`, detached at
  `origin/dev` = `3722a857480b94b30fd4755f3f8e3002bd163ec3`. `git status --porcelain`
  shows only the two untracked qa/ review directories. Python 3.12.11, Textual 8.2.8.
- `scripts/preflight.sh`: **all derived-artifact checks passed** (9 checks).
- ruff `--select E9,F63,F7,F82` over all 52 Tier-2 packages + the 31 `*_Interop`:
  **2 fatals, both in `Audio/meeting_owner.py` (F821 `MeatingCapture` undefined, lines
  657 and 738)**. Everything else 0. Recorded as baseline, not fixed.
- **Scope bug found and fixed.** `slice_paths.txt` as delivered left **102 files /
  50,631 lines unclaimed**, including the whole 43k-line `Backup_Recovery/` package and
  the 10-file `Workflows/` package — both live and reachable from `app.py`, `config.py`
  and `cli.py`. Added as slices **S25** and **S26**; completeness check now `UNCLAIMED: 0`.

### Phase 1 — mechanical candidates (complete)
- `dup_census.py` run repo-wide: **1,943 same-name / 264 verbatim / 709 shape**
  (2,526 files scanned). Baseline re-measured with the same script at the quoted
  baseline SHA `d8fb4053f9`: **1,931 / 267 / 709**. See report.md "Census delta" —
  the prompt's quoted repo-wide baseline (1,823/244/665) does not reproduce.
- `helper_adoption.py` run repo-wide (patched: `Widgets/base_components.py` no longer
  exists, deleted in `5f3adeca33`).
- `pattern_greps.py` re-scoped to the 207 Tier-2 paths: 1,442 files scanned, 28 pattern
  TSVs in `candidates/patterns/`.

### Phase 2 — per-slice read
Wave 1 dispatched: S01, S05, S06, S11, S18, S19 (the six depth-priority slices).
