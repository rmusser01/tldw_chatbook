# Workflows PR2690: rebase and Qodo remediation

Scope: authoring-only Workflows, TASK-32601, ADR-138 and ADR-150. No execution,
server sync, provider, dependency, schema, or SQLite ownership infrastructure was
added. Existing disposable UAT evidence and unrelated worktrees are preserved.

## Rebase

Original remote head: `15b31e4210a919e65060b9b3224c3390b44c3814`.
The prior UAT corrections were committed before rebasing, and backup branch
`codex/workflows-authoring-pre-rebase-20260915` retains that checkpoint.
The first rebase onto `48d40df8ce` preserved both independent additions in the
only conflict, the testing lessons document. After dev advanced, all 14 feature
patches replayed identically onto `94cc1200d5d8f2d2ef73bfb3fd4ce2d4da0ac62a`;
`git range-diff` reported every patch unchanged. Code checkpoint:
`e1e4223745f951b4b1078eebf397726defc45002`.

## Review dispositions

| Qodo comment | Change or evidence |
| --- | --- |
| 4017261511: unbounded lists | Validated SQL pagination (20 default, 100 maximum), all-library Unicode casefold search, reachable workflow/history/draft pages, and independent exact head lookup. Slow previous searches cannot publish over newer results. |
| 4017261520: raw-edit freezes | Reuse prepared projections instead of parsing once per displayed field; retain controls for same-layout raw edits and update derived labels in place. Structural, field-shape, identity, focus, and read-only changes still rebuild. No new draft owner or debounce queue. |
| 4017261530: class documentation | Google-style class summaries and Attributes for Revision, Draft, Issue, StepContract, and DiscoveryEntry. |
| 4017261536: Console types | Both helpers and stored state use the existing nullable HomeActiveWorkItem type; TYPE_CHECKING avoids an added startup import. |
| 4017261546: rejected save blocks quit | Settle expected revision validation/conflict failures before the final durable flush; real draft-write failures still block close. Six rejection/flush outcome combinations are tested. |
| 4017261558: import complexity | Central iterative limits: 500 steps, 64 container levels, 100,000 values/containers, with the existing 16 MiB text limit. Oversized legacy saved definitions remain raw-inspectable and exactly exportable; only an exact durable-base projection can bypass new admission during draft preservation. |
| 4017261568: cancelled close cancels save | False-positive premise: the original reviewed head already awaited the retained save through asyncio.shield. A real SQLite regression cancels close after the save physically commits, then verifies the retained owner reconciles its base and draft. No duplicate shield layer added. |

## Verification

Commands use the repository Python 3.12 environment with `PYTHONPATH=.` in the
feature worktree, and isolated test profiles. No full repository test sweep.

- `pytest Tests/Workflows Tests/DB/test_workflows_authoring_storage.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly`: **271 passed**, 42.61s.
- `pytest Tests/UI/test_workflows_editor.py Tests/UI/test_workflows_paging.py Tests/UI/test_workflows_projection_performance.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly`: **102 passed**, 115.29s.
- `pytest Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly`: **11 passed**, 7.36s.
- After the final rebase, `pytest Tests/UI/test_workflows_editor.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_shells.py -k workflows -q --timeout=60 --tb=short --show-capture=no -p no:randomly`: **77 passed**, 170 deselected, 115.86s.
- After the boundary corrections, `pytest Tests/Workflows Tests/DB/test_workflows_authoring_storage.py Tests/UI/test_workflows_projection_performance.py Tests/UI/test_workflows_paging.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly`: **314 passed**, 54.24s.
- Scoped Ruff check and format: all 26 authoring/domain/new-test files clean;
  `git diff --check` clean. The separately approved, source-attributed legacy
  no-new-static-debt qualification remains unchanged.
- Derived-artifact preflight passed on both rebases. It required registering
  a moved constant diagnostic and the existing history index's real query-plan
  test; no diagnostic sink or database index/schema was added.

The 500-step mounted raw-edit test checks unchanged control identity and exact
raw text. Its measured handler time was 55.84 ms, maximum heartbeat gap 55.87 ms.
The test does not impose machine-dependent timing assertions. Pure validation
of the sampled 500-step near-16-MiB document improved from 28.030s to 30.325ms.
These are measured cases, not worst-case latency guarantees. Initial/structural
500-control mounts and arbitrary TextArea layout remain potentially slower;
the complexity guard bounds downstream traversal, not decoder peak allocation.

Existing RequestsDependencyWarning and pytest cleanup warnings for unrelated old
Kokoro temporary directories remain. A temporary disk-exhaustion run is not
counted as passing; the successful runs above were performed after recovery.

## Integration gate

Independent review confirmed the unchanged rebase and legacy search through
1,000 container levels. It found two aggregate-node boundary bugs: adding lineage
could save an over-limit head or copy invalid history while clearing its error.
Both were reproduced with failing tests. Serialization now checks transformed
complexity before encoding, and copy refuses an invalid validated result before
writing. Six boundary cases cover successful exact-limit Save, rejected Save,
and revision/draft Copy with absent or existing clean target buffers.

The 314-test regression run passed. Independent scoped re-review of
`4fe2c2b5c8b73e5812de04d251dc29b37071c686` confirmed both findings resolved,
including unchanged saved heads/drafts and exact legacy preservation; no new
actionable findings. GitHub replies, required-check results, and merge outcome
must be verified separately; this record is not a merge-success claim.

## Approved CSS CI correction and second review

The pushed `f17bf03c3a` head passed the required derived-artifact check, but
UI latency CI exposed two introduced boot-budget violations. Both reproduced
locally before changes: 770,849 parsed bytes against 768,000, and 275
ancestor-scoped bare-subject rules against 274. Neither limit was changed.

The existing `ScreenOwnedSplit` registry now partitions `_workflows.tcss`,
and the existing app route loader loads its generated sheet on first entry.
Only workflow-prefixed IDs/classes move; bare types and shared helper classes
remain in the boot bundle. Exact-token Python-consumer inspection, with a
no-owner negative control, found no consumers outside the Workflows UI.
The Console-context rule now names its two existing label IDs instead of every
`Static`. No token values, layout dimensions or loading infrastructure changed.

After rebuilding, 3,900 source bytes are deferred; boot parses **767,019 bytes**
(981 bytes below the unchanged cap), and the bare-subject census is **272**.
New real-app regressions failed for absent stylesheet registration before the
fix, then passed for both Home-to-Workflows navigation and Workflows as initial
route. They verify painted Console text, one-row controls, dialog hit targets,
Cancel and restored opener focus, not only a stylesheet-registration flag.
The new route checks, exact partition/build checks and both original budget
guards passed together: **8 passed**, 8.79s. Production-CSS harnesses explicitly
include the deferred sheet to model the real route's styled state.

| Second-pass Qodo comment | Verified resolution |
| --- | --- |
| 4019797880 | Google-style controller API documentation covers arguments, results and applicable failures; local validation does not promise runtime execution. |
| 4019797888 | `change_steps` uses `Unpack[StepEditOptions]` with the three actual optional keywords: integer offset, string step_type and boolean before. Forwarding behavior is unchanged. |
| 4019797896 | One `DRAFT_DEBOUNCE_SECONDS` constant controls the timer and displayed milliseconds; a real-timer regression changes policy and checks durable persistence and status. |
| 4019797904 | Both discards use the existing transition lock and recheck source/version after waits. Pending discard checks durable-base identity; full discard cannot adopt newer protected text. Existing retained repair runs outside the non-reentrant lock. Selection, newer raw/field edits and cancellation have real-owner regressions. |
| 4019797908 | An existing-base draft missing portable identity is invalid; exact authored text and the prior valid projection survive reopen. Save cannot synthesize away the deletion. Base-free create/import still assign fresh identities and retain opaque content. |
| 4019797913 | Conditional `media_ingest.text` is unverified, so reference choices warn that runtime validation is required. |
| 4019797922 | `media_ingest.metadata` is an array; reference choices include it only for compatible array consumers. |

The media declarations were checked read-only against `tldw_server` remote dev
`2e1a5e58d3344a1efd578efb4dbfb1c9465e8767`, matching its local origin/dev object.
`tldw_Server_API/app/core/Workflows/adapters/media/ingest.py:74–79` initializes
metadata as a list and omits text; lines 109–111 add text only for nonempty
extraction. No server execution or server writes were performed.

Second-pass verification (overlapping selections, not an aggregate full suite):

- New discard cases first produced **9 expected race failures**, then the full
  draft-owner file passed **48 tests**, including retained repair/close controls.
- Deleted-identity cases first produced **4 expected failures**, and reference
  choices produced **3 expected failures**. The final domain/catalog selection
  passed **189 tests**.
- Coordinator: `Tests/Workflows` plus `Tests/DB/test_workflows_authoring_storage.py`:
  **301 passed**, 33.99s.
- Coordinator: editor, paging, projection, stylesheet entry, CSS build integrity,
  consolidated harness, design-token and bundle-sync selections: **150 passed**,
  145.46s. This includes painted/focus/hit-target checks at 160×48, 110×36 and
  60×20. Sampled 500-step raw edit: 39.31ms, heartbeat gap 44.66ms.
- The five boot-budget CI modules passed **20 tests**, 42.12s. Existing warning
  headroom is zero for the UI-ready module census; neither that cap nor CSS caps
  were relaxed. The optional transformers import probe also reported a joblib
  low-disk-space serial fallback; the selection still exited 0. Disk inspection
  found approximately 1 GiB free. No unrelated data was deleted.
- All seven `scripts/preflight.sh` derived-artifact checks passed.
- All 13 Python files changed in this round passed the source-attributed
  no-new-static-debt check against `f17bf03c3a`. App lint drops 485→484; its 37
  formatter edits, build.py's 2 lint/3 formatter findings and CSS integrity
  tests' 3 lint/5 formatter findings match identical baseline spans/replacement
  text. All other changed/new files are clean. The touched route map is now
  explicitly `ClassVar`; no suppression or lint setting changed. After that
  annotation-only correction, route-entry and app-import checks passed **9 tests**,
  14.97s. `git diff --check` passes.

Independent scoped review, GitHub replies and checks on the next pushed head
remain integration gates; these local results are not a merge claim.
