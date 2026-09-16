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

## Final dev rebase and recovery-message check

While the second review ran, dev advanced to
`67bfde41d196dd64a4df45f4cb5059e191da4d21`. GitHub reported a conflict although
its PR base SHA was temporarily stale. Live remote refs established the new tip.
Backup `codex/workflows-authoring-before-dev67bf` retains `e5fde9a080`; rebase
produced `5d61911395f5ebef681d74116e008b7e74a41b38`. The only conflict was
independent appends to the lessons document; both remain exactly once. Range-diff
shows 16 identical patches, with only append context differing in the seventeenth.
Core domain/controller and Workflows CSS files are byte-identical across rebase.
All seven derived-artifact preflight checks passed again on this base.

The first fresh integration selection found one recovery-message mismatch
(96 passed, 1 failed). The earlier 150-test UI run had started before the draft
implementer's final shared-message edit, so it did not qualify that wording.
Draft-preservation assertions passed; the new generic wording omitted the visible
"confirmation" instruction expected by the recovery flow. The shared message now
says "open a new confirmation", retaining its applicability to discard and repair.
The strengthened owner check produced three expected RED failures before the
one-line correction. The complete owner/editor/stylesheet/boot/authoring rerun
then passed **145 tests**, 126.62s, including the original mounted recovery case.
On this newer base, boot CSS is **767,424/768,000 bytes** and the selector census
remains **272/274**. No limits changed. Scoped Ruff/format and diff checks pass.

Independent scoped review reported no Critical/Important findings for the second
round, verified unchanged code across rebase, and accepted the recovery-message
correction conditional on its passing owner/UI tests. Qodo review `5217319999`
on `e5fde9a080` marked all behavioral findings resolved and dismissed the prior
performance claim. Its sole new comment `4021489070` requests fuller contracts
for `update`, `discard_pending`, and `discard_draft`. Google-style documentation
now describes raw-input admission, pending versus durable returns, explicit
confirmation preconditions, preservation and applicable failures for those three
methods. These docstrings introduce no additional behavior.

The final narrow review found no blocking issues and independently exercised all
three stale-confirmation cases in memory. Its one documentation precision was
applied: pending discard restores field provenance on the owner, not inside the
returned `Draft`.

All posted original and second-round inline discussions have evidence-backed
replies and are resolved. The final documentation reply, refreshed exact-head
review/checks and requested merge are still to be confirmed on GitHub.

## Final-head library summaries and widget contracts (2026-09-16 UTC)

Head `2e76d7bdf245079a2ccbe2aef2430dfd13e04284` passed PR Fast Lane,
the required derived-artifact job and UI latency guardrails. Qodo review
`5217368702` nevertheless raised four new findings, so merge was held.

| Qodo comment | Verified resolution |
| --- | --- |
| 4021531211 | Google-style public contracts now cover the library, navigator and reference-picker APIs. |
| 4021531225 | Public widget constructors, events and paging signatures declare their existing row shape, argument and return types. |
| 4021531230 | `deepcopy` is a module-level standard-library import in catalog.py. |
| 4021531239 | Both library views read bounded name/identity-only rows through the existing document service, without fetching or projecting complete revisions merely for labels. |

The two new mounted regressions first failed with 20/21 full-document projections
and approximately 10/10.5 MiB processed per page of 512-KiB sample definitions.
Both now make **zero projections**. The same initial sample's main-page load
fell from 18.38ms to 7.25ms; the final combined run measured 10.65ms (main) and
76.92ms (compact, including the UI wait). These samples are not worst-case bounds.
SQLite still inspects JSON to extract names; full definitions no longer cross
the query boundary for these pages. Search still scans bounded batches to retain
Unicode casefold matching and matching-row offsets.

`list_workflow_summaries` returns the existing `(name, workflow_id, revision_id)`
row shape. It shares the existing search logic, bounds, owner and read transaction;
no schema, connection, cache, index or runtime was introduced. Exact head reads
are unchanged. Summary names are display data, not authoring-validation badges.
Legacy 1,000-level and opaque-number content remains selectable without rewriting
saved bytes. Malformed/deeper-than-SQLite JSON fixtures were rejected by the
existing database CHECK and removed from the test design, not supported by
bypassing that constraint. ADR-138 records this narrow contract amendment.

Verification (overlapping targeted selections, not a full sweep):

- Summary service, mounted paging and document service: **166 passed**, 17.71s.
- Complete Workflows domain, authoring storage, editor, paging, projection and
  first-route stylesheet selection: **420 passed**, 156.76s. The 500-step edit
  measured 41.23ms with a 54.27ms maximum heartbeat gap.
- Widget/catalog implementer selection: **20 passed**, 22.51s. An AST comparison
  ignoring documentation, annotations and relocated imports found the four
  widget/catalog files' executable syntax unchanged.
- App-import weight, boot CSS bytes and Textual CSS fastpath: **13 passed**,
  27.64s. Boot remains **767,424/768,000 bytes**, selectors **272/274**.
- All ten changed/new Python files pass scoped Ruff check and format; diff-check
  passes. Existing source-attributed shared-file static debt is unchanged.
- All seven `scripts/preflight.sh` derived-artifact checks pass again.

Existing Requests dependency, old Kokoro temporary-cleanup, boot headroom and
datetime deprecation warnings remain. The optional transformers probe reports
joblib's resource-exhaustion serial fallback; that check exits successfully.
No unrelated files were deleted. GitHub results on the next pushed head, the
independent review disposition and the merge must still be verified separately.

### Independent-review correction: admitted Unicode names

The reviewer reproduced an admitted escaped lone surrogate in a workflow name
causing SQLite's default UTF-8 text decoder to reject the entire summary page,
including when an unrelated normal workflow was explicitly selected. Two new
service cases (high/low surrogates) and both mounted library layouts failed
before the correction; embedded NUL and emoji controls already passed.

Only extracted string names now cross the query boundary as BLOB values and are
decoded locally with UTF-8 `surrogatepass`, matching the existing JSON decoder's
admitted strings. Non-string and missing-name formatting, saved bytes, connection
text factories and storage contracts are unchanged. Both exact summary search
and existing full-revision search retain these names. The corrected summary,
paging and document-service selection passed **172 tests**, 18.61s. Ruff's
duplicate-parameter detector conflates literal lone surrogates, so the regression
names explicitly use distinct `chr` code points; no suppression was added.

The same independent reviewer closed the finding after verifying all four service
cases, both original controller-load reproductions and eleven adjacent string/
non-string cases. No remaining Critical/Important findings; the scoped code review
is ready for commit/push, conditional on final verification and exact-head CI/Qodo.

The next combined run had **425 passed, 1 failed**: the existing incomplete-field
screen test hit-tested the max-token field before its focus-scroll animation
finished. Its isolated rerun passed without a production change. Instrumentation
then observed `scroll_y=15.74` with target `19` before the assertion; waiting for
scheduled animations produced `19/19` and a visible field. The test now awaits
that existing animation contract at both hit checks, as neighboring view-state
tests already do. All three incomplete-field variants passed the diagnostic run;
temporary instrumentation was removed. No UI timing, layout or scroll policy
was changed to satisfy the test.

Final corrected-tree verification: the complete targeted selection passed
**426 tests**, 140.11s, including real app/file-picker/lifecycle paths and all
new summary/Unicode regressions. All seven preflight guards pass on the corrected
production code; all ten changed/new Python files pass Ruff and formatting.
The final sample measured 6.87ms main / 71.23ms compact library loads, both with
zero full-document projections; 500-step raw editing was 37.58ms with a 49.09ms
maximum heartbeat gap. Independent review is closed with no blockers. These
results qualify pushing the fixes, not merging before new-head CI/Qodo completes.

## Search-boundary follow-up (2026-09-16 UTC)

Head `af041970e9afef81c6d60af73912e32cbfabd688` passed PR Fast Lane,
required derived artifacts and UI latency CI. Qodo review `5217561076` added
comments `4021705887` (unbounded search input) and `4021705892` (repeated OFFSET
scans), so merge was held again. The two Windows jobs still failed during checkout
of unchanged dev task32540's long filename; their logs were verified separately.

Both search APIs now use `WorkflowSearchInput` in the existing shared validation
module. Its strict page-size/offset constraints retain existing limits, and raw
queries are capped at 512 Python characters without trimming, coercion or case
normalization. Two oversized-input regressions first proved SQLite was reached
before rejection. A plain Pydantic field validator deliberately preserves lone
surrogates: a direct probe showed ordinary Pydantic `str` validation rejects those
already-admitted JSON names. Errors omit raw input; saved definitions and
connection settings are unchanged. The user guide documents shortening a query.

Search now executes one ordered statement and calls `fetchmany(100)` on the same
cursor/transaction, skipping matching offsets in Python and stopping once full.
Four real-SQLite tests first reproduced 10/11 query executions over 1,000 heads.
Afterward all six sparse/absent/early-stop cases make one query, with at most 100
rows per fetch and the expected matching identities. Instrumented SQLite progress
callbacks (one per 100 VM operations) fell from 471/531 to 180 ticks for the sampled
full scans, and 18 for early stopping. These are deterministic sample work counts,
not universal timing promises. No schema, index, cache or storage owner was added.

Intermediate verification: 177 shared-boundary/service tests passed before the
scan correction; then 191 summary/service/paging tests passed. With early-stop and
mounted oversize/recovery cases added, 50 summary/paging tests passed, and the six
instrumented scan cases passed separately. Both real library layouts recover by
editing the search and retain the exact open draft. These selections overlap.
Shared input_validation.py retains eight source-attributed baseline Ruff findings
(unchanged spans, codes, messages and columns; zero unmatched). Other three Python
files are clean, all four format checks and diff-check pass. Final targeted,
preflight and independent-review results remain to be recorded before pushing.

Independent review reports no Critical/Important/Minor findings for this diff.
Its in-memory verification covered six scan cases, twenty-two invalid-input
cases, exact-limit and unchanged whitespace/Unicode/NUL/surrogate strings. Cursor
reuse and completed identity collection before full-revision retrieval were
verified. All seven repository preflight guards pass on the corrected tree.

Final verification: **520 passed**, 192.06s, covering Workflows, authoring storage,
editor/paging/projection/route-entry and the affected shared reasoning/character
validation consumers. Separately, **13 import/CSS performance checks passed**,
29.13s; boot CSS remains 767,424/768,000 bytes and selectors 272/274. Existing
dependency, resource-fallback and old temporary-cleanup warnings remain disclosed.
No full sweep, model call, suppression or unrelated cleanup occurred. Final
list_workflows documentation spells out the changed boundary without executable
changes; scoped Ruff/format/diff checks pass. Exact-head GitHub gates remain pending.

## Shared SQLite bound (2026-09-16 UTC)

Head `f67867129b8038cae9d681426ec8ae13463daaad` passed Fast Lane, required
derived artifacts and UI latency. Qodo review `5217694677` raised only
`4021826851`: the shared search model duplicated the SQLite integer maximum.
`SQLITE_INTEGER_MAX` now names that value in the existing input-validation module;
the search model uses it and document_service imports it as `MAX_GENERATION`.
This is mechanical consolidation, not a behavior, schema or infrastructure change.

A real SQLite characterization accepted/stored/reloaded the maximum generation
and accepted maximum offsets in all collection paths before the refactor, then
passed afterward. No failing behavioral test is claimed for an unchanged value.
Fresh service, summary, draft-owner and mounted-paging selection: **244 passed**,
22.96s, including existing one-over/type rejection. Both changed implementation
files and the test are formatted; new-code Ruff and diff checks pass. The eight
shared-module Ruff findings remain exactly source-attributed with zero unmatched.
Independent review found no issues and checked maximum/one-over behavior and
unchanged import dependencies. Exact-head CI/Qodo remains required before merge.
All seven derived-artifact preflight guards also passed on this corrected tree.

## Rebase onto the Notes integration (2026-09-16 UTC)

Qodo review `5217761250` on `305b90e0b2426f81cfbe8e056f51a60bd89133e0`
raised only import-spacing comment `4021886696`. The reported internal blank
lines do not exist: local imports are contiguous at lines 13–37, separated from
third-party imports only at line 12. Fresh Ruff import-order and format checks
pass. Evidence reply `4021948606` records the false positive and its thread is
resolved; no formatter-defined import layout was changed to satisfy it.

While the previous head's required Derived, Fast Lane and UI latency checks
passed, dev advanced to `65a1437183de025d776afe0dd2c526c2f2423201` and GitHub
reported a conflict. The backup `codex/workflows-authoring-pre-dev65a143` retains
the old head. All 21 patches were rebased, resulting in `7bca061ada1f05fc64a41564349853f9a6b0f680`.
The only conflict was independent appends in lessons-testing-evidence.md; both
were retained. Range-diff shows 20 identical patches and only changed context
for that documentation patch. Workflows domain/UI and shared input validation
are byte-identical to the prior head. New dev's Notes code and generated styles
are preserved. Fresh targeted integration tests and all preflight guards are
required on this combination before the leased push; exact-head GitHub checks
and refreshed Qodo review remain required before merge.

The new-base integration selection had **311 passed, 1 failed** in 140.52s.
The sole failure was the unchanged startup CSS guard: **768,490 / 768,000 bytes**
after Notes added 1,066 boot-parsed bytes. Selector count remained 272/274 and
all seven preflight guards passed. This is a combined-tree integration failure,
not waived as an unrelated baseline issue.

Six geometry selectors now qualify their existing widget types with the IDs
already assigned by WorkflowsScreen.compose: workflows-editor, workflows-library
and workflows-navigator. The root panes and library/navigator child rules retain
every declaration. The existing conservative splitter can now identify their
single route owner and move them into screen_feature_workflows.tcss. Consumer
search found only those production compose sites; no class, loader, new sheet,
dependency, style value or budget constant was added or changed. Generated files
were rebuilt through build_css.py, not edited. The unchanged byte guard supplies
the failing regression; real route-entry, compact/editor and cascade checks
qualify the specificity/loading change.

Corrected-tree verification: **182 passed**, 161.10s, covering actual app
authoring/file pickers, editor/paging/projection, both route-entry paths, compact
layout/focus, CSS partition/cascade, token governance and import/CSS performance.
The unchanged byte guard now measures **767,878 / 768,000 bytes** (612 fewer);
the broad-selector guard measures **271 / 274**. All seven preflight guards
pass again, and diff-check passes. Existing dependency, serial-resource fallback,
deprecation and unrelated old pytest cleanup warnings remain disclosed. No full
sweep or model call occurred. Scoped independent review and new-head GitHub
gates remain the final prerequisites for push/merge respectively.

The scoped independent reviewer found no Critical/Important/Minor issues and
approved pushing. It independently checked declaration identity, sole production
ID owners, loading before construction, exact generated partition, the boot-byte
census and mounted-fixture style/geometry parity at 160x48, 110x36 and 60x20.
That fixture check supplements the coordinator's full-app tests; it is not
misrepresented as another live-app UAT. New-head GitHub review/CI remains pending.

## Creation and public-contract review (2026-09-16 UTC)

Qodo review5217985076 on92723d5045 published five new findings:
4022051681 (creation-name boundary), 4022051690 (capture SVG dependency),
4022051696 (paging event annotation), 4022051704 (catalog API docs) and
4022051712 (authoring/exchange API docs). Earlier issues remain resolved or
dismissed; merge is held for these new findings.

The initial targeted regression run had **8 failed, 8 passed**: oversized names
were persisted, non-text values raised incidental errors, an invalid creation
replaced the current draft, and unavailable SVG support failed without actionable
installation guidance. A shared Pydantic creation boundary now admits at most256
raw Python characters, checks before trimming/storage setup, trims surrounding
whitespace and rejects blank/non-text values. Plain validation preserves existing
Unicode including lone surrogates. Imports/raw definitions retain their original
name/admission contracts; a 1,024-character imported name remains intact.

The first correction passed16 focused cases, but the new full-app test exposed
multiline Pydantic diagnostics hiding the actual limit in the two-row status
area. A concise content-free shared wrapper fixes the painted message; that test
also creates a valid workflow afterward. A separate failing real-store test
exposed the standalone controller's creation fallback accepting oversized names.
It now uses the same helper, preserving its InvalidDraft error category and the
current draft. The final focused selection passes18 cases. An internal type-error
lint finding was corrected without changing the wrapper's public ValueError
contract; the same focused cases were rerun afterward.

Capture now consults ensure_svg_rendering before importing Cairo packages, and
reports the svg extra plus native Cairo requirement via ImportError (the existing
optional-dependency mechanism has no separate project-wide dependency exception).
The regression supplies the unavailable dependency boundary and prevents either
raw import, verifying the real capture entry's guidance. No capture subprocess or
live profile/model/network was started by this test.

The paging callback has its concrete event and None annotations. Catalog and
authoring/exchange docstrings now describe actual inputs, results, validation,
file/storage errors, and retained accepted work. Eight touched Python files pass
format checks; seven pass whole-file Ruff. The shared input module retains eight
baseline diagnostics, matched by unchanged source spans, code, message and start/
end columns, with zero unmatched. No suppressions or baseline cleanup occurred.
All seven preflight guards and13 import/CSS checks pass (27.60s); boot CSS remains
767878/768000 and broad selectors271/274. Full targeted and independent-review
results remain to be recorded before push.

Final broad targeted selection: **539 passed**, 181.53s, covering Workflows,
authoring storage, editor/paging/projection/route loading and shared reasoning/
character validation consumers. The final internal TypeError/concise-wrapper
adjustment was separately reverified with all18 new focused cases (5.68s).
Independent review found no Critical/Important/Minor findings, checked the final
production boundary and retained controller exception category, and cleared
commit/push. No new architecture, model calls, full sweep or baseline cleanup.
Exact-head review/checks and merge remain pending; AC7 is not yet complete.

## Creation-name portability correction (2026-09-16 UTC)

Qodo review5218124022 on acbbf6b3 raised comment4022157035: the preceding
creation boundary admitted surrogate code points which cannot be bound as server
database text. Verified the cited server d9c245ac against current server dev
59049e094e0845a4611ea725ae19b7c1754ea709: workflow schema and database code are
unchanged, and the endpoint's intervening tenant changes still pass name=body.name
directly. A real in-memory SQLite probe rejected U+D800 and U+DFFF with
UnicodeEncodeError while accepting U+1F600. No server writes or model calls.

The regression selection initially had **7 failed, 18 passed** (5.60s): all six
surrogate creation inputs were accepted, as was the standalone-controller case.
Strict UTF-8 encoding now rejects them in the existing shared creation validator,
after the raw 256-character bound but before trimming or storage setup. The
public wrapper supplies concise, input-free valid-Unicode guidance and retains
ValueError/InvalidDraft contracts. No Unicode replacement or normalization occurs.
This supersedes the prior round's new-name surrogate acceptance, not its lossless
import/raw-definition contract. ADR-138 and the guide explicitly distinguish them.

The corrected focused selection has **25 passed** (4.85s). Cases cover the
surrogate range's endpoints, embedded surrogates and an uncombined pair, unopened
storage, preserved pending drafts, 256 emoji, composed/decomposed valid Unicode,
trimming, imported legacy names and actual-app invalid-name feedback/valid retry.
Both touched Python files pass format checks; the test file passes Ruff. Shared
input-validation retains eight exactly source-attributed baseline diagnostics,
with zero unmatched findings. All seven preflight guards pass.

Independent review found no Critical/Important/Minor findings, separately probing
every surrogate, limit-before-encoding, error categories, valid Unicode and
unchanged raw/legacy search. This is a creation-boundary correction only; no
runtime, schema, synchronization or SQLite infrastructure was added. Broad
targeted verification and exact-head GitHub review/checks remain required.

The final broad targeted run passed **547 tests** in 164.30s, including Workflows,
real authoring storage, editor/paging/projection/route entry and shared validation
consumers. Existing dependency and unrelated old pytest-cleanup warnings remain
disclosed; no warning suppression or baseline cleanup. No full-suite run.

All **13 startup/import/CSS checks** also pass (23.52s). Boot CSS remains
767878/768000 bytes; broad selectors remain 271/274. Existing headroom warnings,
joblib serial fallback and datetime deprecation remain unchanged. The reviewed
tree is ready for push; exact-head Qodo and GitHub gates still precede merge.
