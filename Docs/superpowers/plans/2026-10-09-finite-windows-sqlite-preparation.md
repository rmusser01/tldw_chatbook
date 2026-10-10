# Finite Windows SQLite artifact preparation

Task: TASK-34601, AC19, OPT95. Root owns integration and every native run.

ADR required: yes, an amendment to the existing boundary.
ADR path: backlog/decisions/125-lock-safe-private-sqlite-validation.md.
Reason: share native parent preparation across the fixed SQLite artifact inventory,
with explicit mutation, current-name and retirement boundaries. Preserve ADR029,
ADR126 and ADR225; no connection pool or changed save/checkpoint policy.

## Evidence and selected scope

The original sqlite-setup-detail-1 diagnostic has three complete saved turns,
42 stages, unchanged product source and no detected native overlap. All888 setup
records pair with no cap/errors/unmatched/unfinished rows. Warm setup counts24/17
spend .408/.086s on main preparation and .778/.266s on sidecars, against only
.0295/.0106s on actual SQLite open. Directory verification adds .257/.157s.
These are inclusive concurrent work, not predicted whole-Send savings. OPT94
checkpoint-policy changes remain deferred; no long checkpoint stall reproduced.

Keep admission, initial trusted-directory verification, helper envelope/deadline,
expected identity and final sqlite3.connect ordering. On the qualified stock
ordinary Windows route, one finite preparation owns a verified parent for the
database and its existing three sidecars. Existing generation logic borrows it
only for a no-mutation first attempt. It retains every relative leaf identity,
type, owner, link, mode, writable-open and postcondition check. Required creation
or hardening uses the original fresh-parent route before the effect. Optional
churn retries use fresh parents and retain the original total four-attempt bound.
Do not restart the whole inventory and reset a member's retry budget.

Before SQLite's pathname open, perform a fresh full named-chain verification and
tie its parent identity to the still-held original parent. This does not claim to
eliminate the existing SQLite pathname-open race. Windows share-delete handles
alone are not current-path evidence. Every descriptor must retire before native
SQLite allocation; failure/uncertainty refuses allocation and retains the exact
ordinary admission/custody until process exit, without retrying a recycled FD.

The stock Characters factory is the existing _QuiescentSQLiteConnection, not
sqlite3.Connection. Qualify its exact defining class and constructor/close bodies
through a small resident record; custom subclasses or changed bodies retain the
original route. Do not import the core owner from this low-level seam.

Public APIs stay unchanged. Capture, descriptor, custom-factory, custom/replaced
preparation and source-pin-job routes keep their original behavior. The private
per-call preparation outcome belongs to the existing admission wrapper; no
global cache, new registry, generic job engine, lease transfer or retained executor
connection. Use compact source qualification only for the exact skipped seams.

## Parallel ownership and integrated verification

1. Shared preparation lane owns tldw_chatbook/DB/private_sqlite.py and only a
   small definition-time stock-factory record in DB/base_db.py. Prepare
   the smallest explicit borrowed-parent/outcome candidate outside the worktree
   first; root applies it only after original-source regression evidence.
2. Baseline verification lane owns Tests/DB/test_windows_sqlite_preparation_batch.py:
   real stock SQLite rows, original preparation/parent and native work counts,
   one-attempt/later-attempt distinction, actual closed descriptors and retired
   admission. Source-only construction; root alone executes original RED.
3. Controller integration lane owns Tests/DB/test_windows_sqlite_preparation_edges.py:
   actual parent rename/replacement and ACL changes, mutation fallback before
   effects, bounded optional-sidecar churn, custom compatibility and uncertain
   close refusal/custody. Use original native bodies, not stubs claiming retirement.
4. Root reviews the candidate/controls, integrates both implementation and tests,
   runs the combined targeted controls and relevant existing private-SQLite,
   ownership and Console consumers, then sequential quiet baseline/candidate
   comparisons. No full suite, timeout changes or overlapping native timings.
   Record work reduction separately, retain negative samples, and adopt only if
   the resulting complexity and qualified latency benefit justify the change.

## Review gate

The exact private outcome and generation fallback shape must pass source review
before product edits. New uncertain-preparation cleanup must not fall through the
current pre-constructor exception branch that releases admission. Baseline and
candidate checks must prove real bodies/effects and positive physical retirement,
not absence of observed events. The overall one-second goal remains open.

## Qualification result: not adopted (2026-10-09)

The candidate was implemented, independently reviewed and tested, then removed
because the measured whole-Send benefit did not justify its added ownership and
source-qualification machinery. The original four-parent-walk implementation
remains. No pool, retained connection, checkpoint policy or timeout change was
introduced. The one-second Send target remains open.

Original-source RED: all four stock-factory/sidecar cases reach their real data,
native retirement and lease-retirement assertions, then fail only at four walks
versus the proposed two. The candidate reduces parent walks 4 to 2 and native
opens 67 to 51 with all sidecars present, or 55 to 39 with optional files absent,
for both sqlite3.Connection and _QuiescentSQLiteConnection. Returned CRT-FD
counts reduce 48 to 32 and 42 to 26 respectively; all observed returned FDs
physically retire. These are per-setup counts, not distinct content-file counts.

The combined native selection first reports 50 passes, six genuine unavailable
owner-privilege skips and five edge-fixture failures. The fixtures had attempted
unsupported WindowsOS.chmod modes, used a held-target replace fault, and required
an open in a deliberately preallocation refusal. Corrected real-DACL/unlink
fixtures preserve exact native DACLs, surface injection errors separately, and
require zero opens for the reserved-keyword case. All 14 new cases then pass;
55 distinct cases qualify across the unchanged product and final fixtures.
The uncertain-close test protects an actual native parent HANDLE, proves the
other parent physically closed, refuses SQLite allocation and retains the exact
admission until its isolated child exits. No product fix was needed after the
first integrated run. Six platform/privilege cases remain unqualified locally.

The four predefined quiet runs retain all twelve saved and settled turns:

| Run, in execution order | First Send (s) | Second Send (s) | Third Send (s) |
| --- | ---: | ---: | ---: |
| sqlite-preparation-a1 | 5.094655 | 2.814068 | 2.463775 |
| sqlite-preparation-b1 | 5.259237 | 3.206207 | 3.279363 |
| sqlite-preparation-b2 | 4.622078 | 3.299377 | 3.109563 |
| sqlite-preparation-a2 | 5.360506 | 3.947414 | 2.896135 |

Warm means are 3.030348s baseline versus
3.223628s candidate (descriptively 6.38% slower).
Cold means are 5.227580s versus
4.940657s. The small sample and baseline variation do not
establish a causal slowdown, but they do not establish the required gain either.
Do not discard unfavorable samples or retain this complexity on work count alone.

All runs have unchanged source/HEAD, no detected native overlap, equal loaded
module membership and exactly the two intended normalized product differences.
Default stage/heartbeat observations remain; there is no detailed sampler,
physical-terminal frame, percentile or idle-host claim. All runs save three
user/assistant pairs, three completed trace links and no pending checkpoint.

Compile checks and both new files' Ruff lint/format pass. The two existing
product files retain the same 13 baseline Ruff findings, compared by rule,
message and exact source line; whole-file lint/format are not clean. Independent
source review found no remaining custody/source blocker after corrections.

Receipts: sqlite-preparation-red-2, sqlite-preparation-integrated-1,
sqlite-preparation-edges-2 and sqlite-preparation-{a1,b1,b2,a2} under the
integration owner's claude-watch-final-gate-review artifact directory. Analysis:
sqlite-preparation-comparison.json. Exact tested product and tests are archived
in sqlite-preparation-qualified-rejected/source-identity.json and its four source
files; private_sqlite SHA9576fd9291315f7de2cf2ac455effc1d8ddbf7c789d3657c344fff09a8a4086c.
No experimental product or test file remains in the active worktree.
