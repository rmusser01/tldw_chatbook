# Finite MCP source admission

Task: TASK-34601 AC20; OPT76. Root owns integration and every native run.

ADR required: yes, refine the existing admission boundary.
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: consolidate repeated fresh observations within one synchronous admission,
explicitly changing when a concurrent recovery change is refused. Preserve ADR225.

## Evidence and boundary

The original permission-source-detail-1 diagnostic records four full recovery
observations during each warm MCP scope admission, then independent readable and
final publication observations. The six loads all take the default branch;
none reads or parses permission bytes. Native metadata primitive attribution
does not justify a volume-query rewrite. Consolidate the admission owner instead.

On the exact installed stock pending-preparation route, initial selection chooses
the target. Registration and pre-lock admission validate resident issued identity,
source/binding/config selection, closed/pause and existing lease ownership without
a second full recovery observation. Retain the existing source lock, member
admission, parent pinning and metadata-only preflight. One late full participant
source proof and the original parent proof must complete BEFORE state.active and
actor-local operation publication. Recheck issued attempt, participant/source,
binding, closed/pause and exact live lease custody after those proofs and before
yielding. Do not give proof callbacks a usable operation while proof is incomplete.

Keep the existing pending-preparation checker including execution_context; this
does not select deferred OPT97. Reuse current callback qualification and the
existing raw._check definition anchor for its known custom callback route. A custom
source lock or unsupported/replaced supported callback retains the original
sequence. Do not add a generic qualifier registry, transferable proof token,
cached witness, lease across awaits or new persistence owner. Internal observer
tests may follow the new boundary under ADR220/222's private relocation limit.

All actual reader/writer, default/corrupt recovery, backup/temporary publication
and Console final checked-read gates stay fresh. A concurrent recovery change
may now acquire the mutex or construct an inactive State before refusal; it must
not reach the body or payload effects. This intentionally revises the existing
pre-lock refusal assertion, not the effect or physical-retirement contract.

## Ownership and validation

1. Shared lane owns only raw_participants.py and prepares the smallest candidate
   outside the worktree. No product application before original regression.
2. Baseline lane owns a new test_raw_mcp_finite_admission.py: original empty-scope
   four-to-two witness/count regression for a real installed source, followed by
   a real read and positive native/lease retirement. Tests observe original code;
   they do not stub out source proof or filesystem work.
3. Controller lane owns test_raw_mcp_registration_boundaries.py and a separate
   test_raw_mcp_finite_admission_edges.py: inject real source/pending/parent change,
   pause/close and proof callback reentry; prove body/effect refusal and actual
   retirement. Keep immediate resident source mismatch refusal where applicable.
   Qualify custom lock/raw._check/recovery callback routes and final read freshness.
4. Root reviews plan/candidate and tests, runs original count RED, integrates both
   implementations, then runs focused new and existing source lifetime/coordinator,
   owned permission, nested scope, pause, native retirement and Console controls.
   Preserve assertions and deadlines; migrate observer location only where the
   documented boundary actually changed. Root owns other justified test migrations.
5. After integrated checks pass, run quiet full-profile baseline/candidate/candidate/
   baseline sequentially with fixed source, overlap guards and saved replies/traces.
   Report raw samples and native work separately. Retain only if benefit justifies
   the change; record negative results. No full sweep, timeout or budget changes.

## Review gate

Root and independent source reviews must resolve final resident custody checks,
custom fallback and no-effect-before-proof before application. No new public API.
The one-second application-overhead and100ms actual feedback goals remain open.

## Disposition: qualified experiment, not adopted

The candidate and its eight changed/new test files are archived in the external
review directory at finite-mcp-admission-qualified-rejected, with exact bytes,
manifest and patch. Original product/tests are restored. The qualified route was
the installed permission store; other stock sources lacked the resident lock
qualification and retained their original route. Net product growth was118 lines.

Original empty-scope RED observed four full witnesses, two acquisitions and349
native opens; the candidate observed two witnesses, two acquisitions and208 opens.
Both exercised a real read and proved descriptor/lease retirement. These are
scope counts, not whole-Send savings. Integrated controls qualify86 distinct cases:
36 in integrated-1 and50 lifetime cases, with the obsolete pre-State assertion
corrected under the explicit contract and rerun in pending-2 (14) and pending-4 (1).
The pending-record mutation stays at the original first-witness boundary; a
completion marker proves insertion occurred before refusal. Body/payload/default/
backup effects remain absent and actual resources retire. pending-3 is excluded
because its source changed during execution. No full sweep was run. All eight
test files pass Ruff; product retains only its preexisting E731. Diff check passes.

Quiet sequential full-default ABBA results (cold; warm1; warm2, seconds):

| Run | Baseline | Candidate |
| --- | --- | --- |
| A1 | 5.959775;3.378154;3.291164 | |
| B1 | | 3.871472;3.238077;2.580930 |
| B2 | | 6.542897;8.892869;8.050793 |
| A2 | 7.470477;6.941624;4.526565 | |

Warm means4.534377 baseline and5.690667 candidate; cold means6.715126 and5.207184.
All12 turns save/settle with3 complete traces per run and no pending checkpoint.
Source/HEAD are stable in each run, no native overlap is detected, loaded-module
membership agrees, and raw_participants.py is the only normalized product change.
Substantial variation in both arms prevents a causal slowdown claim, but the
additional machinery has not established a whole-Send benefit and is not retained.
No favorable retry, percentile, physical-feedback or subsecond acceptance claim.
Raw receipts are finite-mcp-admission-{a1,b1,b2,a2}; analysis and detailed stages
are finite-mcp-admission-comparison.json from analyze-finite-mcp-admission.py.
ADR126 records this unadopted alternative. TASK34601 remains In Progress.
