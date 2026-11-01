# TASK-33560 performance and safety evidence — ready for independent review

The final actual-app pair reduces native idle opens from **884 to 286** in ten
seconds: **67.65% lower opens/s**. Source is frozen for review; no task closure,
commit or publication is claimed here. ADR required: yes; existing
[ADR-126](../../../../backlog/decisions/126-complete-local-backup-and-recovery.md)
contains the prospective policy and this measured checkpoint.

## Actual unchanged-work pair

Both runs use the exact frozen `paired_probe_source.txt` (SHA-256
`da5ba790b5d59549bab2345f2418a98dc47a24fa6c77ab58e251039c4d94539f`), normal
profile enrollment/binding before app import, `TldwCli.run_test` and the real
ChatScreen. UI is ready, real boot workers drain, then two seconds of settlement
precede ten seconds of real idle. The app's actual profile hold retains eight
leases at both idle edges; no synthetic owner or reduced timer work is used.
HOME, XDG roots and config are fresh private directories; network connections
are rejected, model refresh is disabled and keyring is null. Pytest uses a fresh
outer profile, repo cwd/PYTHONPATH and default fixture directory depth.

| Measurement | Before | Final after |
| --- | ---: | ---: |
| Idle seconds | 10.001132 | 10.000684 |
| Native opens | 884 | 286 |
| Native opens/s | 88.389998 | 28.598044 |
| Python opens | 34 | 30 |
| Actual native pause probes | 10 | 10 |
| Credential polls | 40 | 40 |
| Real retained ordinary leases | 8 | 8 |
| Boot native opens to UI ready | 41034 | 16571 |
| UI readiness seconds | 3.514570 | 3.646242 |
| Settlement native opens | 33689 | 15206 |

Boot counts decrease, but these single-run UI timings do **not** establish a
startup-speed improvement. Native idle attribution is included in raw JSON:
pinned-directory 492→130, private-path 315→122, admission 46→20, bootstrap
metadata 24→7, SQLite subprocess six→six and Console one→one. The caller trace
located the remaining costs in real admitted cleanup/selection paths and fresh
SQLite parent checks. SQLite parents, activation permission checks and native
Admission directory qualification remain fresh. The unimplemented extra
Admission root proposal/test was dropped when the measured target was met.

The final correction moves child stamp construction outside the coordinator
lock; an explicit positive and deliberate-under-lock negative control verifies
it. Confirmed-directory proof requires two independently positive parent
containment derivations, complete selector/group evidence, full current child
chain/leaf observations and counted-before-final-check ordering. File-only
admission cannot prove its parent.

## Safety and pause boundaries

The hold-owned bounded positive caches reuse existing `_Evidence` with epoch,
complete posture/content stamps, the original one-second settle margin and two
full bracketed derivations. Whole MCP witness derivations include all current
roots and consulted historical path-token chains, every control/selector input
and activation generations/`required.json`. Unknown/symlink/absence/pending or
unqualified cases take the original path. Returned metadata is copied.

Native registry locks, gate opens/identity/flocks and actual per-operation parent
FDs retain their original custody. Pin reuse opens a fresh native FD, checks full
identity/type/uid/mode and re-observes the complete path after allocation. A
positively closed identity mismatch falls back to the original walker. Both
before-close and after-close uncertainty controls retain descriptors, pins and
leases, block drain and prevent maintenance entry. Publication-race controls
preserve foreign bytes. Concurrent earlier cold publication cannot demote
confirmed evidence; filesystem work remains outside the coordinator lock.

The already owner-approved **1 Hz** runtime monitor is unchanged. A real
native-maintenance request started 20 ms after a genuine scheduled probe was
noticed in **0.976281 s**, began actual local pause in **1.069559 s** and entered
exclusive maintenance in **1.108681 s**, then normally resumed with no app
refusal. This was a separate stimulated pause checkpoint before the final lock
placement/concurrent-publication corrections; its busy-window open count is
**not** idle qualification. The interval and native protocol did not change.

## Targeted checks and limits

- Required new controls: **52 passed**, zero failures/errors/skips, 12.642 s XML
  (14.215 s wrapper). Exact source module and receipt are archived.
- Later persisted permission parse-pause control: **two passed** (reuse on/off),
  comparing exact `RecoveryRequired("storage_locally_paused")` and unchanged
  bytes. This confirms conservation of the existing refusal; it does not claim
  nested permission RMW completion across pause.
- Related ordering/lifetime selection remains **non-green: six passed, one
  failed**. Its False seed equals the fresh default, so no document is written
  and its `json.loads` hook does not run. An attempted persisted-document finish
  also failed: unchanged raw binding validation starts a fresh acquisition after
  pause. Hash-bound unchanged seams and excerpts are in the archive. No source
  expansion or blanket old suite rerun was made for this separate behavior.
- New test/probe Ruff lint and formatting pass. Changed production function
  ranges were formatted; difference-aware lint retains exactly **22 prior
  diagnostics**, matching base code/message/source lines. `git diff --check`
  passes. Whole-file/static-suite cleanliness is not claimed.
- Existing cache-write sandbox warnings and foreign late-cleanup warnings remain
  in raw logs. No foreign directories were cleaned. App logs retain the earlier
  Evals package-default enrollment refusal and deferred Buddy migration warning.
- Original completed PR budgets/suites are not renewed. Results qualify this
  isolated Darwin/Python 3.12 native environment and targeted paths; they are not
  broader cross-platform, whole capture/restore or transport certificates.

## Complete retained history

Every task-owned run's raw log, receipt and available XML/JSON is preserved in
[raw-evidence.zip](raw-evidence.zip), with hashes and original paths in its
`index.json`. The archive also contains exact changed source snapshots, the
production diff, stimulus scripts and static/unchanged-seam receipts. It omits
profile databases and unrelated repository/base snapshots.

| Run / checkpoint | Result and permitted claim |
| --- | --- |
| Native MCP microprobe | Call-through setup evidence only; no app idle claim |
| First unbound app / duplicate after denied edit | 123 native opens each; no retained owner; setup only |
| First bound enrollment | Setup fails native qualification on `/var` alias; root resolved normally afterward |
| Actual retained-owner baseline | 884 native opens; authoritative pre-production baseline |
| First after | 574 opens, **35.06% reduction: fails AC4** |
| Remaining-open trace | Authority/caller attribution only; unchanged SQLite/native owners kept fresh |
| Descendant after checkpoint | 286 opens, 67.64% rate reduction; threshold passes |
| Actual pause | Latency/drain stimulus only; open rate excluded from idle claim |
| Final exact-script after | 286 opens, 67.65% rate reduction; final unchanged-work result |
| Initial warm RED | Eight cases: seven expected failures, one copy control passes |
| First/second green setup | Two collection failures from misplaced decorator; preserved |
| First warm/mutation green | 32 pass |
| Abandoned Admission root RED | One failure; no production root optimization |
| Historical/activation/companion extra RED | Five pass, one fresh-history warm failure |
| Descendant and pin controls | One then eight pass |
| First custody/lock controls | Two pass, two fixture failures (drain checked after pause) |
| Corrected custody controls | Three pass; production unchanged for fixture repair |
| Final required safety module | 52 pass |
| Related ordering selection | Six pass, one default-False seed failure; remains non-green |
| Attempted persisted RMW finish | One failure at unchanged original post-pause binding acquisition |
| First original-verdict controls | Two assertion-API failures (`reason` attribute does not exist) |
| Corrected original-verdict controls | Two pass, exact original message/byte equality |

Portable review: verify [manifest.json](manifest.json), unzip into an empty
folder and inspect `index.json`. The maintained `boot_idle_probe.py` is formatted;
`paired_probe_source.txt` retains the exact measured bytes and repo-relative
location. Run it only for new justified work; no further census is needed for
this checkpoint. Archived pytest/pause launcher scripts bind their original
absolute repo/interpreter paths, which must be adjusted when replayed elsewhere.
