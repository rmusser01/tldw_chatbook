# Console performance fixes

The user requests verification and fixes for every measured pause cause, investigation and repair of broader performance defects, and one combined PR against dev. PR #3017 remains open; this branch includes its functional fixes and the native investigation evidence.

Preserve native storage ownership, maintenance exclusion, trace capture, cancellation custody, scope/revision fencing, and operation-owned connection retirement. No full-suite sweep is authorized. Do not use longer waits, suppressed timers, or relaxed privacy checks as fixes.

1. Make unchanged control-bar height synchronization idempotent while correcting real class/style mutations and rendering the recovery row at its original geometry.
2. Batch finite character metadata reads and remove redundant availability reads within the existing operation lifetime. Every current authority/ambient and revision check must still run; no stale result may cross profile/session/character changes.
3. Reduce UI refresh database fan-out using existing tick snapshots and worker publication fences. Keep live run/status changes responsive. Investigate startup trace GC sweeping an admitted revision and fix any confirmed race without retaining orphan traces indefinitely.
4. Qualify Windows storage evidence using the shim's real NTFS change time and fresh ACL/owner/path stamps, with differential full-derivation oracles. Investigate and fix the elevated native runner's SQLite sidecar owner refusal while retaining refusal for foreign and shared objects. The ordinary user path must remain correct.
5. Measure a complete three-message captured conversation on native Windows, Linux and macOS with normal refresh competition. Record native handle and helper costs and heartbeat latency. Expand the bounded performance check to detect these costs, preserving separate platform receipts and source provenance.

ADR required: yes for Windows evidence/ownership and any trace GC contract change; amend existing ADR-126 and ADR-097 rather than duplicate their boundaries. Routine UI batching/layout changes require no new ADR if all existing interfaces and authority fences remain intact.
