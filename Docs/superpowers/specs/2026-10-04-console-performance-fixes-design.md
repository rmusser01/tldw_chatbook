# Console performance fixes

The user requests verification and fixes for every measured pause cause, investigation and repair of broader performance defects, and one combined PR against dev. PR #3017 remains open; this branch includes its functional fixes and the native investigation evidence.

Preserve native storage ownership, maintenance exclusion, trace capture, cancellation custody, scope/revision fencing, and operation-owned connection retirement. No full-suite sweep is authorized. Do not use longer waits, suppressed timers, or relaxed privacy checks as fixes.

1. Make unchanged control-bar height synchronization idempotent while correcting real class/style mutations and rendering the recovery row at its original geometry.
2. Batch finite character metadata reads and remove redundant availability reads within the existing operation lifetime. Every current authority/ambient and revision check must still run; no stale result may cross profile/session/character changes.
3. Reduce UI refresh database fan-out using existing tick snapshots and worker publication fences. Keep live run/status changes responsive. Investigate startup trace GC sweeping an admitted revision and fix any confirmed race without retaining orphan traces indefinitely.
4. Qualify Windows storage evidence using the shim's real NTFS change time and fresh ACL/owner/path stamps, with differential full-derivation oracles. Investigate and fix the elevated native runner's SQLite sidecar owner refusal while retaining refusal for foreign and shared objects. The ordinary user path must remain correct.
5. Measure a complete three-message captured conversation on native Windows, Linux and macOS with normal refresh competition. Record native handle and helper costs and heartbeat latency. Expand the bounded performance check to detect these costs, preserving separate platform receipts and source provenance.

ADR required: yes for Windows evidence/ownership and any trace GC contract change; amend existing ADR-126 and ADR-097 rather than duplicate their boundaries. Routine UI batching/layout changes require no new ADR if all existing interfaces and authority fences remain intact.


## Existing provider audit bugs included in the user's final scope
The instruction to verify and fix each identified issue also includes the five known response-contract failures archived in Docs/superpowers/qa/2026-10-04-provider-response-audit/audit.md, not just Console pause findings. Complete TASK-34367.1 through TASK-34367.5: Groq metadata/usage/errors, OpenRouter choice annotations, Together and Fireworks documented annotations, and Cerebras timing metadata/logprobs. Use real adapter complete/SSE replay at the HTTP boundary, retain strict unknown/malformed rejection, and preserve streamed usage/error semantics. Follow existing ADR-179 provider profiles and ADR-062/063 parsing contracts; no blanket permissive shared parser. These are existing atomic tasks included in the combined PR, not newly invented future work.

## Native bootstrap creation retry correction (TASK-34404)

A real post-mkdir Windows ACL denial established that a retry can inherit a created
but unsynchronized ancestor before any bootstrap pending/registry record exists.
Use a finite private per-entry creation intent in its existing pinned parent,
with caller-derived lexical child and actual parent identity. Synchronize the
intent before mkdir, hold its native exclusive file lock through the required
entry barrier and intent removal barrier, and revalidate exact intent names along
the requested chain on retry. Failed, corrupt, foreign, substituted or live
creation evidence refuses before lower creation or pending publication. Never
infer required durability from private ownership or synchronize unrelated protected
ancestors. Preserve existing native flags-zero metadata/device synchronization.

ADR required: yes. ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: exact durable initialization/retry evidence closes the native barrier gap
introduced by removing the original overbroad publication ancestor walk.
