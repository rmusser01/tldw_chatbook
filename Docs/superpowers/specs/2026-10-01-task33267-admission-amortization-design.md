# TASK-33267: amortize ordinary recovery admission

Status: proposed ADR-126 amendment, awaiting owner approval. Scope is the
existing Python backup/recovery admission path, its maintenance monitor and
guarded MCP JSON reads. No new dependency, subsystem or authority.

## Decision

Reuse verified directory descriptors and immutable parsed control records only
while the existing native `_Hold` is alive. A cache entry never grants authority.
Before dependent I/O, check current pathname-to-object bindings, every verified
ancestor, ownership and applicable permissions/ACLs, source selection, native
gates, current registry publication and pending-recovery intent. Fail closed on
uncertainty; retain existing platform qualification and retirement rules.

This replaces repeated directory opens and repeated parsing, not freshness
checks. Reuse a single checked registry observation within an operation. Start
with bounded current-byte comparison before reusing a parsed record: inode,
mtime, process epoch, TTL or a watcher alone cannot authorize reuse.

## Alternatives

- Keep current checks unchanged: safe but retains repeated disk work and global
  serialization.
- Use connection-lifetime admission or metadata-only caches: smaller, but misses
  path replacement, permission changes or externally edited policy. Rejected.
- Reuse existing holds with fresh barriers: selected approach; no new persistent
  generation counter or control-writer protocol.

## Ownership and concurrency

Evidence is scoped to the actual native hold, PID, exact bootstrap/config/source
selection and namespace group. Retain root-to-leaf predecessor edges, not just
the last directory descriptor. Compare each current child name under its held
parent to the held child's identity and validate current security properties.
Preserve trusted-link rules and hop limits wherever that seam permits links.
Use the native Windows identity, ACL and reparse checks, not POSIX substitutes.

Keep first-use initialization single-flight. Warm borrowers reserve a live
generation under brief coordinator bookkeeping, validate outside the global
mutex, then recheck liveness and selection before I/O. No filesystem validation,
native waiting or unrelated DB transaction runs under that global mutex.
Same-source native coordination still applies. Closing fences new borrowers;
accepted operations and native resources retire positively before descriptors
close. Failed retirement remains visible. Fork, source/group change, pause,
maintenance and failed validation invalidate reuse.

Capture, preview, restore, publication and arbitrary materializer paths retain
their existing fully checked paths in this amendment.

## Monitor and MCP reads

Replace the app's 10 Hz monitor with a native cross-process probe at most one
second apart, plus immediate relevant local lifecycle probes. Every operation
still has its admission barriers. Local events alone cannot observe another
process's flock/LockFileEx contention. Preserve cancellation joining, error
propagation and all existing child/maintenance deadlines. A short maintenance
attempt may still refuse and request owner closure; do not extend its deadline.

Reuse parsed immutable payloads for the four MCP JSON stores after a fresh
bounded byte comparison. Preserve current missing/corrupt/unreadable behavior,
guarded mutation fences and detached return values. Never serve cached permissive
policy over a current read failure. Execution-log rotation/append caching and
credential/backend discovery are outside this change.

## Verification and acceptance

Measure the current source and the historical 840ed2ca58 probe before changing
the path. Compare identical inputs and complete admission boundaries, including
native handle/ACL work on Windows. The existing <0.5 ms transaction and 80% boot
open-reduction criteria remain unproven until measured; unmet goals do not
justify removing checks or claiming completion.

Interpret unrelated-owner concurrency as no global mutex across validation or
transactions, while retaining brief bookkeeping and native registry/alias
coordination. Interpret warm MCP cache as parse reuse with current-byte
freshness, not zero-I/O last-good policy. Native probing remains bounded even
though 10 Hz polling ends.

Add adversarial directory/ancestor rename and replacement, ownership/mode/ACL
change with held descriptors, control replacement/new pending intent,
same-inode record edits, selection/generation changes, fork/pause/retirement
races and unrelated-owner concurrency checks. Verify monitor latency and MCP
read-failure/cache behavior. Use targeted checks and native macOS/Linux/Windows
evidence, scoped Bandit and independent review. No full suite or deadline change.

Read-only independent design review:
`/private/tmp/task33267-admission-design-review-wMGTxE/report.md`.
That review supplies static caller/path evidence, not runtime qualification or
current performance measurements.
