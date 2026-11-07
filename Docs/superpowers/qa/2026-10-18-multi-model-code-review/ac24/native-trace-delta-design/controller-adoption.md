# Controller adoption and implementation boundary

Root adopts the focused design under existing ADR097/063/179 and task ADR240.
This repairs the proven native tool-loop capture gap, with no change to provider
input, checkpoint V3, tool authority or review-job retention/recovery.

Use existing owners and transaction/capability machinery. Runtime loop and current
checkpoint semantics remain unchanged unless a source-proven prerequisite requires
root assessment first. A capture-only typed proof may reuse the existing invocation
custody and projection/sealing provenance; it must never equate raw and durable
checkpoints by ignoring sensitive fields. Retain only bounded current proof state,
not a new registry or accumulated raw checkpoint history.

One exact current continuation node is replaced while exact prepared message
suffix rows are appended. Other groups, ordinals, source/run/policy/destination,
private prefix and call/result pairing stay exact. Same transaction and final CAS
protect the whole operation; existing completed-turn/plain/retry behavior stays.

Store only the newly captured sanitized round plus bounded checkpoint metadata,
using the existing single-node replacement edge as predecessor. A distinct trace-
only versioned tag must never reinterpret user JSON or legacy full artifacts.
All readers/comparisons/archive/fork/GC paths must resolve it consistently. Prove
trace-owned byte growth from actual artifacts and metadata, not compression ratios.

Cold proof must be either re-established through existing explicit host custody or
refused before provider entry. No inferred ownership from matching bytes, silent
capture downgrade, automatic replay or new durable review-job owner. No new SQL
DDL is planned; audit enum/codec/export constraints and report any migration need
before editing schema. Existing checkpoint and trace payload/count/depth limits apply.

Implement as one focused integration task with TDD, strict negatives, actual normal
Console CaptureOn2/3-round/multicall completion, atomic/race/privacy/growth/retention
coverage and independent review. Earlier endpoint-only REDs are the integration
oracle. No full sweep or live API call: actual105spent/5financialunused/1childunused,
accepted-role replay forbidden by current scope.


Ruling61 source-proven refinement: raw-to-projected provenance is discarded before
the current callback, so add the optional capture-only runtime acknowledgment
after successful projected persistence. It carries exact invocation raw/projected
relation to bounded current adapter custody; store projection/sealing/final CAS
remains authoritative. Ordinary/CaptureOff paths keep the default absent hook.
No raw values in generic events/DB/kwargs/logs; no historical registry. Missing,
failed or stale acknowledgment cannot authorize the witness. This prerequisite is
recorded in ADR097/plan/task before runtime edits.
