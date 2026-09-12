# ADR-157: Per-definition child wall-time caps

Status: Superseded by ADR-158 for migration sequencing; policy retained
Date: 2026-09-12
Task: TASK-13154.7
Related decisions: ADR-131 (durable accounting), ADR-134 (fleet and automatic budgets), ADR-135 (delivery/recovery)

## Decision

A named AgentDefinition may carry optional finite positive max_wall_seconds. NULL means no additional restriction and preserves existing definitions. Schema v20 adds the nullable REAL field after recovery's v19 migration. Canonical Settings validates and saves it through definition CRUD; invalid values never overwrite stored policy. Booleans, malformed/nonfinite values and nonpositive values are rejected; positive fractional seconds are supported.

Spawn reads the definition from the turn's frozen roster and takes the minimum of this cap and the existing applicable child budget. The minimum is applied after the legacy helper's floor so an explicit subsecond cap is not enlarged. Inline children retain their parent-remainder policy; threaded survivors retain the existing independent global child ceiling. Automatic elapsed-chain checks and cooperative cancellation remain separate authorities and cannot be widened or reset by a definition.

Continuation keeps its existing current-roster re-resolution for definition instructions, tools and model. Once a lineage uses a definition cap, its admitted per-run maximum is retained in process-local coordinator metadata. Each continuation gets a fresh per-run allowance bounded by that retained maximum, the new baseline and any current definition cap. Raising/removing the definition cap cannot enlarge an already capped lineage. Generic and never-capped lineages retain existing behavior. Retention remains ephemeral and does not authorize restart resurrection.

Definition fingerprints include normalized numeric caps when present. The canonical absent encoding omits the new key, preserving existing uncapped hashes exactly. Historical hashes are not rewritten; persisted run budget records the actual admitted bound even when current definition identity has a higher cap.

## Alternatives and consequences

Treating this field as an override could widen the global/parent/automatic bounds. Re-resolving it from the DB during spawn would violate frozen roster identity. Reapplying only a changed definition on continuation would let policy removal enlarge already admitted lineages. A shared cumulative lineage deadline would be a different product policy from a per-child/run cap and conflicts with existing explicit continuation semantics.

No arbitrary 1-second floor is added to explicit caps. Existing no-cap defaults and historical fingerprint identity remain stable. Runtime timing retains current human-input tool-clock pauses, automatic elapsed deadlines, and cooperative cancellation limitations: a terminal result never proves a worker or remote side effect stopped. Existing physical resource leases remain occupied until actual cleanup. Accepted ADR-134/135 texts are not modified by this supplemental decision.
