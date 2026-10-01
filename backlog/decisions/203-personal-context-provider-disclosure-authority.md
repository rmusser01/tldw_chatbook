# ADR-203: Separate Personal Context model disclosure from Sync and agent authority

Status: Accepted — design direction approved, 2026-09-25; no schema, permissions or runtime rollout approved
Date: 2026-09-25
Task: TASK-25907.7
Accepted future-design amendment to: [ADR-102](102-personal-context-profile-authority-sync-and-encryption.md)
Extends: [ADR-182](182-personal-context-memory-evolution.md), [ADR-201](201-versioned-profile-evidence-and-temporal-claims.md), [ADR-202](202-dependency-aware-personal-context-forgetting.md)
Related: [ADR-119](119-llamacpp-prompt-cache-snapshot-ownership.md), [ADR-147](147-agent-provider-routing.md), [ADR-080](080-trace-v2-exhaustive-event-projection-and-collaboration.md)
Design: [Provider disclosure controls](../../Docs/superpowers/specs/2026-09-25-personal-context-provider-disclosure-controls-design.md)

## Context

V1 canonical controls encode Sync mode and agent visibility. Native profile
context and tools can serialize visible device-only records without a resolved
model-disclosure audience. Adaptive interviews currently treat visible syncable
records as eligible and send prior turns. Summaries, child routes, replayed
history, captures/run logs and future embeddings have separate egress seams.
The unscoped quarantine existence flag may enter the model block. ADR-102's
literal device-only promise exceeds this observed model enforcement.

## Decision

Propose a restrictive separately versioned canonical disclosure ceiling plus
user-reviewed destination/purpose-bound native grants. Agent read authority,
Syncability, imported policy, a saved credential or a previous send grants no
model permission. V1 bytes remain unchanged. Extend unshipped ADR-201 V2 with
`deny`, `on_device_only` or `reviewed_destinations`; if it has shipped, use a
subsequent version rather than mutate canonical semantics. Unknown/absent policy
means deny. Portable opaque audience handles/purposes are ceilings, not runtime
capabilities. Grants are encrypted, peer-local and never synchronize/export. The
on-device-only policy still needs explicit local destination/purpose enrollment.
Only foreground user review can widen policy or enroll a grant; agent read/write
authority and scope promotion never widen audiences automatically.

Explicitly propose amending ADR-102's “never leave Chatbook” wording to distinguish
qualified Chatbook-owned local model execution and separately authorized owner
exports from automatic external disclosure. `device_only` remains a hard ceiling
against Sync and automatic remote model egress; a remote model grant cannot
bypass it. Changing Sync mode alone creates no model grant. `user_only` remains
excluded from every agent/model route. This proposed amendment grants nothing
today; current governance stays until a qualified rollout.

Native destination enrollment binds actual adapter/model, effective endpoint
and route/account identity, provenance, custody and configuration revision.
Display names and loopback/private addresses are not qualification. Local use
requires owned process/socket and no-forwarding adapter semantics. Unknown
custody and unresolved downstream aggregators are outside the first release.
Network consent names service custody, not a promise about its internal retention.
Shared core validates restrictive policy only; native runtimes own source
permissions, grants, authentication, transport and process qualification.

Apply audience filtering before ranking/overrides/token budgets/serialization.
Every governed registered derivative inherits the intersection of its inputs'
restrictions; no LLM paraphrase or new artifact ID widens authority. Current
evidence/source display/transport authority also applies; a record audience
never grants disclosure of restricted inline evidence or source metadata. Mixed
artifacts are omitted whole or the required operation stops. Legacy unknown
copies cannot become unrestricted by replay. Exact reviewed replacements use
new permitted inputs/lineage; manual originals remain independent authorities.

Admission binds destination, purpose, exact versions, actor/root ceiling,
policy/grant revisions and purge/retirement controls. The native ADR-202 gate
covers final qualified adapter entry, not queue insertion. Retries, child routes,
fallbacks, summaries, interviews and resumed jobs require fresh admission. No
network await is held inside the gate. Model-generated web/MCP/skill/shell/file
arguments and independent publication conservatively inherit governed input
restrictions. Ordinary tool permission is no disclosure grant. External or
unqualified tool/publication paths remain disabled in governed runs until a
separate reviewed purpose/custody and owner contract qualifies; only qualified
native local tools preserving restrictions may run. Current authority changes invalidate
cached/prepared payloads and their Next Send projection. Optional profile input
may be omitted; profile-dependent required input fails generically. Already
entered sends have uncertain/begun disclosure, not promised recall.

Remove the global `unsupported_records_present` flag from model and ordinary
Next Send outputs. Absent/hidden/wrong-scope/unsupported/audience-denied records
share generic tool results and cannot affect permitted ranking/omission counts.
Owner-only maintenance/consent views require their own current permissions and
never become automatic logs, capture metadata or provider content.

Migration defaults all legacy/unreviewed records to deny model disclosure;
existing credentials and visibility are not grandfathered grants. Use profile-wide
version/capability gates under ADR-201; old active consumers block cutover or are
explicitly retired. Restrictive/unknown policy conflicts deny pending review.
Recovery never imports local grant authority. Preserve `server_trusted_v1` Sync:
storage receipt or canonical policy is no grant for server-side model jobs.

The first future release is native deny enforcement plus explicit qualified
on-device enrollment for new registered-lineage V2 input in new conversations,
including context, profile tools, root/child rounds and governed history. All
included seams must qualify; unsupported interviews, summaries, embeddings,
background jobs, replay and remote grants stay disabled for governed data.
A later remote release qualifies exact direct destinations independently.
Explicit owner exports remain separate; upload still requires model consent.
ADR-119 prompt-cache binaries/live slots are governed input state: new
conversation identity is not clean-state proof. Until native cache/launch
lineage, inherited policy and final-use controls qualify, Save/Restore/reuse of
governed state stays disabled. Initial local execution needs explicitly reviewed
clean owned process/slot state or refuses, without implicit cache reset/deletion.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Provider-name/URL-label allowlist | Endpoint, account, route and fallback custody can change without a label change. |
| Agent visibility or Syncability implies model consent | Conflates local read/storage with model egress and violates privacy intent. |
| Grandfather all previous sends on migration | Historical disclosure does not establish current consent. |
| Union permissions or redact mixed artifacts with a model | Can widen disclosure and creates another egress operation during enforcement. |
| Portable native grants in canonical profiles/exports | Imports could grant another runtime/provider permission without local consent. |
| Rewrite V1 controls or claim global legacy coverage | Breaks canonical compatibility and invents lineage for unknown copies. |

## Consequences and approval boundary

Implementation needs versioned canonical policy/fixtures, native encrypted grant
storage and revisions, qualified destination adapters, governed dependency/policy
propagation, final-entry admission and compatibility/deny migration. It changes
future behavior visibly: unreviewed profile material is omitted until enrolled.
This task is documents/tracking only; no permission or provider behavior changed.
The user explicitly approved the reviewed written contract. ADR-203 accepts
design direction only and remains provisionally numbered against concurrent
branches until integration-time allocation checks. Implementation and current
grants remain separately scoped.
