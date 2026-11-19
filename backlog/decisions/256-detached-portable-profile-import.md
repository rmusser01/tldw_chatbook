# ADR-256: Detached portable-profile import

Status: Accepted (user approval, 2026-10-28)
Date: 2026-10-28
Related task: TASK-37205
Extends: [ADR-283](283-windows-portable-wezterm-spike.md)
Preserves: ADR-029, ADR-125, ADR-126

## Decision

Support moving first with the original unavailable through an explicit reviewed
detached import into a fresh isolated generation. Retain copied Data and recovery
records unchanged and inert. Never rewrite ancestry/native identity receipts,
manufacture live capture authority, or import execution approvals as active.

Use a separately confined installed source-layout resolver, private stable
acquisition, pathless pending/journal inspection, installed owner/schema and
dependency validators, and a strictly partial archive. Raw copied files cannot
prove historical coherence. Reuse the existing isolated restore, journaled
publication, fresh installation identity, activation review and fresh-process
launch. A relative active-profile locator is not launch authority.

No source-side preparation or automatic checkpoint is required. Block unsupported,
ambiguous, pending, changed or unsafe sources without mutating their bytes.
The initial policy excludes managed credentials through the existing staged
credential policy; the destination requires credential setup again.

The full approved contract and its alternatives, concrete boundaries, crash
behavior, validation and limitations are in
[the design](../../Docs/superpowers/specs/2026-10-28-portable-profile-relocation-design.md).
The user approved this design, including managed credential exclusion, after
requiring destination-only operation with the original unavailable.

Implementation clarification: ordinary startup seeds references to bundled
built-in persona artwork. The installed immutable asset owner may supply those
exact bytes as explicit support, only after its ID/hash/reference validation.
This never permits opening an old user path or substituting custom content.

Owner validation runs in a one-shot child which exits before the coordinator
obtains native publication gates. This retires any legitimate startup lease
introduced by an owner dependency without bypassing or manually clearing it.
Pure agent-model imports must not initialize ordinary config: Chat's existing
service exports are lazy so SQLite goal validation can run inside recovery.
An interrupted current local candidate before pending-pair registration resumes
through its receipt-bound plan and the same existing isolated executor; copied
or escaped paths are refused before source-target IO.

Detached partial input may represent a default SQLite store as unused only when
it is provably absent, has no explicit custom selector and leaves no main,
WAL/SHM/journal or lock residue. This supports normal lazy base-DB creation (the
portable note probe creates Notes alone). Report absent default owners during
review; never manufacture an empty replacement payload. Included owners still
run installed reference/dependency checks and refuse required missing stores.
Ordinary live capture's required-core policy remains unchanged. Historical
deletion/coherence cannot be proved from raw copies and is not claimed.
