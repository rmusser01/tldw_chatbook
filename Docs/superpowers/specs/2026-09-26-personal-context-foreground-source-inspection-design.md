# Foreground Personal Context source inspection

Status: Accepted design direction after user endorsement and technical review, 2026-09-26; no source access or runtime rollout.
Date: 2026-09-26
Task: TASK-25907.15

ADR required: yes.
ADR path: [Accepted design ADR-191](../../../backlog/decisions/191-foreground-personal-context-source-inspection-authority.md).
Reason: a new interface joins profile inspection, conversation ownership and UI publication.
Existing evidence direction: [ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md).

## Outcome

Let a user inspect one exact bound span from an already open, authorized local
conversation. Show a quote only when the live message owner still provides the
bound version, exact representation and matching whole-text/span digests.
Report “Text matched when checked” with the host observation time. This proves
an exact match at that observation, not authorship, semantic support, approval
or continuing truth. The quoted document remains data and cannot issue instructions.

This task specifies the interface and qualification gates. It does not add a
resolver, UI action, source capture, V2 record storage or a new evidence store.
The standalone binding component exists, but current V1 profile records do not
contain admissible complete bindings. No existing V1 provenance UI acquires
quotation by assembling its separate message IDs and hashes.

## Approaches

1. **Foreground selected-conversation inspection (recommended).** A host-owned
   user action joins an independently inspectable profile selection with the
   currently open persisted conversation. This bounds access and gives a real
   UI caller for qualification. A different conversation must first be opened
   through its normal authorized owner; the inspector does not navigate for it.
2. **Generic resolver accepting IDs and capability flags.** Convenient for tools
   and imports, but a DTO or callback supplied by the caller does not establish
   source ownership. Reject this as the first release; it also enlarges egress.
3. **Ship all Profile V2 together.** Allows portable source-bearing records but
   entails compatibility, migrations, retirement, forgetting and disclosure.
   Keep those separately qualified; this contract does not authorize that rollout.

## Verified native boundaries

- `Personal_Context/service.py:settings_provenance` inspects a Settings-owned
  current record/proposal under profile identity/version checks. It resolves
  metadata only. READY and DISABLED allow user inspection; agent context grants
  from `authorized_context_view` are a different authority.
- `Chat/console_chat_controller.py:_compose_profile_tool_provider` captures a
  stable profile grant and a Console user-role message. Neither its role nor
  persisted/local ID fallback attests source origin or grants arbitrary reads.
- `Chat/citation_source_locators.py` separates source authority namespace,
  capabilities and inert imports. Its authorization DTO is a host assertion,
  not a credential that imported data can mint. Citation identity is independent
  of the Personal Context profile; do not equate their IDs.
- `DB/ChaChaNotes_DB.py` current message getters do not authorize profile or
  workspace access. Revision lineage identifies versions; it is not an archive
  of guaranteed original bodies. See the [native readiness audit](../../../backlog/docs/personal-context-source-readiness-audit.md).
- `Chat/console_chat_controller.py` has session lifecycle revisions. Existing
  counters and locks do not yet establish the new complete source-read fence.

These observations constrain the design. No current method is relabelled as
an existing production source-inspection grant.

## Trusted foreground action and source owner

The entry is an explicit Inspect source action inside the foreground My Profile
provenance view or its foreground proposal-review modal. That Settings-owned
inspector owns the gesture, request and disposable result. An app-owned
coordinator joins its profile subject/binding selection to the currently loaded
Console conversation owner. Settings can cover/suspend ChatScreen while its
controller/runtime survives; do not require both screens to be foreground or
resume Console just to inspect. The originating inspector must remain foreground.

The event carries selection IDs, not permission booleans, raw source text,
filesystem paths or resolver URLs. The coordinator obtains current owners from
the application, never from event-supplied service/database/namespace values.
Use already present owner services and source identity only. Do not call
`get_personal_context_service(retry_locked=True)` or another bootstrapping getter,
create/load a missing Console by navigation, provision a key, or repair a missing
semantic revision. Missing or locked owners return unavailable. The service
that issued the Settings selection and the current app-owned service must still
be the same instance.

Profile inspection must first re-read the selected record/proposal from its
current repository and compare the exact subject, profile identity, purge
generation, record/proposal revision and complete binding digest. The binding
must belong to that current object. Disabled profile runtime does not revoke
local Settings inspection; locked, purged, unavailable or stale selections do.
Profile READ_ONLY, PROPOSE or DIRECT_WRITE agent grants never substitute for this.

Separately, the source owner must establish all of the following before fetching
or decrypting message content:

- The action is a live foreground gesture in the originating mounted Settings
  inspector, with no pending old worker occupying admission capacity. The
  surviving Console owner has that exact persisted session selected. A suspended
  ChatScreen is acceptable; a detached or replaced runtime/controller is not.
- Its current conversation belongs to the actual host database/source namespace
  and remains open under the current local workspace/conversation read policy.
  Active selection alone is insufficient; detached, ephemeral, missing,
  unauthorized, deleted or imported owner contexts fail closed.
- The selected binding names that exact namespace, governance scope, persisted
  conversation and persisted message. No global/workspace fallthrough, local
  message-ID fallback or search across other containers is permitted.
- Current native source authority is supplied by its existing host owner. For
  the first local adapter, use the already provisioned current local citation
  identity (`load_local_citation_identity_context` reads it without key material).
  Missing identity cannot trigger initialization; compare all namespace fields, and keep the separate
  Personal Context profile identity distinct. Missing owner identity returns
  unavailable. Never construct a new citation identity from profile metadata.
- Source read capability and revocation state permit this specific foreground
  operation now. A parsed binding or `CitationReadAuthorization` reconstructed
  from it has no effect on that decision. Tenant/server/imported bindings are
  unsupported in this local release even if structurally valid.

The source owner issues a short-lived, nonserializable inspection lease scoped
to this handler, request, exact conversation/message and current policy epoch.
The lease is an internal host artifact, never returned to the agent, stored on
a profile or accepted from a plugin. It is not a caller-provided `authorized=True`
flag. All consumers still validate the current owner when using it.

A production host factory/guard implementing these checks does not exist yet
for Personal Context. Its real admission route and revocation/write coverage
must be implemented and tested together with the UI caller; a test-only DTO
factory does not satisfy this contract.

## Coherent source observation and exact matching

The native source owner reads conversation membership, nondeleted message,
existing current immutable semantic revision metadata and exact decoded `content`
in one coherent SQLite snapshot under the source lease. It verifies the live
locator belongs to that message and conversation. Missing revision metadata is
unavailable; do not call `ensure_current_revision`, create/bootstrap a ledger,
or use a provider projection to acquire a version. A snapshot is consistency,
not permission. Authorization must precede body access/decryption.

This requires a narrow source-owner read. Generic `get_message_by_id` includes
image BLOBs; its no-blob sibling still hydrates unrelated JSON and both can log
message IDs on failure. Neither generic getter nor the semantic message-envelope
helper qualifies. The owner-controlled parameterized query loads only necessary
membership/version metadata, bounded exact content and size information. It
must not hydrate images, attachments, sidecars or provider metadata, and its
failure path must not log IDs, SQL parameters or SQLite exception text.

Only `message_content_text_v1` is admitted: unmodified decoded message content,
excluding title, role, attachments, rendering and provider projections. Require
`version_kind=owner_immutable`, the owner's actual current version ID and exact
codepoint bounds. Then call the existing exact-text helper for the entire
representation and `[start,end)` span; both SHA-256 values must match. The
complete binding digest must also still match the current profile selection.

No Unicode normalization, newline conversion, joined message text, substring
search, approximate matching or automatic rebinding is allowed. Old-version
mismatch is unavailable even when the current body happens to be identical.
The sync log, trace snapshots, Notes and historical message projections provide
no fallback. An empty span can be valid metadata but yields no inspection quote.

Use a worker for database/decryption/hash work. Reject whole representations
over 262,144 codepoints or 1,048,576 strict UTF-8 bytes and selected spans over
4,096 strict UTF-8 bytes; do not truncate a span and call it verified. The owner
must enforce a bounded stored-payload read before loading/decrypting an oversized
body, with a bounded decoded-size check afterward. Unpaired surrogates fail.
The current message content is SQLite TEXT. Its pre-hydration byte check must
measure complete stored UTF-8, including text after an embedded NUL: character
`length(content)` alone is not a byte guard. The narrow owner query must gate
content projection on its full byte bound before Python hydration, then enforce
codepoint/strict-UTF-8 limits after decoding. Any future encrypted encoding also
needs a separately qualified bounded stored-payload/decryption path.

One app-owned inspection capacity slot exists, with no waiting queue. It belongs
to the actual worker until its final cleanup completes, even if its UI waiter
is cancelled, replaced, suspended or timed out. A new gesture during that drain
returns unavailable; it cannot launch another lingering worker. At the 5-second
monotonic deadline, clear the pending/result UI and signal worker cancellation.
Discard late results before scheduling publication; retain the worker only to
close its owner-managed resources and release capacity in its own finally path.

The read needs inspection-owned deadline/cancellation controls: bound lock/busy
waits by the remaining budget, use SQLite progress/interrupt where appropriate,
and check cancellation between bounded decode/hash stages. Do not inherit the
normal 15-second connection timeout, change a shared connection's busy/progress
handler, or interrupt unrelated queries. The inspected connection must join the
owner's normal maintenance/quiescence lifecycle. A deadline guarantees stale
publication rejection, not forcible Python thread termination; residual work
cannot free admission capacity early. No provider or model is involved.

The captured source-role annotation and `captured_at` remain unverified unless
an independent native origin/capture receipt establishes them. A stored USER
role can contain quotations or imports. Matching text can be displayed with
that limitation; it cannot silently upgrade the annotation to direct user
assertion. Never create support assessments or approval receipts in this path.

## Publication, invalidation and freshness

Capture host view/request identity and revisions for profile lifecycle, selected
object/binding, source namespace/policy, selected Console session/conversation
and current source version. Revalidate after worker completion through the
current application owners; a replaced service/controller is not the old owner.

Publication has two phases: observe privately, then validate and commit to the
owning UI. The second phase uses owner-held read leases and a fixed acquisition
order (profile lifecycle, source policy, profile-head snapshot, source snapshot);
release in reverse.
The final source and profile-head probes must use fresh committed owner-managed
snapshots opened after worker observation. A borrowed/pre-existing transaction
or a cached epoch is not a final probe: it can keep seeing an old WAL snapshot.
Reject such a transaction or obtain a fresh owner-managed read connection through
its normal custody/quiescence API, without committing or rolling back another
caller's transaction. A nominal `transaction()` wrapper is not sufficient.

Never retain locks from the worker across an await or a queued UI callback.
The final UI phase acquires its own nonblocking leases and checks only bounded
owner metadata/epochs; it performs no decryption or full-text hashing. It may
commit synchronously to the owned result widget without an await or external
callback, then releases all leases. If that phase cannot acquire authority or
remain below the UI 100 ms work budget, return unavailable. Check request/view
identity immediately before rendering. All relevant canonical profile/proposal
writes, binding replacement/removal, accept/reject/expiry transitions, incoming
Sync admissions, lock/purge/policy changes and source edit/delete/revocation
writers must join this gate or invalidate it before their transition becomes
visible. Verify consistent reader/writer lock ordering rather than assuming
existing writers follow this new order. A double check alone is insufficient.
No new shared cross-database transaction is implied.

If the implementation cannot establish this publication gate across actual
writers, it cannot ship quotations. In particular, existing lifecycle counters
without source-write/revocation coverage do not satisfy it. Avoid adding a
lock that only the new reader observes.

Clear a displayed quote and cancel its request when selection, service owner,
profile lock/purge, source policy/namespace, session/conversation, message
revision/deletion, originating inspector collapse/screen suspension or unmount
changes. ChatScreen suspension caused by entering Settings alone is not result
revocation; a Console source-owner/session/policy transition still is.
Pending proposal expiry and any timed source-authority/lease deadline must be
checked at final publication and every later result use. Schedule a one-shot
clear at the earliest applicable expiry, rechecking the trusted owner clock
when it fires. This timer disposes the result; it does not fetch source bodies.
A resumed inspector never restores the quote or automatically resolves a source.
The existing one-second metadata refresh does not trigger source inspection.
Repeated selection cannot reuse a prior successful result. A quote is excluded
from Settings snapshots, trace capture, debug diagnostics and context previews.

A successful observation is truthful for its coherent snapshot and final gate;
it does not promise uninterrupted currentness afterward. Source changes known
to the host invalidate immediately. External-process writes cannot be assumed
to emit native events: the source owner must detect them through its authoritative
change/revision mechanism before any subsequent inspection or publication.
Until cross-process invalidation is qualified, label the result with observation
time and prohibit a “currently verified” badge or automatic reuse. No background
poller is introduced by this contract. Privacy revocations owned by another
process need a shared revocation gate or the release must stay disabled for
that deployment; the time label does not excuse an authorization gap.

## Result and privacy contract

The UI receives either a disposable matched span plus host observation time,
`changed` for its own stale selection, or content-free `unavailable`. Denial,
absence, policy restriction, version/hash mismatch, unsupported binding,
invalid Unicode, oversized content and deadline use the same unavailable shape;
no hidden existence, identity, count, path, snippet or database exception leaks.
Internal reason codes are content-free and never include binding/message IDs.

Hash only the untouched source/span; build a separate display-only escaped
projection. In that projection preserve LF as a line break, double literal
backslashes, and render every other Unicode Cc/Cf/Zl/Zp codepoint as a visible
`\uXXXX` or `\UXXXXXXXX` escape. Thus CR, tab, ESC/OSC and bidi controls cannot
execute or silently rearrange the view, and literal escape spelling remains
distinguishable from an escaped control. Label changed display “Control characters
shown as escapes”; never claim those rendered bytes were the hashed source.
Do not normalize or mutate the original text/digests. The escaped display is
bounded by 24,576 UTF-8 bytes for the 4,096-byte original-span cap.

Use Rich Text and markup-disabled widgets only after this escaping, with no
automatic link/action activation. `Text(raw_span)` alone retains ESC and bidi
controls in the installed Rich implementation; `markup=False` is not sufficient.
Source text cannot issue commands or become a prompt instruction.
Quotes and lease/observation objects have no automatic repr/log serialization.
Known local view changes may explain “Selection changed; inspect again” without
revealing whether an unknown source exists. Clear the prior span before showing
any failure. Memory cannot be guaranteed zeroized in Python; promise no deliberate
retention or reuse, not physical erasure of all allocations.

No quote is saved, synced, exported, sent to tools/providers, copied automatically,
indexed or retained as an excerpt. Future explicit copy/export or retained
capture needs its own disclosure/derivative policy and lifecycle qualification.

## Concrete runtime qualification matrix

Tests use synthetic sources and real temporary SQLite only; no real profile,
keyring, provider, companion server or network. Every denial has an authorized
successful control, and spies prove no body read/decryption before admission.

| Group | Required evidence before shipping |
| --- | --- |
| Positive caller | Real Settings-to-coordinator action with ChatScreen suspended but its current source owner surviving; current profile binding, independently authorized persisted conversation, matching owner version and both digests yield the span plus host observation time. Hidden/suspended Settings cannot read or publish. |
| Namespace/owner denial | Other profile lifecycle, source authority, scope, database, workspace, conversation, message, imported or tenant binding; forged flags/DTO/lease and stale host instances never read bodies or publish. |
| Binding admission | Current object actually contains the binding; legacy IDs, changed/removed binding, expired/nonpending proposal and invalid V2 admission do not resolve. Missing owner/revision cannot cause bootstrap, key provisioning, navigation or ledger writes. |
| Exact matching | Astral Unicode, combining forms, CRLF, bounds, empty span, malformed text, one changed whole-text digest, one changed span digest, identical text under a different owner version. No approximate or historical fallback. |
| Budget/rendering | Full UTF-8/NUL/multibyte size guards before hydration; tiny content with multi-megabyte image/attachment/JSON payloads proves they stay unread. Raw ESC/OSC, CR/tab, bidi, literal backslash escape spelling and instructions are safe in actual compositor output; limits apply before/after display escaping. Failures clear old spans and reveal no sensitive diagnostic. |
| Worker custody | A held read remains the sole admitted worker after waiter cancellation/deadline/unmount; repeated gestures cannot create more jobs. Real busy/progress/deadline controls cancel only the inspection connection, cleanup releases capacity once, and no late body/result publishes. |
| Concurrent publication | Barriers around observation/final probe/render: profile binding edits/removal, proposal resolution/expiry, Sync admission, lock/purge, source revoke/edit/delete and owner/request changes. Real owner writers join the gate. Two-connection late source/profile edits and deletes invalidate publication; a pre-existing read transaction cannot pass as a fresh probe. |
| Retained visible result | Source edit/revoke, selection change, originating inspector collapse/suspend/unmount and owner replacement clear the quote. Advance proposal/source-lease clock past expiry without a DB write: the quote clears, and resume or metadata refresh cannot restore it. External DB changes are detected before the next read or publication without cached authority/body reuse. |
| Deployment/privacy | No quote in provider/tool/trace/Settings/export/sync artifacts; shared-process revocation deployment either qualified or unavailable. Existing V1 and metadata-only inspector tests still pass. |

## Implementation entry gates and sequence

Acceptance of this document approves the contract, not all V2 rollout. Before
an implementation plan for quotations, require separately scoped, user-reviewed
V2 local admission/record ownership with a real containing binding and the
retirement/disclosure controls required by ADR-185. Binding IDs, roles, spans and
digests themselves are governed metadata even without a retained excerpt.
Qualify canonical/outbox/Undo/cache retirement and source-policy revocation for
that metadata. The current `device_only` model/tool disclosure gap is unresolved;
a device-only flag alone cannot qualify V2 admission. Keep source-bearing V2
metadata out of every existing agent serialization, provider/tool request,
export/Sync and recovery path until the applicable controls pass their own tests.
No dormant production resolver or fake authoritative wrapper is filed just to consume the component.

The source inspection implementation is one vertical slice: real host admission,
bounded coherent read, exact verification, publication/invalidation and mounted
UI caller with the matrix above. Its plan must name all actual mutation/revocation
owners and deployment assumptions and link the accepted scoped ADR. If that
slice cannot fit an independently testable PR, reduce supported deployments or
sources before splitting away its authority from its caller.

Server/tenant sources, historical exact body retention, automatic source capture,
V2 sync negotiation, support assessments, correction history, forgetting and
provider-disclosure rollout remain separately qualified work. The independent
generated-answer effectiveness task remains untouched.
