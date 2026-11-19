# Controller scope clarification

The architectural report is the requirements for this necessary identity repair,
with this clarification: keep ordinary text/fenced sends and explicitly ephemeral
session-only endpoint work supported. Absence of durable continuation authority
must refuse a persistent native continuation before any tool executes; it must not
create a blanket ban on ordinary session-only text sends. Endpoint-origin privacy
and the existing 2,048-byte destination cap remain. If an existing ready resolver
omits a known protocol, fix the host resolver's captured fact at its existing owner;
never derive accepted authority later from current configuration or a candidate.

This extends existing ADR-063/147/179 ownership, with nested destination JSON V2 and
continuation JSON V3. SQL columns/relational schema stay unchanged; no SQL version
bump or backfill. Legacy JSON readers and strict rejection are tested. Any discovered
need for actual DDL/migration, new owner or broader permission requires controller
assessment before code. No paid run/retry or new allowance is authorized.


Source-proven fifteenth owner: Chat/Chat_Functions.py must forward the existing
host-owned provider_continuations parameter to custom-hosted. The actual second
request currently drops it at the generic dispatcher despite prepared tool history;
then the engine has no current checkpoint and produces no completion candidate.
CUSTOM_PROVIDER_PARAM_MAP is used only for custom-hosted; add the one entry there
if confirmed, rather than a redundant wrapper. Keep both legacy custom handler
maps unchanged and document/test this sole intentional parity difference. No other
parameter/permission/provider behavior changes. This refinement is recorded before
editing the fifteenth owner.


Actual reopened pending Resume revealed that production gateway lacks the existing
expand_provider_continuation callback; fake gateways hid this prerequisite. Add the
small canonical projection in the existing gateway owner: assistant text and native
call rows with exact IDs/arguments, followed only by recorded completed/failed result
rows. No synthesized pending results or private reasoning in ordinary text. Existing
controller/runtime target/permission/executing-state guards and provider-specific
private replay stay authoritative. Prove actual explicit Resume, exact-once result
pairing and no automatic execution. This is within the existing gateway owner and
ADR-063 promised recovery contract, with no added framework or allowance.


Actual pending Resume then reached the real recovery stream and exposed restored
assistants' terminal live status. Extend existing begin_generation_attempt with
explicit expected active checkpoint plus expected persisted continuation/message
version, passed only by the host explicit restore path. Validate exact current
record/owner/state under its generation scope; reject stale, foreign or executing
state. Reset only live status and previous terminal-attempt bookkeeping. Preserve
content/checkpoint/durable versions/CAS; do not call generic retry clearing. Normal
callers stay unchanged. Prove closed/reopened explicit completion, retained partial
content and negative activation/no-automatic-execution controls.
