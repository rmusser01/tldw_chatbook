# First sequential workflow: execution design checkpoint

Status: Approved by the user, including the final written design. The five-step
example and option A, session-bound execution, are approved. Runtime implementation
has not begun. TASK-32690 records the design; TASK-32691 tracks implementation.

ADR required: yes; the approved lifetime choice is recorded in existing ADR-138.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: session-bound delivery defers ADR-138's durable run/wait/recovery
guarantees. It is not an assertion that those guarantees already work. ADR-125
private SQLite behavior and ADR-036 service composition remain unchanged.

## Approved user outcome

Run an editable saved workflow from the existing Workflows screen:

1. `media_ingest`: select and read one local UTF-8 `.txt` file.
2. `prompt`: render an editable instruction, initially a three-bullet summary,
   with the file text using the existing bounded reference resolver.
3. `llm`: call the selected llama.cpp model at `localhost:9099`.
4. `wait_for_human`: display the generated text, allow edits, accept or reject.
5. `notes`: create a Local Note containing the accepted text; offer Open Note.

Rejection/cancellation stops this linear run; it is not alternate routing.
The existing `Tests/fixtures/workflows/file_to_note.json` supplies the canonical
step IDs and references. It remains a normal workflow definition, not a separate
hardcoded wizard. Unsupported operations/configuration block execution without
rewriting the definition. Branching remains v2 and parallelism v3.

## Evidence and source boundaries

Inspected merged Chatbook dev `657f70ffe7fbb92b637a89cc2961e8bf95ec6005`, which
contains PR2690, and preserved source `8aa1987af9357655af9354610b878f247fd1e929`.
The server dev API returned `59049e094e0845a4611ea725ae19b7c1754ea709`.
This is source inspection, not live-model or server-conformance evidence.
No local app, model, user profile, or server write was exercised.

| Area | Observed reusable boundary | Constraint |
| --- | --- | --- |
| Authoring | `Workflows/authoring.py`, document/draft services, `expressions.py` | Keep immutable saved revisions and recoverable authoring drafts unchanged. |
| SQLite | `DB/Workflows_DB.py`, existing v1-v4 migrations | Historical run tables exist, but the merged class exposes authoring transactions only. Their presence is not a recovery implementation. |
| Parked run state | `RunService._writing()` calls `db.execution_transaction()`; `close()` releases execution authority | Do not copy it while replacing those guards with ordinary transactions. Recovery assumes OS ownership. |
| Parked ownership | `runtime_lock.py` checks database/lock identities and acquires a workflow lock | Excluded. Its raw live-database opening is incompatible with the adopted SQLite boundary. No helper-owned replacement is proposed here. |
| Current instance warning | `Utils/instance_lock.py` explicitly says detection-only; failures can return `acquired=True` without a handle | Not execution authority, not a basis for stealing or resuming another process's work. |
| Model | `LLM_API_Calls_Local.chat_with_llama()` accepts `api_base_url` and `api_key_resolved` | Blocking Requests transport; cancelling its awaiting coroutine does not end the worker. Pin endpoint/model/credential intent explicitly. |
| Notes | `NotesScopeService.save_note(scope=LOCAL_NOTE, create_note_id=...)` | Retains Notes policy; local work blocks despite the async signature. Same-attempt ID/readback is not general cross-process recovery. |
| Permission | `BuiltinToolGate.check_detailed()` | Preserve effect-time Off/kill-switch/Ask decisions. No `strict=True` API exists; do not copy that parked assumption. |
| Navigation | `NavigateToScreen(TAB_LIBRARY, {LIBRARY_NAV_CONTEXT_NOTE_ID: note_id})` | Open the confirmed Note using the existing Library route. |

The parked local adapter's literal-numeric-loopback check rejects `localhost`;
run setup must explain this instead of relabeling the endpoint as Ollama.
Configured custom endpoint identity and the `llama_cpp` transport family are
different values. Keep the selected local target visible and bound to the run.
For the user's `localhost:9099` selection, resolve and display its concrete
loopback address before approval, and pin that numeric address for dispatch.
Do not allow a non-loopback resolution, silently switch addresses after failure,
follow redirects, use environment proxies, or inherit another provider's key.
This first slice admits keyless loopback requests only. This preserves
numeric-loopback dispatch; it does not grant general DNS egress.

## Selected lifetime: A — session-bound first delivery

The user selected A. Restart-resumable execution is not part of this delivery.

One application-owned sequential run at a time in each app instance. Its
immutable revision, progress, outputs, edited review and launch identity live
with the application, not the mounted screen. Leaving Workflows does not cancel
or lose that run. Saving/editing the definition does not change the running copy.

The Run action captures the selected file, actual model endpoint/model, local
Notes destination and user identity. Imported metadata grants no authority.
Every effect still passes the existing permissions/policy checks. Accepting the
summary cannot override a denied Note write or the kill switch.

During an active run, ordinary Quit offers Stay or Cancel run and quit. A review
wait may be cancelled immediately; physical file/model/Note work must settle
before its resources close. Cancelling means stop advancing to later steps and
retain the outstanding operation until it returns. The UI must not say Stopped
while its local worker is still active, or promise that a disconnected llama.cpp
request stopped server-side generation. No forced process termination is added.

After normal quit or a crash, there is no run resume, automatic replay, review
restoration, or persistent run-history promise in this delivery. Unsaved review
edits and intermediate outputs are lost. The saved workflow and its authoring
drafts remain durable exactly as today; a committed Note remains in Library.
The UI must disclose the session-only limitation before Run and on quitting a
review wait. No background scheduler or multi-instance ownership is claimed.
Different app instances create independent runs; no cross-instance deduplication.

The Note step creates, never updates, using one generated Note ID for that run's
attempt. Duplicate UI delivery cannot dispatch a second write. If the commit wins
a cancellation race, report the saved Note; do not label it rolled back. If a
write response is uncertain, retain its ID and use the existing local read path
to reconcile within the session. Do not generate a new ID or retry blindly. A
missing/unverifiable receipt produces an explicit uncertain outcome. A crash may
leave a committed Note with no workflow receipt; inspect Library before rerunning.
Readback must match the captured destination, attempted ID, title and accepted
content; finding an ID alone is not sufficient to claim this run's save succeeded.

This delivery adds no run schema, database owner, lock file, PID record, helper
protocol, recovery scanner or persistent execution service. Existing SQLite
utilities still operate normally for authoring and Notes; this is not a claim
that the application uses no existing helper processes.

### Deferred alternative: restart-resumable runs

Persist review edits and run identity, recover a waiting review after restart,
and reconcile uncertain effects durably. This retains the wider historical
milestone, but requires a separately explicit ownership/claim/recovery design
against current dev, plus multiprocess and abrupt-exit evidence. Existing tables,
UUIDs, or an in-memory flag do not establish exclusive physical execution.

Do not silently reintroduce the withdrawn implementation to obtain this behavior.
A later delivery must resolve its ownership design before implementation. Choosing
A defers these guarantees; it does not demonstrate that restart recovery is
impossible without new infrastructure.

Restoring the complete parked runtime is not recommended: it couples this one
example to obsolete service signatures and the excluded ownership machinery.

## Step contracts

| Step | Inputs and result | Authority and stop behavior |
| --- | --- | --- |
| `media_ingest` | One selected local UTF-8 `.txt`; exposes extracted `text`. No URL fetch, Media record, file modification or ingestion pipeline. | Approve the captured file read. Reject unsafe paths and oversized/invalid text before model dispatch. Cancellation discards a late result. |
| `prompt` | Saved editable template plus earlier typed references; exposes rendered `text`. | Pure bounded evaluation, no code execution or effect approval. Unresolved references fail, never become guessed inputs. |
| `llm` | Rendered prompt, captured `llama_cpp` endpoint, actual model and explicit output budget; exposes generated `text`. | Authorize sending this input to this endpoint. One request, no tools, fallback or automatic retry. Cancel/timeout blocks later steps and drains the owned request. |
| `wait_for_human` | Generated text shown in an editable review; Accept exposes the exact edited `text`. | Only the captured local reviewer may respond. Reject/cancel/expiry ends the run without a Note. Accept is single-use and does not itself authorize a denied Note write. |
| `notes` | Accepted text and nonempty title; exposes the confirmed created Note identity for Open Note. | Recheck local Note-create authority immediately before writing. One create ID, no update or remote destination. Reconcile uncertain completion; a confirmed commit remains saved even after cancellation. |

These are declared local subsets, not claims that all server configurations of
these step types work. Execution admission checks the entire definition before
reading the file: only supported sequential operations and backward references,
zero retries, no completion callbacks or undeclared effects. Unknown executable
fields/configurations block Run with a reason; they remain losslessly editable
and exportable. Display metadata is not an execution grant. Execution must never
coerce an unsupported definition into the example.

## Integration boundary

The intended shape is one lazy app-owned session coordinator, with a small
closed dispatcher for the five validated operation subsets and the existing
editor/navigation surfaces. Reuse the bounded expression code and current
domain services. Reuse parked tests for relevant behavior, not their ownership
assumptions or broad source files. Do not build a generic provider/plugin/runtime
framework, replace the SQLite factory, or copy the old run service wholesale.

The boundaries below are requirements for implementation qualification, not
assertions that current service signatures already satisfy them:

- **Model:** keep request construction and transport in the existing LLM domain,
  with one opt-in bounded local request path; no workflow-owned HTTP framework
  or Console-run/trace dependency. The unmodified `chat_with_llama()` path is not
  sufficient: it reads timeout/retry settings at dispatch, and Requests socket
  inactivity timeouts are not absolute deadlines. Capture the effective settings,
  enforce an absolute connect-through-body deadline, cap response bytes before
  unbounded decoding, refuse redirects/proxies and retries, and redact private
  payloads. Qualify this narrow seam before enabling Run; do not copy the parked
  transport/socket ownership implementation wholesale.
- **Notes:** use the app's existing `NotesScopeService.save_note()` with
  `scope=LOCAL_NOTE`, the captured user, one `create_note_id`, and no Sync v2
  profile or organization changes. Its local branch performs synchronous work:
  invoke it through a retained local-only worker bridge, not on Textual's event
  loop. Preserve its policy check and transaction; do not call the DB directly
  to bypass them. Capture/recheck the actual destination through the existing
  `NotesInteropService.notes_db(user_id)` owner, not just its mutable template.
  A changed/unavailable destination stops the write/readback rather than routing
  to the current selection. No new DB owner or raw database identity probe.
- **Permissions:** use the existing permission resolver and structured gate
  verdicts for file/model/Note effects; refresh authority at each effect, including
  after a human wait. Missing or unreadable permission authority blocks execution.
  Current `check_detailed()` alone does not establish that strict-read guarantee;
  add only the necessary opt-in validation through the permission owner, without
  changing ordinary callers' defaults. An Ask approval is bound to the run, step,
  resolved effect and destination, not a portable flag or blanket workflow grant.
  Off/kill-switch revocation wins over earlier approval. Keep review/UI callbacks
  on the app loop and DB/file blocking work off it.
- **Files:** reuse the existing path/private-file validation, reject visible
  live-database/sidecar aliases before raw I/O, and require a stable selected file
  and containing path during the read. Do not claim protection from concurrent
  inode substitution or introduce another SQLite proof/helper protocol. File
  contents go only to the approved model and Note, not persistent run logs.
- **Shutdown:** first refuse new Run/Accept dispatches; settle authoring drafts
  through their existing owner; cancel the session and retain/drain outstanding
  file/model/Note work; then close dependent services through normal app teardown.
  Navigation only detaches the view. Timeout/cancellation cannot free the run slot
  while work remains live. A drain failure keeps an explicit stopping/error state;
  it does not force-close a database still in use or announce successful shutdown.

### Bounds and timing

Retain the previously approved local admission defaults: at most 100 steps and
2 MiB canonical definition, 10 MiB serialized run inputs, 1 MiB serialized output
per step, 100 MiB aggregate outputs, 60 minutes cumulative active execution and
100,000 input-plus-maximum-output token admission units. These counters live only
in this session. The authoring store can preserve larger definitions without
promising they are executable. No artifact spill, silent truncation or configurable
limit-increase UI is included. Stricter domain/model limits win.

Bound the source read and HTTP response body to 1 MiB each before decoding;
rendered prompts and edited review results must also fit the per-step output cap.
Use strict UTF-8 for the file. Reserve a conservative input estimate plus explicit
`max_tokens` before a model call; the example uses 512 output tokens. A model
context overflow fails visibly; do not shorten the prompt or switch models.

`step.timeout_seconds` is the active-attempt deadline (the example uses 300 s).
Require an explicit finite positive whole-second value for active attempts;
missing/invalid values block admission rather than becoming unlimited.
The model's captured finite positive request timeout cannot exceed that deadline;
absent an explicit setting, use 120 s. For a human step,
`config.timeout_seconds` is the separate response wait (the example uses 3600 s),
not a 300 s model/worker deadline. Preserve the existing omitted/non-positive
human-wait meaning of no response deadline. Track a finite wait from when review
opens, not when the screen mounts; navigation never extends it. Expiry blocks
Accept and downstream Note creation. Restarts restore neither waits nor counters.

No automatic retry, fallback provider, PDF/RAG expansion, server publication,
workflow synchronization, Console execution engine, background schedule, or
21-adapter rollout belongs to this first delivery. Later v1 milestones remain
in ADR-138; they have not been deleted or moved to branching/parallel releases.

## Acceptance evidence for the selected scope

- Deterministic tests of saved-snapshot isolation, typed references, unsupported
  effects, duplicate Run/Accept deliveries and permission revocation.
- Real temporary Notes/authoring databases, including failure and cancel-before/
  after-commit barriers; never infer cross-database rollback or exactly-once
  recovery from an in-memory mock.
- Retained workers through screen navigation and cancel/quit, with tests proving
  physical completion rather than merely coroutine cancellation.
- A mounted actual-app path through Run setup, editable review, rejection,
  successful Note save and Open Note, at the existing supported terminal sizes.
- An isolated-profile live run against the user's llama.cpp at `localhost:9099`,
  with the actual selected model recorded and the resulting Note read back.
  Success is the reviewed Note, not merely a healthy endpoint or HTTP response.
- Verify that a new app session never auto-resumes/replays a run, restores review
  text, or modifies historical execution/ownership rows. Two app instances must
  not share in-memory run authority; no cross-instance exclusion is claimed.
- Qualify the opt-in model path against redirect/proxy refusal, no retry, body
  bounds, slow-body absolute deadline and actual cleanup. Test unreadable/revoked
  permission authority and changed Notes destination before any effect.
- Targeted tests only, design-token/CSS checks if UI changes, unchanged performance
  budgets, and no new static-analysis debt. No whole-suite or host repair is
  authorized by this design task.

## Current checkpoint

The user selected A and approved this written design; ADR-138 records that
decision. Implementation planning is authorized. The model/Notes/permission seams
must be qualified against these requirements; their existence is not evidence of
correct execution. No runtime code, persistent run schema, live-model test or
server-conformance claim is included in this design-only checkpoint.
