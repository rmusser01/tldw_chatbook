# First sequential workflow: execution design checkpoint

Status: Draft. The five-step example is approved; the execution-lifetime choice
below is not. No runtime implementation or restart guarantee is approved by this
document. TASK-32690 remains In Progress until that choice and the final written
design are reviewed.

ADR required: yes, amend the delivery scope in existing ADR-138 after approval.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md.
Reason: session-bound delivery would defer ADR-138's durable run/wait/recovery
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
do not import that restriction accidentally or relabel this endpoint as Ollama.
Configured custom endpoint identity and the `llama_cpp` transport family are
different values. Keep the selected local target visible and bound to the run.

## Choice requiring user approval

### A. Session-bound first delivery — recommended

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

This option adds no run schema, database owner, lock file, PID record, helper
protocol, recovery scanner or persistent execution service. Existing SQLite
utilities still operate normally for authoring and Notes; this is not a claim
that the application uses no existing helper processes.

### B. Include restart-resumable runs in this delivery

Persist review edits and run identity, recover a waiting review after restart,
and reconcile uncertain effects durably. This retains the wider historical
milestone, but requires a separately explicit ownership/claim/recovery design
against current dev, plus multiprocess and abrupt-exit evidence. Existing tables,
UUIDs, or an in-memory flag do not establish exclusive physical execution.

Do not silently reintroduce the withdrawn implementation to obtain this behavior.
If this option is selected, resolve its ownership design before any runtime port.
The recommendation of A defers these guarantees; it does not demonstrate that B
is impossible without new infrastructure.

Restoring the complete parked runtime is not recommended: it couples this one
example to obsolete service signatures and the excluded ownership machinery.

## Implementation boundary after the choice

For A, the intended shape is one lazy app-owned session coordinator, with a small
closed dispatcher for the five validated operation subsets and the existing
editor/navigation surfaces. Reuse the bounded expression code and current
domain services. Reuse parked tests for relevant behavior, not their ownership
assumptions or broad source files. Do not build a generic provider/plugin/runtime
framework, replace the SQLite factory, or copy the old run service wholesale.

The implementation plan must pin the exact transport and off-loop Notes seam,
finite input/output/request bounds, structured effect-time permission handling,
and application shutdown order before code starts. Blocking service work must
not be placed directly on Textual's event loop. These are unresolved integration
details to finish after the lifetime decision, not implementation instructions.

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
- For A, verify that a new app session never auto-resumes/replays a run. For B,
  add the separately designed restart, competing-owner and uncertain-write tests.
- Targeted tests only, design-token/CSS checks if UI changes, unchanged performance
  budgets, and no new static-analysis debt. No whole-suite or host repair is
  authorized by this design task.

## Current checkpoint

The approved five-step flow and source audit are captured here. The user must
select A or B before finalizing the runtime design and any ADR amendment. The
brainstorming design gate is still active; implementation planning has not begun.
