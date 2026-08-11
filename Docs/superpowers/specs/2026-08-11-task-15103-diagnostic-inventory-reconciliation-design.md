# TASK-15103 Diagnostic Inventory Reconciliation Design

Status: approved for implementation planning

## Goal

Reconcile the persistent-diagnostic inventory drift currently present on
`dev` without using the inventory refresh to bless private values, provider or
user content, exception details, or traceback capture. Every changed owner is
reviewed under ADR-029, unsafe diagnostics are repaired without unrelated
behavior changes, and the six-file persistent-sink topology remains unchanged.

## Current verified state

The approved design was revalidated on exact `origin/dev`
`97a75fb8bf45a0fc53fc98cf18af12f5018cf458` before any branch change.

- `Tests/Architecture/test_persistent_diagnostic_inventory.py` reports 13
  passing tests and one failure: the canonical generated-versus-stored
  inventory comparison.
- Canonical regeneration changes only
  `Docs/security/production-diagnostic-inventory.json`, with 44 additions and
  30 deletions.
- The generated inventory contains 487 owner files, 1,177 TASK-492 calls,
  6,986 TASK-494 calls, and six persistent-sink files.
- The stored inventory contains 485 owner files, 1,144 TASK-492 calls, 6,962
  TASK-494 calls, and the same six persistent-sink files.
- The 23-call TASK-492 offset from TASK-15103's recorded incident totals is the
  already-reviewed TASK-3796 summarization repair. It changed both the stored
  and generated sides equally and does not alter this task's 17-owner delta.
- None of the 17 production owner paths changed between the task's recorded
  incident base and this exact `dev` revision.

The drift is limited to these owners:

1. `tldw_chatbook/Agents/agent_service.py`
2. `tldw_chatbook/Chat/console_agent_bridge.py`
3. `tldw_chatbook/Chat/console_chat_controller.py`
4. `tldw_chatbook/Chat/console_chat_store.py`
5. `tldw_chatbook/Chat/console_context_compaction.py`
6. `tldw_chatbook/Chat/console_provider_gateway.py`
7. `tldw_chatbook/MCP/client.py`
8. `tldw_chatbook/MCP/local_server_tools.py`
9. `tldw_chatbook/MCP/prompts.py`
10. `tldw_chatbook/MCP/server.py`
11. `tldw_chatbook/RAG_Search/fusion.py`
12. `tldw_chatbook/RAG_Search/simplified/rag_service.py`
13. `tldw_chatbook/RAG_Search/simplified/search_service.py`
14. `tldw_chatbook/UI/Console_Modules/session.py`
15. `tldw_chatbook/UI/Screens/chat_screen.py`
16. `tldw_chatbook/UI/Screens/library_screen.py`
17. `tldw_chatbook/app.py`

## Governing boundary

ADR-029 already owns this decision. Persistent application logs may contain
fixed operational text and metadata such as provider or operation names,
status codes, counts, lengths, durations, retry counts, and exception class
names. They may not contain prompts, messages, request or response bodies,
identifiers whose values are private, paths, filenames, credentials or key
fragments, tool payload values, exception messages, or traceback/stack
capture.

No new ADR is required. This task applies an existing privacy boundary and
does not change storage ownership, sink admission, or the metadata policy.

## Considered approaches

### 1. Audit-first atomic reconciliation — selected

Reconstruct every owner delta, classify each changed diagnostic, add a focused
fail-closed regression boundary, repair unsafe sites test-first, and regenerate
the inventory only after the source audit is complete. Keep the work in one
task and one PR because the architecture gate represents one repository-wide
review invariant.

This approach keeps the checker red until the complete review is ready, but it
prevents a partial refresh from hiding unreviewed drift and gives the final
manifest one coherent provenance boundary.

### 2. Refresh first, then repair

Regenerate the manifest immediately and audit afterward. This is faster but is
rejected because a green checker would falsely represent unreviewed private
diagnostics as accepted. ADR-029's inventory exists specifically to prevent
that workflow.

### 3. Split owners into subsystem PRs

Repair Agents, Chat, MCP, RAG, and UI owners independently. This reduces each
review diff but leaves the canonical architecture checker red between PRs and
allows the reviewed incident boundary to drift. It is reserved only for an
unexpected product decision or behavior change that cannot be resolved within
this privacy-only task.

## Architecture

### Incident reconstruction

The checked inventory is a per-owner content digest, not an individual-call
ledger. The implementation therefore reconstructs the incident from Git
history and the current AST inventory:

1. generate the current candidate inventory without accepting it;
2. prove its owner-path delta is exactly the recorded 17 paths and that sink
   topology is unchanged;
3. use content-sensitive Git history and source inspection to enumerate every
   added, removed, reworded, re-levelled, or structurally changed diagnostic;
4. record a disposition for every delta: reviewed-safe, metadata repair, or
   justified deletion; and
5. persist the complete review in
   `Docs/security/task-15103-diagnostic-review.json`.

The review artifact records the exact incident and final base revisions plus
one row for every changed diagnostic. Each row carries the owner path, stable
before and after call identities (method, content digest, and duplicate
ordinal, with `null` for an addition or deletion), introducing commit,
disposition, rationale, permitted dynamic fields, and their proven provenance.
It also records the starting and reviewed-final complete call-count/digest pair
for each of the 17 owners. Removed and rewritten historical calls therefore
remain reviewable rather than disappearing behind aggregate counts.

The final task Implementation Notes retain the exact ledger hash and reconcile
its per-owner and per-disposition totals. A mismatch, ambiguous call, missing
ledger row, or eighteenth owner stops reconciliation; it is not guessed or
silently absorbed.

### Permanent review boundary

`Tests/Architecture/test_persistent_diagnostic_inventory.py` remains the
canonical source-level guard. Its reviewed metadata map is extended for the
TASK-15103 call sites that survive the audit. Each entry identifies a fixed
diagnostic label, exact permitted dynamic expressions, and the provenance that
makes each value ADR-029 metadata rather than private content. An expression
named `status`, `provider`, `operation`, or similar is not safe by spelling
alone. It must be code-bounded, reduced to a closed enum, or proven through the
real production boundary not to carry an adversarial private sentinel. The
guard rejects:

- wholly dynamic messages plus raw positional, f-string, percent-format,
  `.format`, concatenation, keyword, or bound private expressions;
- `logger.exception`, dynamic `logger.opt(exception=...)`, stdlib `exc_info`,
  or `stack_info` capture;
- labels that are missing, duplicated, or rewritten without review; and
- fields that differ from the explicitly reviewed metadata schema.

Synthetic mutation tests must independently prove rejection of a wholly
dynamic message, positional data, f-string data, percent formatting,
`.format`, concatenation, bound data, keyword data, an exception message,
`logger.exception`, constant and dynamic `logger.opt(exception=...)`, stdlib
`exc_info`, and `stack_info`. Each mutant must fail on the assertion that owns
that syntax, not merely make some other test red.

For every permitted dynamic field whose source is not statically code-bounded,
a direct real-production-function sentinel supplies a distinctive private
value through the actual config, transport, provider, filesystem, or state
boundary. The test first proves the legitimate behavior and then proves the
sentinel is absent from captured diagnostics while the intended metadata event
remains. Tests operate on source or real production functions; no Textual
application, test application, reduced application, or simplified substitute
is constructed.

### Production repairs

For each unsafe diagnostic, the narrowest repair is applied at its existing
call site:

- replace private message content with fixed operational text;
- retain only ADR-029 metadata such as counts, lengths, status, operation,
  provider, or `type(exc).__name__`;
- replace exception/traceback-emitting methods with the same-severity
  non-capturing method where necessary;
- delete a diagnostic only when it is redundant and removal does not erase the
  sole operational state transition; and
- preserve call order, control flow, return values, raised or returned error
  text, transport payloads, cancellation semantics, and public APIs.

No helper or abstraction is added unless multiple repaired sites require the
same nontrivial policy transformation. Fixed call-site replacements are the
default.

If computing replacement metadata could add evaluation or change a historical
failure point, a direct regression test first pins the existing behavior.
Tests call the production function and substitute only its real configuration,
transport, filesystem, or provider seam.

### Inventory acceptance

The inventory is regenerated only after every source repair and focused guard
is green. The resulting manifest may change only:

- the 17 reviewed owner entries, including additions or removals already
  identified by the incident;
- the derived TASK-492 and TASK-494 totals; and
- no persistent-sink entry or topology field.

The candidate is compared against both the original branch base and the final
source. Unknown top-level data, an unreviewed owner, a changed classification,
or sink-topology movement fails closed.

The comparison uses deep copies of the complete checked and generated
documents. It removes only the exact 17 reviewed owner rows from the general
owner-list equality check, then validates those rows separately for exact path,
owner, reason, reviewed call count, and reviewed digest. Every other field,
section, list order, exclusion, classification rule, owner row, and sink row
must remain deeply equal. Both documents' summary totals are independently
recomputed from their own owner and topology rows before any normalization;
summary equality produced by the generator is not trusted as independent
evidence.

Mutation tests must make this boundary red for an unknown top-level field, a
forged derived summary, an eighteenth owner, an owner/reason classification
change on one of the 17 paths, and a persistent-sink change. Each mutant is
restored before the next, and byte hashes plus clean status prove restoration.

Immediately before the final rebase, record the complete diagnostic-call
multiset for every one of the 17 owners, not only its inventory entry. After
rebasing onto the final `origin/dev`, compare every complete population with
that checkpoint. Any added, removed, or changed call—including a new call
inside an already-authorized owner—reopens the per-call audit, provenance
review, ledger, tests, and manifest comparison before closeout.

## Error and behavior preservation

Diagnostic repair must not turn a formerly lazy operation eager, suppress or
introduce an exception, change a returned error string, change a retry or
cancellation branch, alter transaction or persistence order, or change any
provider request. When historical diagnostic formatting itself raised, the
existing externally observable contract is characterized before deciding
whether an explicit equivalent must be retained.

Fixed event wording must remain truthful for empty, missing, partially
configured, cancelled, and failed states. A label such as `configured` or
`completed` is not emitted unless the surrounding branch proves that state.

## Verification strategy

Verification remains limited to touched files and affected functionality:

1. canonical inventory generation and the persistent-diagnostic architecture
   module;
2. focused source guards for the exact reviewed calls;
3. direct production-function tests where a repair can affect evaluation,
   exceptions, ordering, transport, cancellation, or returned results;
4. the complete syntax/provenance mutation matrix plus representative
   historical private-value and traceback shapes, each failing its owning
   assertion for the intended reason;
5. independent manifest-boundary mutations for unknown data, forged derived
   totals, an eighteenth owner, classification change, and sink change;
6. ledger reconciliation proving every historical delta has exactly one
   disposition and every reviewed-final owner digest matches generated source;
7. Ruff lint and focused formatting for edited Python, `py_compile`, JSON
   parsing, and `git diff --check`; and
8. a final current-`dev` rebase followed by the complete 17-owner call-population
   comparison, the same touched-scope gates, and an exact branch-scope audit.

Repository-wide pytest is not run. Application tests, test applications,
reduced applications, and simplified application substitutes are prohibited.

## Acceptance mapping

- **AC1:** the reconstructed ledger plus TDD repairs prove that unsafe private
  values and exception details are removed without unrelated behavior change.
- **AC2:** the canonical generated manifest contains only reviewed owner
  changes and the exact original six-file sink topology.
- **AC3:** the architecture checker and focused regressions pass without an
  application harness.
- **AC4:** the durable per-call disposition ledger, provenance evidence,
  complete owner-population reconciliation, and permanent guard cover every
  generated-versus-stored delta for all 17 owners under ADR-029.

## Out of scope

- redesigning the persistent-diagnostic inventory format or sink filter;
- changing ADR-029's admitted metadata or six operational events;
- general logging cleanup outside the 17-owner incident;
- refactoring Agents, Chat, MCP, RAG, UI, or application state ownership; and
- accepting unrelated inventory drift that appears after the final rebase.
