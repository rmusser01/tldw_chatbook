# TASK-15103 Diagnostic Inventory Reconciliation Design

Status: approved for implementation planning

## Goal

Reconcile the persistent-diagnostic inventory drift currently present on
`dev` without using the inventory refresh to bless private values, provider or
user content, exception details, or traceback capture. Every changed owner is
reviewed under ADR-029, unsafe diagnostics are repaired without unrelated
behavior changes, and the six-file persistent-sink topology remains unchanged.

## Current verified state

The approved design was revalidated at the implementation stop gate on exact
`origin/dev` `82b595049d97836482c118cfeb4d31df537a86a1` from a detached
canonical regeneration. The branch was not rebased and the checked manifest
was not written.

- The stored inventory contains 485 owner files, 1,144 TASK-492 calls, 6,962
  TASK-494 calls, and six persistent-sink files.
- The generated inventory contains 488 owner files, 1,180 TASK-492 calls,
  6,990 TASK-494 calls, and the same six persistent-sink files.
- Detached canonical `--write` regeneration changes only
  `Docs/security/production-diagnostic-inventory.json`, with 53 additions and
  32 deletions. Its Git-patch SHA-256 is
  `adee369a60248da32fbc77c36b703618c73c61f5d5ef63d95460ada758f15a0f`.
- Persistent-sink topology is unchanged between stored and generated
  inventories. The exact six-file topology remains
  `tldw_chatbook/Local_Ingestion/ingest_parse_worker.py`,
  `tldw_chatbook/Logging_Config.py`, `tldw_chatbook/MCP/execution_log.py`,
  `tldw_chatbook/Utils/private_paths.py`, `tldw_chatbook/app.py`, and
  `tldw_chatbook/config.py`, with every sink kind, method, scope, and digest
  unchanged.
- Historical comparison: TASK-3796 reduced TASK-492 by 23 calls on both the
  stored and generated sides before the earlier stop gate; the three approved
  controller additions remain present, producing the current generated
  TASK-492 total of 1,180 while the stored total remains 1,144.
- The frozen prior ledger proves the approved 18-owner starting population at
  exact `85863257dd7a30b16451f8f32e0c7142dd1d5273`. Comparing every one of
  those owner count/digest pairs with latest dev finds exactly one changed
  prior owner: `tldw_chatbook/UI/Screens/library_screen.py` moves from 84
  calls/digest `c14a8222d35aec3a6e34` to 86 calls/digest
  `ae0fac2e87bf1a6ee81c`. The other 17 approved owner populations are
  identical.
- `tldw_chatbook/Utils/text_selection_crash_guard.py` is the sole new owner.
  It has no stored row and a generated row of one call/digest
  `f90a373ef5fcc81a8c1c`, owner `TASK-494`, reason
  `remaining Chatbook production diagnostic owner`.
- The three new controller diagnostics are one
  `console_visual_compaction_prepared` call and two
  `console_visual_compaction_fell_back_to_text` calls. Each binds private
  `conversation_id` plus code-bounded requested/effective representation and
  page-count/renderer-version or fallback-reason metadata. They require the
  normal ADR-029 ledger review, direct sentinels, and any ledger-proven metadata
  repair; this design does not declare them safe from shape alone.
- The new settings diagnostic is `Console appearance refresh failed after
  settings save (screen_type=%s, generation=%s, error_type=%s).` with
  expressions `type(screen).__name__`, `generation`, and
  `type(exc).__name__`. It appears metadata-only, but must still be proven and
  ledger-reviewed rather than auto-blessed.
- Commit `566b4f0ea5db6b6fd6c6c9658b1633e134e53227` adds the two Library
  diagnostics. `LibraryScreen._load_library_media_trash` calls warning with
  fixed event `Failed to load the Library media trash page.`, extracted
  expressions `module='LibraryScreen'` and `exception=True`, and
  `captures_exception=True`; this is a metadata repair, not reviewed-safe. The
  same commit adds
  `LibraryScreen._restore_library_media_from_trash`, warning with fixed event
  `Failed to restore a Library media item from the Trash view
  (error_type={}).`, expression `type(exc).__name__`, and no exception
  capture; this is reviewed-safe subject to the frozen-ledger and direct
  production-function proof.
- Commit `27779ef37108fef4f4c1bfecba22df6a2e5389bc` adds
  `TextSelectionCrashGuard.on_event`, warning with fixed event projection
  `Dropped a MouseDown that hit Textual's text-selection begin path while its
  target widget was mid-recompose (detached parent): target=,
  screen_offset=(,). Upstream Textual race (screen.py _forward_event,
  container=None) -- the click was not delivered; the app stays alive
  (task-14903).`, expressions `target`,
  `getattr(event, 'screen_x', '?')`, and `getattr(event, 'screen_y', '?')`,
  and no exception capture. Because `target` is unbounded
  `repr(select_widget)` and the interaction coordinates are not ADR-029's
  approved metadata, this call requires metadata repair and a direct sentinel;
  the source comment's safety claim is not policy evidence.

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
18. `tldw_chatbook/UI/Screens/settings_screen.py`
19. `tldw_chatbook/Utils/text_selection_crash_guard.py`

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
2. prove its owner-path delta is exactly the recorded 19 paths and that sink
   topology is unchanged;
3. use content-sensitive Git history and source inspection to enumerate every
   added, removed, reworded, re-levelled, or structurally changed diagnostic;
4. record a disposition for every delta: reviewed-safe, metadata repair, or
   justified deletion; and
5. persist the complete review in
   `Docs/security/task-15103-diagnostic-review.json`.

The review artifact separates its policy oracle from mechanically derived
acceptance evidence. Its planned contract is committed before production is
edited: exact incident/planning revisions, the starting owner pairs, and one
row for every changed diagnostic multiset atom. An atom carries the owner path,
method, full content digest, multiplicity delta, and optional qualified scope
for navigation. Rewrites are represented by a change group containing one or
more removed atoms and one or more proposed surviving atoms; additions and
deletions need only one side. The change group—not each atom—owns the single
disposition, rationale, permitted dynamic fields, and proven provenance. This
does not claim a unique identity for indistinguishable duplicate calls or treat
pure relocation as a review event when the canonical inventory deliberately
does not.

Each change group records an exact introducing commit when Git proves one, or
the narrowest verified commit range when rebases, copies, duplicates, or
intermediate rewrites make single-commit provenance non-unique. Production
repairs may not edit their own disposition, permitted fields, rationale, or
proposed surviving-call contract. If implementation proves a contract wrong,
work stops and a separately reviewed ledger-contract amendment is committed
before production resumes.

The planned-state source gate reads immutable blobs at
`incident.planning_base` through Git plumbing; it never substitutes the live
worktree for planned evidence. For every owner, the guard reconstructs the
complete `--follow` history from the stored-population match through the
planning base, calculates every introduced/removed semantic transition as the
independent denominator, and requires ledger groups to consume that transition
multiset exactly once—no missing, duplicate, or extra claim. The expanded
19-owner reconstruction is authoritative; later transition, group, and atom
counts are derived rather than copied or hard-coded from the prior ledger.

The canonical semantic atom is the compact, key-sorted JSON serialization of
`method`, `event`, `message_shape`, `expressions`, `captures_exception`, and
`level_expression` from the alias-aware `DiagnosticCall`; its digest is the
full lowercase 64-character SHA-256 of those UTF-8 bytes. Owner path is carried
by the group rather than duplicated in the digest. Qualified scope, line, and
occurrence are navigation aids only and never participate in semantic equality.
Identical semantic atoms are compared as multisets with explicit multiplicity,
so relocation is ignored while deleting or adding a duplicate remains visible.

After every repair matches the frozen semantic contracts, acceptance appends
the exact final base revision plus the reviewed-final complete
call-count/digest pair for each of the 19 owners and changes the ledger state
from `planned` to `reviewed`. The schema requires final evidence only in the
reviewed state and forbids it in the planned state, so the canonical artifact
never contains nulls, TODOs, placeholder digests, or speculative raw-source
digests. Removed and rewritten historical calls remain reviewable rather than
disappearing behind aggregate counts or false one-to-one pairings.

The otherwise exact top-level schema permits `integration_checkpoint` only on
a reviewed ledger during final integration. Its pre-rebase half is committed
before rebasing and its post-rebase half is required afterward; both contain
exact base/HEAD, aggregate semantic-multiset count/digest, and every owner's
count/digest. Unknown checkpoint keys or an invalid pre/post lifecycle fail.

The final task Implementation Notes retain the exact ledger hash, reconcile
change-group totals by disposition, and separately reconcile atom multiplicity
by owner and before/after side. A mismatch, ambiguous call, missing ledger row,
or twentieth owner stops reconciliation; it is not guessed or silently
absorbed.

### Permanent review boundary

`Tests/Architecture/test_persistent_diagnostic_inventory.py` remains the
canonical source-level guard. Existing pre-TASK-15103 entries stay in their
current reviewed map; TASK-15103 policy data lives only in
`Docs/security/task-15103-diagnostic-review.json`, and the guard loads the
surviving reviewed calls from that artifact instead of duplicating their
labels, fields, and provenance in a second constant.

The guard reuses the alias-aware diagnostic-call extractor in
`Tests/LLM_Calls/summarization_diagnostic_guard.py`; it does not grow a second
shallow parser for dynamic message forms, logger aliases, `.bind()`, `.opt()`,
or exception capture. Task 2 also closes the proven extractor gap for aliases
introduced or mutated in `try`, `for`, `while`, `with`, and `match`: joins must
be conservative across control-flow paths, shadowing and reassignment must not
retain stale logger identity, and focused mutation tests must own each case.
Each TASK-15103 ledger entry
identifies a fixed diagnostic label, exact permitted dynamic expressions, and
the provenance that makes each value ADR-029 metadata rather than private
content. An expression named `status`, `provider`, `operation`, or similar is
not safe by spelling alone. It must be code-bounded, reduced to a closed enum,
or proven through the real production boundary not to carry an adversarial
private sentinel. The guard rejects:

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

Rendering a value that ADR-029 prohibits from persistent diagnostics is not a
supported behavior contract: the repair must remove that rendering even when a
custom value's `__str__` has side effects or raises. The task does not recreate
an implicit logging-time failure merely to preserve prohibited evaluation. It
does preserve the surrounding method's branch selection, state mutation,
returns, raised or returned operational errors, ordering, and collaborator
calls.

If computing replacement metadata could add evaluation or change any of those
legitimate behaviors, a direct regression test first pins the existing
contract. For a method owned by `TldwCli` or another framework class, the test
may invoke the exact unbound production method as a function with a narrow
state record and signature-checked real collaborator seams. That is a function
unit test: it does not instantiate, subclass, mount, or run an application.
Tests otherwise substitute only the real configuration, transport, filesystem,
or provider seam.

### Inventory acceptance

The inventory is regenerated only after every source repair and focused guard
is green. The resulting manifest may change only:

- the 19 reviewed owner entries, including additions or removals already
  identified by the incident;
- the independently derived owner-file, TASK-492, and TASK-494 totals,
  including the known 485-to-488 owner-file transition; and
- no persistent-sink entry or topology field.

The candidate is compared against both the original branch base and the final
source. Unknown top-level data, an unreviewed owner, a changed classification,
or sink-topology movement fails closed.

The comparison uses deep copies of the complete checked and generated
documents. It removes only the exact 19 reviewed owner rows from the general
owner-list equality check, then validates those rows separately for exact path,
owner, reason, reviewed call count, and reviewed digest. Every other field,
section, list order, exclusion, classification rule, owner row, and sink row
must remain deeply equal. Both documents' summary totals are independently
recomputed from their own owner and topology rows before any normalization;
summary equality produced by the generator is not trusted as independent
evidence.

Mutation tests must make this boundary red for an unknown top-level field, a
forged derived summary, a twentieth owner, an owner/reason classification
change on one of the 19 paths, and a persistent-sink change. Each mutant is
restored before the next. A non-equal mutant hash proves the mutation occurred;
restored byte hashes and equality to the recorded legitimate pre-mutation diff
prove restoration without falsely requiring a clean worktree before the
intended manifest/ledger changes are committed.
Except for the mutant that deliberately forges a summary, each mutated document
must first recompute its owner-file, TASK-492, TASK-494, and sink-file totals so
it is internally consistent. Every mutant must then fail the assertion that
owns its intended invariant—with a specific unknown-field, unreviewed-owner,
classification, or topology failure—not a stale-summary check or another
incidental assertion.

Immediately before the final rebase, commit to the canonical ledger the
complete canonical-semantic-atom multiset hash and per-owner count/hash for all
19 owners, not only each inventory entry. Scope/line/occurrence remain external
navigation evidence and do not affect equality. The committed checkpoint keeps
the rebase worktree clean and makes the comparison reproducible. After rebasing
onto the final `origin/dev`, append the post-rebase per-owner evidence and
compare every complete semantic population with the checkpoint. Any added,
removed, rewritten, re-aliased, multiplicity-changing, or capture-changing
atom—including a new call inside an already-authorized owner—reopens the
per-call audit, provenance review, ledger, tests, and manifest comparison before
closeout. Pure relocation does not.

## Error and behavior preservation

Diagnostic repair must not turn a formerly lazy operation eager, suppress or
introduce an exception, change a returned error string, change a retry or
cancellation branch, alter transaction or persistence order, or change any
provider request. The one explicit exception is an error caused solely by
rendering an ADR-029-prohibited diagnostic value: removing that rendering and
its logging-time failure is required privacy behavior, not a regression, and
no equivalent failure is retained. A focused characterization may prove that
the historical failure was isolated to prohibited rendering and that all
surrounding legitimate behavior remains unchanged.

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
   totals, a twentieth owner, classification change, and sink change;
6. ledger reconciliation proving every historical delta has exactly one
   disposition and every reviewed-final owner digest matches generated source;
7. Ruff lint and focused formatting for edited Python, `py_compile`, JSON
   parsing, and `git diff --check`; and
8. a final current-`dev` rebase followed by the complete 19-owner call-population
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
  generated-versus-stored delta for all 19 owners under ADR-029.

## Out of scope

- redesigning the persistent-diagnostic inventory format or sink filter;
- changing ADR-029's admitted metadata or six operational events;
- general logging cleanup outside the 19-owner incident;
- refactoring Agents, Chat, MCP, RAG, UI, or application state ownership; and
- accepting unrelated inventory drift that appears after the final rebase.
