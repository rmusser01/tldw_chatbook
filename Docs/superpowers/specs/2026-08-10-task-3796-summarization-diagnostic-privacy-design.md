# TASK-3796 — Summarization diagnostic privacy design

- Date: 2026-08-10
- Status: implemented and verified
- Backlog: [TASK-3796](../../../backlog/tasks/task-3796%20-%20Remove-private-summarization-values-from-diagnostics.md)
- Existing decision: [ADR-029: Local Private Data Boundary](../../../backlog/decisions/029-local-private-data-boundary.md)
- ADR required: no
- ADR path: `backlog/decisions/029-local-private-data-boundary.md`
- ADR reason: ADR-029 defines the persistent-log boundary: persistent records may retain bounded operational metadata, but not user/model content, key fragments, private values, or exception messages. TASK-3796 applies the same allowlist at the leaking call sites as a scoped defense-in-depth repair, without changing the global contract for other UI/terminal diagnostics, storage, sink admission, provider contracts, or another cross-module interface.

## Outcome

The two summarization modules retain useful operational diagnostics without submitting private summarization values to either logging implementation they use. The repair covers every one of the 200 verified direct-private diagnostic sites and prevents equivalent new sites from entering either module unnoticed.

The summarization functions keep their existing inputs, outputs, streaming behavior, retry behavior, exception handling, and user-visible error strings. This is a diagnostic-data repair, not a provider or application-state refactor.

## Verified starting point

TASK-2118's final review replaced an identifier-filtered candidate list with a complete review of all 523 logger calls in:

- `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py` — 242 calls, including 100 direct-private diagnostics;
- `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py` — 281 calls, including 100 direct-private diagnostics.

The 200 sites comprise:

| Category | Local | General | Total |
| --- | ---: | ---: | ---: |
| Raw, processed, or extracted input | 13 | 8 | 21 |
| Prompt content | 8 | 9 | 17 |
| Credential fragments | 8 | 13 | 21 |
| Private endpoint or path values | 6 | 5 | 11 |
| Response or generated-output content | 29 | 43 | 72 |
| Exception messages or error detail | 36 | 22 | 58 |
| **Total** | **100** | **100** | **200** |

### Final-review audit correction

The approved pre-implementation audit originally recorded `199 private + 324 reviewed-safe`. Independent final review found that stable site `general-2efc909241862caf` renders the provider-controlled Cohere stream event type and had been incorrectly frozen as status metadata. Under this design's approved misclassification procedure, the authoritative starting population is corrected to `200 private + 323 reviewed-safe = 523`: General is `100/181`, `general_mid` is 24, and response/output content is 72 overall. Historical batch results that quote `199/324` describe the ledger before this correction and remain evidence of why a late review-fix tranche was required.

The task's verified inventory is authoritative for the pre-implementation sites. Its line numbers are navigation aids. Stable identity is the module, qualified enclosing function, fixed diagnostic event/label, category, and an occurrence ordinal only when those fields would otherwise collide.

The current persistent sink rejects ordinary Chatbook records that do not pass through its metadata admission helper. The inventoried private values are nevertheless formatted and submitted to stdlib logging or Loguru, where non-persistent handlers can observe them and a later sink change could admit them. This task removes those values at the narrower source boundary; it does not claim they all currently reach the persistent file.

The focused pre-change baseline on current `origin/dev` is 37 passing tests across:

- `Tests/LLM_Calls/test_summarization_analyze.py`;
- `Tests/Chat/test_cohere_summarize_v2.py`;
- `Tests/Internal_Prompts/test_summarization_migration.py`;
- `Tests/Internal_Prompts/test_summarization_prompt_parity.py`.

## Goals

1. Replace every inventoried private diagnostic value with operationally useful metadata, or remove the diagnostic when it conveys no useful state without the private value.
2. Prevent private values from being formatted before the logger boundary, including values discarded by the current persistent-sink filter.
3. Preserve diagnostic severity and placement when the event remains useful.
4. Prove the repair through direct production-function tests, exhaustive source reconciliation, mutation checks, and the production diagnostic inventory.
5. Keep the change atomic to the two summarization owners, their focused tests and verification data, and the task/spec/plan/implementation records.

## Non-goals

- Do not change summarization output, retry policy, provider routing, request construction, streaming protocol, or user-visible error contracts.
- Do not add a new logging wrapper, sanitizer, dependency, configuration flag, support mode, sink, or persistent event.
- Do not route ordinary summarization logs through `persist_event()` or otherwise widen persistent-sink admission.
- Do not decompose application state or refactor the provider implementations beyond the diagnostic expressions required by this task.
- Do not repair unrelated diagnostic-inventory drift by blessing it into this branch.
- Do not use a test application, a reduced application, or a simplified version of the application. Tests invoke the affected production functions directly and replace only their transport/config seams.
- Do not run the repository-wide test suite. Verification is limited to files and functionality reached by this change, plus the cross-cutting diagnostic-inventory contract the change necessarily affects.

## Selected approach

Apply direct, one-for-one call-site repairs. Each private argument is removed before the logging call and replaced, where useful, with fixed event text and a deliberately small metadata schema. Delete a log only when the private value was the whole diagnostic and no safe metadata would help identify an operational state.

This is deliberately not a new abstraction. The two modules already use different logging imports and numerous provider-specific control-flow shapes. A new wrapper would add a second diagnostic policy surface without eliminating the need to inspect and edit every leaking expression. Direct substitutions make the privacy property visible at the actual source and keep behavior reviewable.

### Allowed metadata for the 200 repaired sites

A replacement for an inventoried private diagnostic may contain only fields justified by the event:

- a fixed, code-authored event label;
- integer counts and lengths, such as input characters, chunk count, response bytes, or retry number;
- booleans, such as streaming state;
- HTTP status codes;
- fixed provider/backend identifiers already implied by the function;
- a bounded exception class name, never an exception instance or message;
- another bounded identifier only when it passes the existing `safe_metadata_token()` boundary.

Prefer fixed labels and numeric/boolean fields. Do not add a dynamic provider/model/type token merely to preserve the shape of an old private log. If a replacement needs a dynamic string token, pass it through `safe_metadata_token()` before it reaches the logger; the helper's fixed `invalid` result is the only fallback. Raw `type(value).__name__`, model names, provider names, object class names, and arbitrary string values are not automatically trusted metadata for a replacement.

The other 323 logger calls were reviewed and excluded from the private-site inventory because their current values are fixed, type, length, status, count, provider/model, or other bounded operational metadata. This task does not mechanically rewrite those calls to the stricter replacement style. Instead, the exhaustive guard freezes each reviewed call's current normalized expression structure and rationale. That exact legacy-safe structure is not a general allowlist: a new call or any changed dynamic expression fails until it is reviewed against the strict schema above or assigned to a separately approved scope.

### Forbidden values and operations

No repaired diagnostic may contain or pre-format:

- raw, normalized, processed, decoded, or extracted user input;
- system, custom, combined, or provider prompt text;
- model response bodies, parsed response objects, generated summaries, streamed lines/events, or other output content;
- API keys, bearer tokens, credential prefixes/suffixes, or any other credential fragment;
- endpoint URLs, hostnames, filenames, or local/private paths;
- exception instances, `str(exception)`, exception messages, response error details, or arbitrary object representations;
- `response.text`, decoded stream data, provider payload dictionaries, or values derived from those bodies;
- tracebacks through `logging.exception()`, `logger.exception()`, `exc_info=True`, Loguru exception capture, or an equivalent mechanism.

The rule applies before the logging call. Building an f-string, `%` expression, `.format()` result, string concatenation, exception string, object representation, or decoded response value and then passing that string to a logger is still a violation even if a sink later filters the record.

### Category-specific substitutions

| Existing private category | Safe replacement shape | Deletion rule |
| --- | --- | --- |
| Input | fixed load/extraction event plus type-independent count/length when already available | delete a duplicate value dump if an adjacent safe lifecycle event already proves the same state |
| Prompt | fixed prompt-built event plus character/message count when useful | delete a prompt preview whose only purpose was content inspection |
| Credential fragment | fixed credential-configured/source-selected event | delete duplicate key-prefix diagnostics; never log fragments or dynamic source paths |
| Endpoint/path | fixed endpoint-selected, input-file-detected, file-read-failed, or summary-saved event | delete a path echo when success/failure is already represented safely |
| Response/output | status code, response byte count, stream event kind from a fixed allowlist, or fixed parse/summarize event | delete raw previews and duplicate body dumps |
| Exception/error detail | fixed failure event plus bounded exception class and already-available status/count metadata | remove traceback capture and delete duplicate exception-message logs |

Safe replacements preserve the original log level unless the old call is deleted as redundant. No replacement may change a branch condition, return value, raised exception, retry decision, yield sequence, or user-facing error string.

## Data and control flow

The provider function continues to construct requests, invoke its transport, parse responses, and return or yield the same data. Only the side-channel into diagnostics changes:

```text
private runtime value ──> provider/summarizer control flow ──> unchanged result/error
                  ╲
                   ╲ old: formatted private value ──> logger
                    ╲ new: fixed event + bounded metadata ──> logger
```

Where the old diagnostic is deleted, the main control-flow edge remains untouched. The implementation must not catch a broader exception, move a return/yield, consume a response earlier, or introduce work solely for logging. Streaming functions must remain lazy; tests must consume their generators to exercise logging that occurs after the first yield boundary.

## Exhaustive source boundary

The permanent guard must enumerate every stdlib-logging and Loguru call in both modules, including nested functions and bound forms such as `logger.opt(...).error(...)`. It may use AST extraction plus reviewed source context, but it may not claim completeness from identifier names such as `data`, `payload`, `prompt`, or `response`.

The guard records stable call identity as module, qualified enclosing function, fixed event label, privacy category/classification, and occurrence ordinal where required. It does not key expectations to line numbers. It reconciles two explicit classes:

1. the 200 inventoried sites, each recorded as an approved strict-schema replacement or an intentional deletion; and
2. the 323 reviewed-safe calls, each recorded with its exact normalized dynamic-expression structure and its existing safe classification.

An unclassified call, a changed expression in the frozen reviewed-safe class, or a replacement outside the strict schema fails closed. The implementation does not change a previously reviewed-safe call merely to make its style uniform. If the fresh all-call review finds that one was misclassified, update the verified task inventory and obtain approval before expanding production scope. The starting population, the two classes, and the deleted-site ledger must reconcile arithmetically and by stable identity.

Structural rejection includes, at minimum:

- dynamic message templates or arbitrary f-strings;
- exception variables/messages and traceback capture;
- body/content/object renderings;
- endpoint/path and credential values;
- dynamic expressions not present in the reviewed approved-field schema.

The source guard is one part of the proof, not a substitute for runtime behavior. It is paired with direct-function sentinel tests, mutation checks, and the existing module-level diagnostic digest. The guard itself must be mutation-tested by restoring representative forbidden expressions and confirming that it fails on the intended site.

## Test design

Add focused tests under `Tests/LLM_Calls/` that call the real affected functions. Replace only external transport, file, configuration, sleep, or provider-client seams needed to deterministically enter a path. Do not construct any Textual application or synthetic substitute application.

### Runtime sentinels

For each of the six privacy categories in each module, exercise at least one representative production path with a category-specific, distinctive canary. Capture both stdlib and Loguru output where the production path can use either logger. Assert after the function returns, raises, or its stream is fully consumed that:

- the private canary does not occur in any captured record;
- the expected fixed event remains when the design retains that diagnostic;
- only the declared metadata fields occur;
- exception tests expose at most the bounded class token, never the message or traceback;
- stream tests consume the generator and cover decode/error paths after iteration begins;
- return values, raised exception types, yielded chunks, and user-visible error text match the characterized pre-change behavior.

The 12 category/module cases are a minimum matrix, not permission to ignore provider-specific shapes found by the exhaustive guard. Add a targeted runtime case when a distinct logging mechanism or control-flow form is not represented by that matrix.

### Mutation evidence

Mutation checks must prove the tests and source guard can detect the repaired behavior. At minimum, independently restore one private diagnostic for every category in each module, run the narrow test that owns that sentinel, observe the expected red result, restore the production file, and rerun green. Mutations must be independent and restoration must be proven by Git blob/hash comparison plus a clean diff for the restored files. Use cache-disabled Python (`-B`) so stale bytecode cannot satisfy a mutation run.

If one runtime path proves multiple categories, each mutation still restores only one category at a time. A red result caused by a different sentinel or by setup failure does not count.

### Focused regression set

The final plan derives exact node IDs after the new tests exist. The required boundary is:

- the new summarization diagnostic privacy tests and exhaustive source guard;
- `Tests/LLM_Calls/test_summarization_analyze.py`;
- `Tests/Chat/test_cohere_summarize_v2.py`;
- affected tests in `Tests/Internal_Prompts/test_summarization_migration.py` and `Tests/Internal_Prompts/test_summarization_prompt_parity.py` when their production paths are reached;
- `Tests/Architecture/test_persistent_diagnostic_inventory.py`;
- Ruff lint and format checks limited to edited Python files/ranges;
- `py_compile` for edited Python files;
- `git diff --check` and exact branch-scope checks.

No test application and no repository-wide pytest run are part of this task.

## Production diagnostic inventory

Regenerate the diagnostic inventory only after the all-call and runtime privacy gates are green. The task may change only the two owner entries for:

- `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py`;
- `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py`.

Their diagnostic digests must change. Their call counts may change only for explicitly reviewed redundant-log deletions. Owner paths, TASK-492 ownership reason, and sink topology remain unchanged. Every unrelated owner entry, reason, count, digest, and topology record must be byte-for-byte identical to the current-dev comparison boundary.

If the architecture checker is already red on current `origin/dev`, run the identical command in a clean worktree at that exact commit and compare normalized failure sets. Do not regenerate or commit unrelated drift. Assign any independently actionable baseline drift to a separate, already-created task before closeout.

## Rebase and drift handling

Before implementation and again before integration:

1. fetch and rebase onto the latest `origin/dev`;
2. prove the branch is a descendant of that exact commit;
3. compare the two production modules and focused tests against the reviewed base;
4. rerun the complete all-call extraction and reconcile any added, removed, or changed logger call;
5. rerun the focused runtime, source, and diagnostic-inventory gates.

Upstream additions are reviewed under the same privacy policy. Counts in this design are evidence for the verified starting commit, not a reason to discard a legitimate current-dev change.

## Scope and deliverables

Expected production scope:

- `tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py`;
- `tldw_chatbook/LLM_Calls/Summarization_General_Lib.py`.

Expected verification/documentation scope:

- focused direct-function privacy tests and their stable all-call guard data;
- `Docs/security/production-diagnostic-inventory.json` for exactly the two owners;
- TASK-3796, this design, and its implementation plan/notes;
- a testing/privacy lesson only if implementation uncovers a new generalizable incident.

No other production module is in scope without first amending TASK-3796 acceptance criteria and obtaining approval.

## Alternatives considered

### Add a shared summarization logging wrapper

Rejected. It would create another policy owner, require a broad mechanical rewrite, and still require reviewing all 200 leaking sites. The existing `safe_metadata_token()` is sufficient for the few dynamic identifiers that genuinely need to survive.

### Remove every diagnostic in the two modules

Rejected. It is privacy-safe but discards useful status, retry, parse, and transport metadata. ADR-029 intentionally preserves bounded operational diagnostics.

### Rely on the persistent-sink filter

Rejected. Private strings are formatted and submitted to logging before sink admission, can reach non-persistent handlers, and can be resurrected by a future sink change. Containment belongs at the call site.

### Use an identifier-based static search as the acceptance proof

Rejected. TASK-2118 demonstrated that such a search missed private diagnostics whose variables had unexpected names. The proof must begin from every logger call and explicitly classify every dynamic expression.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Private text survives under an unexpected variable name | Exhaustive all-call enumeration and reviewed expression classification, not an identifier denylist |
| A replacement leaks during eager formatting | Reject dynamic message construction and forbidden pre-formatting before the logger boundary |
| Removing exception text changes user-visible behavior | Assert logging separately from the existing return/raise/yield contract |
| A streaming test never reaches the leaking code | Fully consume the real generator and mutation-test the exact path |
| A replacement's dynamic metadata token contains user text | Prefer fixed/numeric metadata; otherwise require `safe_metadata_token()` and accept its fixed `invalid` result |
| A broad guard turns the task into cleanup of reviewed-safe legacy calls | Freeze their exact reviewed structures separately; apply the new strict schema only to inventoried replacements and future changes |
| Manifest regeneration blesses unrelated drift | Compare exact current-dev and branch inventories, permitting only the two owner entries |
| Line-number churn invalidates the guard | Key by module/function/event/category/occurrence; retain lines only as navigation aids |
| A test passes without observing the changed call | Independently restore each category/module leak and require the owning assertion to fail |

## Completion conditions

TASK-3796 may be marked Done only when:

1. all 200 starting sites reconcile to an approved metadata-only replacement or an explicitly justified deletion;
2. every category in both modules has non-vacuous direct-function sentinel coverage;
3. the complete all-call guard reports no unclassified or private expression;
4. mutation evidence proves each category/module boundary can fail;
5. focused behavioral/static gates pass, with no application harness and no repository-wide suite;
6. the diagnostic inventory changes only the two owned entries and any call-count change matches reviewed deletions;
7. the branch is current with `origin/dev`, the task has checked acceptance criteria and concise implementation notes, and a whole-branch self-review finds no unresolved issue.
