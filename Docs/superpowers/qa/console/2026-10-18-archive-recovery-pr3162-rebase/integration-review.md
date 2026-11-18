# Independent integration review — PR3162 rebase

What this change does: Integrates the approved archive search, exact confirmation, restoration and explicit Open flow with fetched dev `9af0c59263dce00fa44be4a74d5d7e78712346bf`. The review covers current-dev compatibility, including native ownership and trace provenance, against both dev and original feature `fa86ea508a4c7e607dc954c0d76077bd24d50437`; it does not repeat the historical feature-wide qualification.

## Finding resolved during review

1. **[P1] Preserve the saved assistant owner when adapting thinking markers** (`Chat/console_agent_bridge.py`, `ConsoleAgentTraceRequestFactory.build`). The feature normalizes a validated incoming `_tldw_call_thinking_owner` to canonical `THINKING_OWNER_KEY`, while a retained dev block renamed the admitted canonical marker back to the alias. Equal historical assistant rows could therefore never match. The factory emitted an `ACTIVE_REQUEST` artifact instead of retaining the saved revision; `saved_response_source_revision_id` returned None, no completed-turn witness existed, and the strict service correctly rejected replacement of four tool artifacts with assistant/user rows. The real captured thinking follow-up was blocked before confirmation.

   Minimal correction approved and made by root: delete the stale reverse alias block. Keep incoming normalization restricted to exact admitted thinking owners and keep exact row equality. All service owner/source/value/range/ancestry/privacy checks remain unchanged. Actual RED: `root-alias-red.log`, 1 failed/7 passed. GREEN: `root-alias-green.log`, 9 passed (eight positive/negative owner cases plus the real `thinking-True` follow-up). Negative cases cover unknown/missing owner, content, image, extra fields, role and conflicting markers.

## Integration inspected

- Storage: chat schema82 remains immutable; chronology appends82→83. Workspace8 appends8→9. Exact recovery catalogs retain predecessor lineages and add83/9; candidate authorizers allow only fixed migration callbacks/statements and existing post-validation. Subscription hybrid catalogs include exact83 shapes. SQL package wildcards in pyproject/MANIFEST include the renamed migration. Current Library LIKE escaping is reused from `library_search_engine`.
- Native I/O: deadline acquisition extends current `SQLiteStoreKernel`/BaseDB ownership instead of resurrecting retired Workspace connection handling. Default calls retain no-keyword paths; failed setup closes newly opened handles, and expired acquisition does not close a borrowed transaction. Existing core acquisition/registration and operation-owned physical retirement remain.
- Runtime: authenticated archive provider stays separate from Library/Canvas. Names participate in reserved collision filtering. Current no-ID fenced calls use existing native correlation; archive LoopDeps fields append after current slots. Primary-only service activation, policy caps, capacity reservation, automatic-child refund and root cleanup remain. Cancellation remembered during archive close/dispose is re-raised only after current-dev cleanup finishes.
- Received input: exact opaque UI proof is issued by validated visible Send production, copied only after checked capture and accepted at the common current-owner gate after received custody validation. Generic snapshots mint no proof. Legacy queued and automatic/wake origin checks remain.
- Review extraction: archive Persona review is carried through current authoritative `console_interrupt_rounds.py`; controller remains a forwarder. Only `approve_once` grants search approval; deny/session decisions remain refusals. Real extracted-owner regression covers those outcomes.
- Trace: version2 local response source formats retain the approved closed provider/model profile; version1 meaning is preserved. Hosted finish-policy selection uses current dev generic resolver. Typed thinking tool-call rewrites keep tool-call identity; older generic thinking nodes are neither rewritten nor newly admitted. Strict completed-turn validators were not broadened.
- UI/CSS: source card styles live in current `_console.tcss`; removed legacy `screen_agentic_console.tcss` remains removed. Generated bundle derives from all65current modules. Current dev agentic-terminal source remains byte-identical. Disposable view hooks add presentation only; exact restoration and explicit original-ID Open remain separate.
- Approved recovery contracts/provider/coordinator/runtime/card/view bodies compare unchanged to original feature; query body differs only to reuse current Library escaping. Reviewed combined source/callers, worker reports, actual raw logs and current-dev attribution evidence. No other actionable P1/P2 found.

## Evidence and attribution limits

All evidence below belongs to root/helpers; the reviewer ran no tests, production imports, model/network/app calls or Git/index/ref mutations. Per-run counts overlap and must not be summed.

- Runtime report:184unique passing cases across core authority, received custody and archive dispatch cohorts, plus syntax/critical Ruff and no-new-diagnostic comparisons.
- Storage report: main254passed/6fixed import failures; query-final7passed; recovery28passed; focused2passed; final-pins21passed/2unchanged Evals guard failures. Physical worker-retirement passes. Main run warned of cumulative file-descriptor growth295; attribution was not isolated.
- UI/root raw logs include earlier fixture/bootstrap/current-source failures; final relevant reruns and final command/source hashes must be retained by root. CSS governance has3baseline failures on byte-identical dev source; Evals guard has2baseline failures on unchanged tests/allowlist.
- Initial root trace cohort:61failed/247passed.52follow-up failures are split between pre-admission fixture profile refusal and the diagnosed alias issue;4thinking PII tests similarly need the existing bootstrap profile. Root is completing only failed/unfinished cohorts after those fixture corrections.

Five additional root trace failures are independently attributed to unchanged dev behavior by AST/byte comparisons, without rerunning them:

1. Rollback observer test `[False]` and `[True]`: entire `TransactionContextManager` class and the test function are AST-identical to dev. The fake calls `__exit__` without `__enter__`, leaving current dev `_maintenance_context` unset. The rollback observer is called before the AttributeError. Do not remove current lifecycle reservation handling to satisfy this fake.
2. Oversized response and queued oversized response tests: unchanged strings are1,048,576characters, exceeding unchanged CredentialSanitizer max_text_codepoints1,000,000. Sanitizer correctly yields `credential_sanitizer_unavailable` before response-size artifact handling; tests expect the later response-size branch. Both test ASTs and settlement/redaction source are unchanged.
3. Cold restart test: unchanged repository reserves with current `created_at` (2026-10-18), but unchanged test recovers at fixed2026-08-31. The SQL inactivity grace predicate correctly excludes the reserved call; explicitly Aug30-stamped dispatch/response calls qualify. Test and recovery wrapper ASTs are unchanged; settlement/repository source is byte-identical.

Unchanged source SHA256: settlement `b9405a28630ef683f1c4f8cd86883277bf3fa337bc5cb426b8ed0ef2b4a8c367`; redaction `e2bfce0d0f35ff9e53daa8484688cd30c84dc68ddfebe94dc4262778f0ac5c4a`; repository `02223f5915839993dd6578c2fb2acfc33b62081eee0225e2664bf03e3620fd84`.

## Verdict

Source review: approve after the minimal alias correction, now verified by9targeted passes. No remaining actionable integration finding. Qualification remains conditional on root finishing failed/unfinished follow-up/UI cohorts, attributing every failure, final static/generated/source gates and preserving these limitations in PR/task notes. Root alone owns rebase completion and exact-lease publication. This report does not claim rebase/push completion.

Not checked: full suite, installed artifact build, new live app/server/model checks (30/30model requests already consumed), or final Git/remote state. Retain the10unchanged baseline test failures and cumulative descriptor warning as explicit verification limits; do not describe all targeted runs as green.

## Root qualification closure

The pending source paths now have passing evidence: remaining follow-up/PII selection50passed; all59archive-followup cases are covered across attributed runs; UI160distinct functional passes with one existing xfail, plus28CSS/token/bundle passes and3unchanged baseline failures. The late closed-source UI test omitted current production close completion's full refresh; fixture alignment passed without a production UI mutation. Ten baseline failures and cumulative descriptor warnings295/213/255 remain explicit. Root completes final source/Git/remote gates separately.
