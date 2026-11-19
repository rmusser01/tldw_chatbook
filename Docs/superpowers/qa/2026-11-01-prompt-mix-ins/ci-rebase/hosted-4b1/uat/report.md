# PR3346: approval-row readiness correction

Diagnostic published head: `4b1e070c972320e1353b6bac7e185035db125571`.
Hosted run `41547453582`, attempt 1, admission-2 job `125628364692`.
Task/ADR/status publication belongs to root; no Git/index/branch/commit/push or external-service mutation was performed by this worker.

## Cause and change

The hosted wrong-family case failed its original `Built-in` / `list_directory` assertion with `header == ""`. Its receipt records exactly three model requests, legitimate approval waiting, no model errors, and no unexpected network destinations. The original predicate waited only for queryable `.approval-row` parents.

The native approval owner synchronously hands pending header children to a Vertical, then calls `rows_container.mount(*rows)` without awaiting. Installed Textual registers the row before that row's message pump dispatches Compose and Mount; `_is_mounted` becomes true only after those dispatches. The deterministic native-card probe observed one queryable row, zero header children and `is_mounted=False` immediately after original `set_batch`; one native Pilot turn produced the header. This is mechanism proof, not a replay of the hosted scheduling order. The unmodified exact full-app wrong-family UAT passed locally in 33.02s.

The test now waits until every queried approval row is mounted. It returns the complete row list and never examines header contents, names, or cardinality. All original UAT assertions, request/call arguments, real Approve Once action, waits and timeout expressions are retained. No production code, provider, schema, permissions, performance/size ceiling, workflow or census changed. Existing ADR103 evidence contract applies; this routine test readiness correction requires no new ADR.

The permanent control uses a real Textual App and original ChatApprovalCard.set_batch. It proves empty DOM is unready, two published parents with no header children remain unready, and both same rows become ready after native mounting. A deliberately unexpected nonempty header remains ready and still fails the original expected-content expression. The existing whole-file support test selection enrolls it without changing the exact four-case receipt validator.

## Verification

| Evidence | Result and qualification |
| --- | --- |
| Initial default-sandbox original UAT | Loopback PermissionError before app/model work; setup failure, not native/product RED |
| Original full-app wrong-family UAT, authorized loopback | 1 passed in 33.02s; hosted schedule did not naturally reproduce |
| Permanent native-card control, original factored predicate | Intended RED at pre-mount readiness assertion: 1 failed in 1.16s; two rows/zero headers already proved |
| Same permanent control, mounted predicate | 1 passed in 0.89s |
| Final original CI three-file cohort, uninstrumented | 74 passed in 156.49s; all four actual-app UAT variants plus support and report controls |
| Independent complete-receipt checker | Exit 0 against final original JUnit plus four complete receipts/screens |
| Ruff / format | Zero added diagnostics: same three inherited findings; new helper formatted, complete support file formatter-clean; inherited UAT full-file formatting debt retained |
| Contract comparison | Every original UAT assertion and every wait_for/timeout call AST-identical to 4b1 |
| Frozen source | All 3026 product Python/TCSS plus modified-test source paths unchanged across final native cohort |
| Fixture/report/workflow provenance | Six selected fixture/helper/validator/workflow owners byte-identical to 4b1 |
| Independent source review | CLEAR, no source findings; final evidence review requested separately |

The wrong-family receipt completed in 41.642s with native Approve Once for list_directory, final request_count 6, persisted run/chat/assistant identities and no model/network errors. The unchanged approval-stage assertion still checked request_count 3 before the click.

## Exact sources and evidence

- `Tests/QA/test_project_chat_uat.py`: SHA256 `bff4182da1994ceb691a644f7c458fd3fc321d64bf58df51be60670eac985681`.
- `Tests/QA/test_project_chat_uat_support.py`: SHA256 `8f22db83a9ebf3e3c5f2921894f838b1e745b181f07d6fa77c2f48c4c5a93329`.
- `commands.json` records exact Python/pytest argument vectors and artifact environment variables.
- `hosted-provenance.json` identifies original failure artifacts and raw-job-log hash; original hosted receipt/screens copied without modification.
- `lint-format-contract.json`, `support-source-provenance.json`, `frozen-source-sha256.json` and `final-proof.json` record comparisons and final receipt facts.
- `artifact-inventory.json` contains only explicit bounded evidence paths and hashes; profile directories, databases, child debug logs, intermediate stack snapshots and unrelated files are excluded. Originals remain in this private directory.

No full suite, additional performance run, or real-provider/billing check was performed. Fresh published-head protected CI and actual merge remain root-owned; this local result makes no hosted-green or merge-complete claim.
