# SQLite-inside-config final independent review

APPROVED at the exact hashes below. No remaining actionable findings in the bounded product/test/selection delta against HEAD f104d61f5b. This repairs independently admitted default-native SQLite initialization inside an active raw config scope; no custom-factory expansion, native permission exception, cache or deadline change.

## Independent evidence

All12 new native cases passed in8.51s, exit0: /private/tmp/uat-sqlite-config-independent-final.log. Unique disposable fixture root /private/tmp/uat-sqlite-config-independent-final-fixtures. Existing dependency warning only. No TldwCli app boot, network access or repository edits by this reviewer.

Parent evidence inspected/reported separately: corrected9 cases6.43s; failed-close-inclusive10 cases6.96s; final12+existing21 Console cases33PASS32.69s (/private/tmp/uat-sqlite-config-final.log), selected compatibility11PASS2.29s (/private/tmp/uat-sqlite-config-compat.log). Parent's actual installed Console run is separate and pending at this review; no Windows acceptance claimed. Earlier wrong _SQLITE_CONNECT spy outcomes were test-seam mistakes, not product RED. Actual paired-failure RED2fail5.53s preceded the precedence correction.

## Authority and restoration

_independent_sqlite_helpers runs only after the existing connector obtains independent SQLite admission/resource attachment, and only for the default sqlite3.Connection factory outside capture. It validates the existing raw runtime operation, suspends its thread-local discovery during SQLite's trusted native connector work, then checks/restores the exact prior operation. Outer raw registration, source locks, pins, leases and native holds remain live throughout. Core scope discovery is untouched. All existing directory/ACL/path/private artifact and post-open checks execute unchanged. Custom/capture/descriptor/memory/foreign-read branches retain existing behavior.

The real config route revalidates the exact selector: raw._check -> _participant_state(owner=config) -> config_participants.binding -> config._get_effective_config_path, comparing the resulting lexical file against participant.selected. Same-directory config.toml→other.toml therefore refuses; sibling-anchor checks are not needed for this route. Direct getter coverage is important because repository-operation-only suspension would fix the observed count path but miss getters with no outer core operation.

## Error and close accounting

The final helper follows _owned_helper's primary/cleanup precedence. An ordinary restoration error cannot replace an active connector error or cancellation; a fixed note is added through BaseException.add_note, avoiding an overridden add_note callback. With no primary, restoration refusal propagates. Pointer restoration always occurs in finally. A secondary control-flow exception retains existing helper precedence. Paired native error/cancellation + selector-change cases verify primary identity and the fixed note, with no returned native handle or leaked new lease.

If successful connector allocation is followed by restoration refusal, only the exact new AdmittedConnection is closed. constructing remainsTrue while close positively closes the underlying native handle, then allocation_started=False lets the existing outer exception branch retire the independently acquired lease. Publication into _ordinary_connections/repository registration has not occurred yet. If close fails, allocation_started remainsTrue and the existing uncertain resource_close_failed lease is retained. The native failed-close test demonstrates SELECT still works on the retained object, the extra lease remains live, and maintenance cannot drain; the disposable child exits without clearing the uncertainty or forcing retirement. No pre-existing borrowed connection is closed.

## Test and runner boundaries

The tests call real AgentRuns native count/getter APIs under actual config operations; they do not replace admission, private helpers, scope checks or native close accounting. The corrected sqlite3.connect wrapper observes the real connector seam and injects only controlled failures/pause/selector transitions. Cached/reopened/direct-getter, pause-before/during, ordinary error/cancellation, simultaneous selector errors, positive close, failed close and unchanged custom behavior are covered. Existing custom callbacks stay out of the new suspension interval.

Runner AST comparison proves the only changes are one addition of the12-case file to _PRODUCT_TESTS and one to _SUPPORT_DIAGNOSTIC_TESTS. Removing those literals restores full AST equality with HEAD; selection names, prior cases, assertions and all budgets are unchanged. Diff check clean.

Limits: this review verifies the native local regression and preservation contracts, not resolution of separate Windows Library/Settings timing failures. Installed Console and Windows execution remain acceptance steps owned by the parent.

Exact SHA256 (final static-only bytes):
- `tldw_chatbook/DB/private_sqlite.py`: `40b1515de3676d304b348203b3adab0b057deab9c407b6763d500af7d9f8a00d`
- `Tests/Backup_Recovery/test_sqlite_inside_config_scope.py`: `4fe0ad2b721b00406d104e6564adb7c665bd38bae4c3d3f7fdcb88bedbfcd8a2`
- `Tests/Backup_Recovery/run_platform_product.py`: `3decfc624f51efa288153277ebdda49d603e0a86e5bf817d8370b51e641cf24b`

Final formatting acknowledgment: reconstructing prior product bytes by removing only the explanatory noqa comment reproduced reviewed a2d00b63... exactly. Undoing only test parentheses/comment/import blank-line formatting reproduced reviewed5188d948... exactly. Both module ASTs, including the raw child literal, are identical. Proof: /private/tmp/uat-sqlite-final-format-verification.json. Approval retained; no native rerun for these nonbehavioral changes.

Parent reports actual installed Console PASS53.21s (/private/tmp/uat-sqlite-config-installed.log), with verified-package.json recording2474 exact Python sources,2867 total files and the sole product comment-only AST match. This is parent-run installed evidence, separate from independent12 native cases; native Windows acceptance remains pending. Parent static comparison reports36 inherited Ruff and7 inherited Bandit findings, zero new.
