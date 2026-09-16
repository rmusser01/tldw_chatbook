# Primary-core unavailable dependency observation

TASK-32562; base e683efb86433d3598a99e31033da38f778b59896. Product scope: tldw_chatbook/DB/recovery_core.py only. Tests: new Tests/Backup_Recovery/test_core_dependency_discovery.py and the exact corrupt-bytes expectation in existing test_core_owners.py. No commits or pushes.

The primary core's optional references can only be inferred from a validated database observation. Previously, failed validation left status included and retained every optional owner as required, even when subsequent successful owner probes found those owners unused. The correction marks that same primary owner/logical ID/path unavailable and emits only its known config dependency. Successful validation and its SQL, all other owner branches, snapshot identity checks, schema policies, scope digests, capture admission, timeouts, and retry behavior are unchanged.

This remains fail-closed: a failed observation is not Complete and cannot satisfy the reviewed capture. A later fresh stable observation can be reviewed normally. It does not ignore unavailable dependencies or automatically retry capture. The original Windows validation reason was not retained, so exact occurrence attribution remains qualified; native reproduction establishes the concrete producer defect independently.

TDD:
- RED 3 failed / 2 passed in 6.02s, /private/tmp/uat-core-dependency-red.log. A real note write between native private-copy state checks produces core_validation_unavailable but the original code falsely reports included; a real unknown schema does the same; the prior corrupt-file declaration assertion was corrected to require unavailable. Stable ordinary-note and actual required-file tests already passed.
- GREEN 22 passed in 11.17s, /private/tmp/uat-core-dependency-green.log. Four new native cases plus the corrected declaration, existing missing referenced file, existing orphaned-domain validation, and all 15 existing schema/version/table/trigger corruption cases pass.
- Final new test file verification: 4 passed, /private/tmp/uat-core-dependency-final-native.log. The final static-only cleanup separates a fixed literal child script from its prefix rather than representing fixed SQL within a literal concatenation; no query or test semantics changed.

Native test behavior:
- Uses real private bound config, CharactersRAGDB, native validation, private main/WAL copies, and a real writer thread. Only scheduling at the source-state observation edge is controlled; validator and SQL results are never mocked.
- Records the actual first validation issue and verifies subsequent successful probes cannot upgrade the initial unavailable item or invent optional edges. The same unused Notes/Persona targets remain unused.
- A fresh successful discovery after the ordinary note write equals the original StorageItem exactly. Stable ordinary writes do not change dependencies.
- An actual file-backed note still adds its exact file-notes dependency; missing owner/path remains blocking. Unsupported native schema stays unavailable; existing domain and schema refusals remain intact.

Static checks:
- Ruff baseline/current 46/46, same code/message multiset; zero new. New test Ruff check/format clean.
- Bandit baseline/current 78/77: only one fewer pre-existing test assertion (two guessed-edge assertions became one exact-config-dependency assertion). Existing B404/B603/B608 counts unchanged; zero new findings and new test clean.
- Reports /private/tmp/uat-core-dependency-{ruff,bandit}-{baseline,current}.json; git diff --check clean.
- Frozen exact hashes and static summary: /private/tmp/uat-core-dependency-hashes.json.

Original read-only investigation and preserved source metadata: /private/tmp/uat-windows-library-dependency-review.md and /private/tmp/uat-core-optional-dependency-native-metadata/test_native_commit_during_firs0/home/dependency-review.json.log. Actual Windows fixture remains untouched. Independent review requested via parent before integration.

Independent parent review: approved exact frozen3files. Product only changes primary-core validation-failure status/dependency declaration; exact IDs/path and successfulquerybody preserved. Five independent native/declaration cases PASS6.66s, including actual concurrentwrite and unavailable source/domain behavior. Fresh installed mountedLibrary backupPASS47.04s; source/wheel receipt verification follows. This corrects producer classification, not a claim that originalWindowsoccurrence or startup delays are resolved.
