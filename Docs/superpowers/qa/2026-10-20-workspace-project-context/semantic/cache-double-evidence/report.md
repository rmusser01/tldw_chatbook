# Library cache HTTP double correction (test only)

Base aa42cc0b51206adc0fbc7c8a4dd59317a2a97265. Root Task36423 plan19 and corrected-runtime ruling preceded edits. The same scoped reviewer is pending. Only the strict HTTP double now accepts allow_redirects and asserts True, the actual Library default. Existing real factory, endpoint vectors, restarted-cache reuse, namespace privacy, SQLite access and finally/fixture closes remain unchanged. No production, CI, ADR, documentation or runtime source change; no new architectural decision.

Exact selector for all3 attempts:

`Tests/RAG/test_embedding_cache_ownership_review.py::test_same_openai_model_at_different_endpoints_does_not_reuse_vectors`

## Actual commands and results

Each attempt used fresh repair-sdd-test-trust.ps1, an exact snapshot-cache-double.py <phase>-pre before pytest and <phase>-post after natural retirement. Trust verified both exact private paths trusted_directory/trusted_existing_directory. ROOT ALLFS WRITE HOLD covered each complete test-parent lifetime through process absence and pre/post equality; no source was edited during a run.

ADVERSE first setup attempt:

```powershell
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/repair-sdd-test-trust.ps1 -RunName cache-double-red-trust-01
C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/snapshot-cache-double.py red-pre
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-task4-hosted-fixture-targets.ps1 -RunName cache-double-red-01 -Selectors @('Tests/RAG/test_embedding_cache_ownership_review.py::test_same_openai_model_at_different_endpoints_does_not_reuse_vectors')
C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/snapshot-cache-double.py red-post
```

Session81655 naturally exited1: `1 failed, 1 warning in 4.95s`. Real EmbeddingFactory refused the core runtime's absent SentenceTransformers before HTTP. This is ADVERSE SETUP, not intended RED. Source12 pre/post equal, remaining matching Python=[]. No source edit or blind repeat; root then specifically ruled use of the existing read-only qualification venv and unchanged qualified helper without a local model run.

Intended RED and GREEN used this exact unchanged helper/node command, with RunName respectively cache-double-qualified-red-01 and cache-double-qualified-green-01, preceded by fresh trust cache-double-qualified-red-trust-01 / cache-double-qualified-green-trust-01 and source snapshots qualified-red-pre / qualified-green-pre:

```powershell
# Prior process-local PYTEST_ADDOPTS captured; finally invokes
# [Environment]::SetEnvironmentVariable('PYTEST_ADDOPTS',$priorAddopts,'Process').
$env:PYTEST_ADDOPTS = '--timeout=180 --junitxml=<fresh private parent.xml>'
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-qualified-targeted.ps1 -RunName <distinct RunName> -Selectors @('Tests/RAG/test_embedding_cache_ownership_review.py::test_same_openai_model_at_different_endpoints_does_not_reuse_vectors')
```

The actual environment.json and pre.json retain literal flags, argv, cwd, shared-site, source hashes and configured offline cache/model ID. RED's literal XML flag was --junitxml=C:\Users\GDesktop-1\.codex\worktrees\workspace-project-guidance\tldw_tui\.superpowers\sdd\2026-10-20-workspace-project-semantic-index\task2-receipts\cache-double-qualified-red-01-parent.xml; pytest shlex consumed its backslashes and created one literal worktree-root filename. After retirement the exact raw XML was moved unchanged to private receipts; original/retained paths are in post.json. No replay was made for this reporting quirk. GREEN used the same full absolute path with forward slashes and green run name; its parent XML was written directly to the intended private receipt path. Both use --timeout=180, with no cap widening/helper edit.

Intended RED: natural session64589 exit1, `1 failed, 1 warning in 9.26s`. The real OpenAI factory reached the strict replacement; TypeError reports respond() got an unexpected keyword argument 'allow_redirects', wrapped by the real embedding RuntimeError. Source12 and raw runtime/helper3 pre/post equal; no matching Python process remained before the two-line test edit.

GREEN: natural session7988 exit0, `1 passed, 1 warning in 10.11s`, call8.65s/setup1.26s. Source12 and raw runtime/helper3 pre/post equal, current equals GREEN post, no matching Python process remains. Exact original XML summaries:

- cache-double-red-01: tests=1, failures=1, errors=0, skipped=0, time=4.950
- cache-double-qualified-red-01: tests=1, failures=1, errors=0, skipped=0, time=9.264
- cache-double-qualified-green-01: tests=1, failures=0, errors=0, skipped=0, time=10.107

This bootstrap_profile test has no private_profile_test decorator. It runs in the finite pytest parent; no private child XML/log is expected, reconstructed or claimed. The GREEN test obtains first/second endpoint vectors with exactly2 fake HTTP calls; restarted wrapper hits the second endpoint's existing cache; real SQLite contains2 namespaces disclosing neither secret nor endpoint. Both HTTP replacements assert allow_redirects is True. All3 existing wrapper finally-close calls and the db.close fixture teardown complete without error, then the interpreter retires physically. This node does not explicitly inspect a retained SQLite handle for closed-state, count Session.close calls, or supervise semantic workers; no such stronger claim is made.

Actual qualified stdlib preflights record Python3.12.10, SQLite3.49.1 exact source873d4e274b4988d260ba8354a9718324a1c26187a4ab4c1cc0227c03d0f10e70/options, worktree cwd and existing SentenceTransformers spec under qualification-venv. The probe explicitly closes its SQLite handle. Existing helper/pyvenv.cfg/shared-site .pth raw copies match pre/post; no runtime/shared-environment/install/download change. Package resolution/configuration is structural current-runtime proof, not a historical per-import trace. The test constructs real OpenAI factories with an HTTP replacement, no local transformer model or real provider request. Known pytest-asyncio and Pydantic deprecations remain; GREEN has no thread/executor/unretrieved-exception warning.

## Environment restoration reporting correction

Both process-local prior values were null. Finally invoked SetEnvironmentVariable(...,null,Process), but both preserved raw environment-restored.json records report strict false. Windows/.NET returns an empty string after this null set, so null-versus-empty strict equality fails. A harmless metadata-only TLDW_TEST_ADDOPTS_RESTORE_PROBE reproduces prior=null/length0 -> after non-null/length0 with strict false; a fresh shell reports current PYTEST_ADDOPTS=null/length0. Both finite test shells exited. No pytest options remain and no persistent/shared environment was changed; exact null restoration within those now-retired shells is not claimed. An earlier progress summary incorrectly said the restoration records were true; packaging caught the mismatch before retention, notified root and preserved the false originals. No raw record was rewritten or test replayed to hide it.

## Static/source and retention

Scoped literal static records: Ruff format --check --no-cache --range35-38 returns0 with `1 file already formatted`; Ruff check the one test returns0 with `All checks passed!`; AST parse passes; explicit test-path git diff --check returns0, empty stdout/stderr. Source12 pre/post equals for all3 attempts; current source equals GREEN post. Six raw source snapshot sets plus both qualified runtime3 pre/post sets retain the exact observed bytes. Test current SHA256 c38f777f8a1924ac59b1d24b674807914698fbf3f22b096bb3616bcaef4571dd (10609 bytes). Only its callback parameter and assert True differ from aa42cc0; all production/CI/docs/ADR files are unchanged.

Separate semantic/cache-double-evidence retains only this adverse/intendedRED/GREEN family, snapshots, raw runtime/trust/retirement/environment/static/delta records and immutable precommit report. Manifest lists every byte count/SHA256; raw ZIP members and actual index/HEAD blobs are checked against originals. This same section is appended to live final-fix-wave-report; later commit verification remains outside the immutable archive. Earlier final-fix/integration-fix/model/release evidence and root Task36423/semantic README/final-reviews remain untouched/excluded. No extra node/suite/model/20k/Console/Settings replay, push/PR/rebase/merge/new review/task closure. Commit only this one test plus the new five-file evidence group, deliver aa42cc0 plus the test-only delta to the SAME pending reviewer, then HOLD.
