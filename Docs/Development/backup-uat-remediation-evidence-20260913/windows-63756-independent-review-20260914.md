# Exact 63756 Windows independent review

Read-only review of revision `63756d61d621be37f4b5a94df9640fd565575333`. No tests, app boots, remote operations, or repository edits.

## Verified outcomes and provenance

All 21 indexed artifact files match SHA256, with safe relative-path/non-symlink checks. Native JUnit: **59 passed**, 1.707s suite. Product JUnit: **1 passed, 2 failed**, 425.852s suite; all three selected cases have terminal outcomes, with no skips or errors.

| Product case | Outcome | JUnit seconds |
|---|---|---:|
| Actual F9/create/restore/open `[plain]` | Passed | 335.202 |
| Native guidance/config-save coherence | Subprocess timeout45 | 45.027 |
| Native display pairs/later fresh send guard | Subprocess timeout45 | 45.025 |

Source receipt reports clean exact Git revision, private tracked-head execution copy, and 16,673 source files. Independently checked seven relevant driver/test/helper source digests against exact Git, allowing only Git LF→CRLF conversion. Installed receipt contains 2,867 files; all 2,475 package Python files match source and exact Git (2,443 CRLF, 32 byte-exact), zero mismatches. Wheel SHA256 `e052272b565f91cf3387da7bd0271e90d6094cf622fb55feb0a6021fd85eadc0`. These are independently verified retained receipts, not new inspection of a remote live installation.

The actual installed plaintext test retains its assertions for source preservation, successful restored profile opening and local reads, original/resumed native note reads, and no blocked network attempts. It passed. Relative to verified f311: the original 59 native cases and plaintext case still pass (f311 plaintext337.758s); this selection additionally runs the two newly added guidance cases. The guidance cases invoke `_run(... installed_package=None)` against the source copy, not the installed-wheel fixture.

## Guidance failures: exact supported conclusion

Both tracebacks terminate at Windows `subprocess._communicate` raising `TimeoutExpired(..., orig_timeout=45)`. The helper has `capture_output=True`, its existing45s timeout, and no restart. There is no retained child assertion/error indicating a wrong output, count, credential convergence, or stale send guard. The first guidance fixture's retained faulthandler file is empty; no corresponding useful display-pair stack/checkpoint is present. The script constructs TldwCli, enters `app.run_test`, executes the test-specific checks, and tears down before its only final success marker. Therefore this evidence **does not locate the timeout before or after those assertions**, and cannot establish either behavioral failure or completion of the intended guidance proof. The parent Popen repr's returncode after timeout/kill is not an independently observed child assertion result.

The two native Windows guidance acceptance gaps remain open. Startup cost or teardown may explain them, but neither is proved by these child artifacts. Preserve45s and add/use a finite stage checkpoint if further localization is required; no blind timeout correction is justified here.

## Separate startup job

31 non-UI tests pass in12.58s. All four original UI cases hit their existing60s pytest timeout; both primary and diagnostic steps exit1. The failure-only diagnostic is not a passing acceptance result despite workflow continue-on-error behavior.

The same two keyboard cases pass against pinned dev `4631b60f8dd9623fc55bf16f4a37e29fcb1240c7`: cpp49.87s session/41.25s call, file41.72s session/38.13s call (each includes two warnings). Both candidate comparisons time out. Extracted23 bounded records: dev6/candidate6/dev5/candidate6, all source_matches=true, cProfile disabled. This is instrumented monitoring evidence, not ordinary startup timing.

At the final candidate samples60.047/60.079s, both test cases remain at `test_llm_gguf_source_modes.py:1367`, inside `_mount_models:114` (`await app.push_screen(llm_screen)`). Neither records test_settle or test_focus; no keyboard step has been reached. Candidate config_operation counts are551/490. Dev completes mount,10settles,focus and close. The final sampled stacks include Console rail/inspector readiness and guarded configuration reads. Both actual terminal diagnostic thread stacks instead show `_deferred_wire_collections_capture_services`→`get_library_collections_db_path`→`get_user_data_dir`→raw/config native checks; cpp is in a default-data-root/private-file path check, file in storage admission/registry/native open. These are different samples of ongoing work, not additive timing or proof that either sampled owner dominates the delay.

Verified f311 likewise had31 non-UI passes/four primary timeouts and candidate comparisons still at mount:114. Thus63756 has not resolved that acceptance failure, and these records do not justify attributing a speedup or slowdown to display-pair reuse. Root's separate tray-loop reproduction must supply its own causal proof.

Compact machine-readable receipt: `/private/tmp/uat-63756-windows-independent-summary.json`; full verifier outputs remain under `/private/tmp/uat-windows-63756-restore/`. Original local logs and artifact index hashes are recorded in the summary. No broad suite or whole-PR success claim.
