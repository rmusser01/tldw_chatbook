# Handoff file observation independent review

Final disposition: APPROVED after the failure-path correction described below. The original finding and RED evidence are retained as history; it is resolved. Base HEAD86ad9993dc. Exact reviewed hashes: /private/tmp/uat-handoff-file-independent-hashes.json. No repository files edited, network/platform emulation or extra agents used.

## Finding

[P2] test_home_citation_retirement.py _run only observes/cleans the fresh child after subprocess.run returns0. If the parent reaches its existing deadline after a child has started and its atomic PID/create_time receipt exists, subprocess.run kills only the parent and raises before wait_for_restart. The identified descendant stays alive.

Native reproduction: /private/tmp/test_handoff_file_review_probes.py::test_parent_timeout_retires_already_identified_descendant starts an actual child, writes its exact atomic identity receipt, then lets the parent hit a1s timeout. The original TimeoutExpired is preserved but the child is still running:1 failed in2.15s, /private/tmp/uat-handoff-parent-timeout.log. Probe finally positively identifies and cleans its child. This demonstrates the harness edge, not that the earlier Windows run followed it; the real Windows restart normally calls os._exit immediately after Popen.

Smallest correction: on original parent timeout/failure, retire only an already-present, valid, identity-matching receipt through the existing bounded observer. Do not wait for a missing receipt, reset the deadline, kill a mismatched PID, or replace the original parent exception with observer/cleanup errors. Cover this real descendant case; analogous nonzero parent exit should not strand an identified child either. No new process coordinator or product change is needed.

## Confirmed behavior

The file-output route is opt-in; ordinary _run retains capture_output/text/result assertions and does not create restart-output.log. Fresh driver publishes identity before expensive product imports using atomic replacement. Installed-origin, clean-source/environment, original-target-byte, UI state, closure and final success-marker assertions remain unchanged. The70s call argument is unchanged; the observer receives an absolute deadline created before subprocess.run, so parent elapsed time consumes the same allowance. Existing identity checks protect against PID reuse. Existing cleanup wait can take up to5s solely to reap a killed process; it is not extra successful-product runtime.

Independent existing observer suite:7 passed in3.23s (/private/tmp/uat-handoff-observer-independent.log), --noconftest. Three additional real probes: ordinary success, ordinary nonzero stderr preservation, and delayed-parent time consuming the restart deadline all pass in3.60s (/private/tmp/uat-handoff-file-probes.log). The fourth failure-path probe above is the remaining finding. Parent installed workflow is separate and was not duplicated.

File logs are collector-compatible and preserve child output beyond original-parent exit. The fresh success marker is still required when POSIX cannot recover an unrelated process exit code; actual POSIX exec's exit code also remains the initial subprocess result. Native Windows non-child exit status still needs platform acceptance; no emulation was used here.

## Follow-up correction and independent verification

The parent run and successful-parent child observation now share one exception boundary. An existing receipt is checked with an immediate deadline on error; missing receipts are not awaited, identity mismatches are not signalled, and the original exception is re-raised. Parent nonzero assertions now occur inside this boundary. The earlier leaked-descendant finding is resolved.

Final independent run:14 passed in6.17s, /private/tmp/uat-handoff-final-independent.log. This combines the existing7 observer cases with7 private native cases, including parent timeout, nonzero parent, wrong identity preserving the unrelated real child, and cleanup error preserving the exact primary object with no private error-message text in notes. Every probe cleans only its own verified actual process in finally. No new deadline, network call, platform emulation or heavy installed run.

A nonblocking consistency suggestion was sent: make primary.add_note best-effort like stop_observer already does, so an annotation failure cannot replace the primary. Current tested built-in exceptions retain identity correctly. Hashes were refreshed after the cleanup correction; later test promotion requires a final hash refresh.

## Final promoted scope

APPROVED at the refreshed four-file hashes. Independently reran final observer module:11 passed in5.25s (/private/tmp/uat-handoff-promoted-independent.log), --noconftest. Original4 observer cases plus4 real descendant outcomes and3 parent failure modes retain the necessary coverage. AST equality verifies the promoted parent-failure function matches the private probe exactly apart from its required local _run import. Parent-timeout integration uses the same real spawn/atomic identity, stalls the parent, requires the original TimeoutExpired and verifies descendant retirement. The initial missing-import promotion failures are author-reported fixture errors corrected before this final run; they do not count as native successes.

Runner adds only the existing observer module to support-diagnostic; full selection already contains it. Fresh process identity publication and original70s installed handoff allowance remain unchanged. Helper's last change is explanatory BLE001 annotation only. The optional custom add_note override hardening suggestion was considered and not adopted: no such custom exception occurs in the demonstrated subprocess/assertion/file/observer paths, and the existing tested primary exceptions retain identity. This nonblocking suggestion does not withhold approval or require speculative additional layers.

No remaining actionable finding in the bounded scope. Windows platform acceptance remains distinct from these real local process lifecycle regressions.
