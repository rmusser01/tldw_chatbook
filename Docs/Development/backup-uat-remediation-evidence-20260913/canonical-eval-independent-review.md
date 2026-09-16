# Evals canonical preserved-path independent review

Disposition: approved for the exact Evals correction. No actionable source findings. Full combined lifecycle acceptance awaits its separate running native test.

Frozen reviewed SHA-256 values:
- tldw_chatbook/Evals/recovery.py: aa3376537520c4082dbf59c0d3245dfbe539d7cbd1a27156be43fc4c6c2dce78
- Tests/Backup_Recovery/test_eval_rollback_retention.py: 269b3b4deedb2cc2dbfc545dde213b7b0c4cfc8ba351555199447f0d47c2bfa9
- Revised lifecycle test_created_persona_subtree_rollback.py: 679c1d533a4d3dfc70e67e41cf4c223ad49b88da1f6e4ed0803b95088b0f1e57

The six-line product change omits only the exact installed _default_config_path() when preserved_only is true. It follows target completeness, unique config dependency, authenticated rollback-source consistency, preserved selection and included/path validation. It does not return new authority for that path. The unchanged installed adapter still declares and validates its canonical file normally. Noncanonical extras retain the existing coverage, bound-root, ancestor and duplicate checks; ordinary rollback coverage semantics are unchanged.

Independent full native retention suite: 9 passed in 37.30 seconds, exit 0. Evidence: /private/tmp/uat-eval-canonical-independent.log. This includes canonical preserved/no-safety success, uncovered arbitrary-extra refusal, captured original restoration and recapture, malformed plan/terminal/association refusal, missing/symlink handling and unrelated-file exclusion. Pytest reported a teardown cleanup warning; no test failed.

The revised native Persona lifecycle retains aggregate Complete inventory and now additionally checks a fresh ordinary mounted reopen sees the original target note and neither the incoming source note nor the later-created note, then performs an actual next Complete archive capture. Those added assertions were source-reviewed; the author run is still pending, so they are not claimed as passing here.

This review is separate from the previously approved Persona mapping/snapshot correction. No product or test files edited by this reviewer.

Final acceptance update: author's exact frozen combined lifecycle completed successfully, 1 passed in 254.31 seconds (/private/tmp/uat-persona-eval-final-lifecycle.log, independently read terminal result). The final hash receipt matches the four reviewed hashes. Aggregate Complete inventory, exact native graph/original pack restoration, fresh ordinary original-note/new-note-absence readback, and next actual Complete capture are now covered by that passing native test. This is the author's native run, not a second independently executed lifecycle. The reviewer's separate 9-case Evals run remains independently passing.
