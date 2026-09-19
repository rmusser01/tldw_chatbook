# MCP exact-input Remove and Re-allow — TASK-32866

Remove and Re-allow now retain the rule, owning profile, reviewed profile and tool
shown by their actual mounted control. Replacing the panel invalidates old controls
before pruning; a live control submits once. Re-allow also compares the definition
fingerprint captured when the tool was presented. Retry, session-list refresh and
successful action completion preserve that fingerprint until a fresh selection.
Failures can be retried, and late completion cannot replace a newer selection.

Branch `codex/mcp-rule-action-review` is stacked on PR2731 at `6de5273903` because
the fixes share the permission renderer. No policy, storage, service API or styling
changes. Existing ADR-032 and ADR-150 apply; no new ADR. Keep draft/unmerged pending
current-head CI/review and owner visual approval.

## Verification

- **56 targeted cases pass:** [37 mounted cases](tests/green-003.txt), including
  27 new cases and 10 parent Revoke regressions; [19 permission-store cases](tests/adjacent-001.txt).
  [Distinct case census](qualified-cases.json). No full-suite run.
- Original code reproduced 18 stale/duplicate/completion failures in red-001.
  Two additional failures in that attempt were harness mistakes (a nonexistent
  `snapshot()` method and absent schema). Corrected fixtures reproduced both
  real definition-drift failures in [red-003](tests/red-003.txt).
- Internal review identified fingerprint drift during cached refreshes:
  [retry/session refresh failed](tests/red-004.txt), then
  [successful Remove completion failed](tests/red-005.txt). All pass in the final
  run. [Final internal review](independent-review.txt) found no remaining blocker.
- [Seven preflight guards pass](preflight.txt). [Ruff](static-analysis.json)
  introduces no diagnostics (legacy totals remain 13/65); new test/runner are
  clean and [formatted](formatting.txt). No generated CSS changes.
- Pytest's existing cleanup warnings concern unrelated old garbage directories;
  those were not touched.

## Native verification

[Visual gallery](GALLERY.md): twelve inspected captures show Remove focused,
Re-allow focused after removal, then the cleared warning and Allow state, at
120×40 and 170×48 in dark/light. Controls have positive visible geometry, complete
painted labels, keyboard focus and center hit-test ownership. Enter drives the
real inspector/workbench/service/store path. Another profile's rule survives.

A disposable disconnected discovery fixture supplies `review-docs::search` via
the real local store. No connected-server behavior is claimed; there were zero
network attempts and no tool execution. [Native receipt](native/result.json)
and [lifecycle check](lifecycle.json) verify normal exit, absent process, released
instance lock, ten healthy private databases, zero stored conversations/messages,
unchanged defaults, nine source hashes and the runner hash.

Initial harness launches stopped before app construction: the private database
parent directory was missing, then the replacement venv's textual-image package
lacked the old private `probe_terminal` symbol. Creating the directory and using
the application's `warm_up_image_protocol` fixed the harness; production was not
changed for either issue. The old linked venv had disappeared, so this checkout's
ignored symlink now uses the main repository's existing venv (Textual 8.2.8).

[Export manifest](export-manifest.json) records original/normalized artifact
hashes. Fresh all-ref/worktree census, CLI allocation and sole-owner/history
checks confirmed TASK-32866 without correction. Remaining permission actions,
connected runtime journeys and the broader screen review remain open.
