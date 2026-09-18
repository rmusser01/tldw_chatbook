# MCP root settings implementation plan

> **For agentic workers:** Use executing-plans for this single bounded workflow.

**Goal:** Preserve root drafts and report ordered saves truthfully across MCP
refresh, navigation and shutdown.

**Architecture:** A narrow app-owned FIFO coordinator outlives UI observers.
Tools retains its local draft revision and projects the current-profile receipt.
Existing config mutation and path-validation boundaries remain authoritative.

**Tech stack:** Python3.12, Textual8, asyncio, existing atomic TOML configuration.

**Spec:** `Docs/superpowers/specs/2026-09-18-mcp-root-settings.md`.

ADR required: yes.
ADR path: `backlog/decisions/168-mcp-root-save-lifetime.md`.
Reason: The admitted configuration write and its receipt acquire app lifetime.

## Global constraints

- Explicit Save only; no automatic write retries or permission changes.
- Capture configuration path and relative-path context at admission.
- No full suite, real default-profile writes, or provider/tool execution.
- Use existing token classes and generated CSS workflow if CSS changes.

## Workflow

- [x] Add failing mounted regressions for refresh-dropped drafts, overlapping
  writes, stale canonicalization/ABA, validation/failure/retry and honest copy in
  `Tests/UI/test_mcp_root_settings.py`.
- [x] Implement `tldw_chatbook/MCP/root_settings.py`: typed bounded results,
  synchronous task registration, FIFO lock, shielded observation, profile fence,
  partial-success truth and shutdown admission/drain. Add targeted unit tests.
- [x] Wire the owner in `mcp_workbench.py` and app shutdown; project outcome before
  best-effort catalog refresh. Add cancellation/recreated-screen/shutdown tests.
- [x] Update `mcp_tools_mode.py` draft revisions, root-specific receipt and copy;
  update generated config guidance and `local_server_tools.py` docstring. Keep
  master-toggle behavior separate. Update the existing root roundtrip tests.
- [x] Run focused UI/ownership/config/governance checks; request independent
  review and fix concrete findings.
- [x] Exercise real private native dark/light at80x24 and170x48 with validation,
  retry, saved roots, newer drafts, navigation and clean shutdown; render captures.
- [x] Update QA, completion/MCP ledgers and task notes/AC; commit and push to
  existing draft PR2707 with its own visual review still required before merge.


Evidence: Docs/superpowers/qa/2026-09-18-mcp-root-settings/README.md. Independent
review added explicit dirty-state ownership, publication/file-revision fencing,
and accurate foreign-view failure copy. Native review moved the receipt above
help for compact visibility. The related Permissions resize failure is reproduced
on the saved baseline and tracked separately; no full sweep or CI-green claim.
