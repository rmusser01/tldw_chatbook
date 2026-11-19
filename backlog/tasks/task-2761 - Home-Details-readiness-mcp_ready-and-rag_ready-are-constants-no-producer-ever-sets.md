---
id: TASK-2761
title: 'Home Details readiness: mcp_ready and rag_ready are constants no producer ever sets'
status: Done
assignee: ['@t2-home']
created_date: '2026-08-06'
updated_date: '2026-10-11'
labels: [home, bug, honesty]
dependencies: []
---

## Description (the why)

`HomeDashboardInput` declares `mcp_ready: bool = True` and
`rag_ready: bool = False` (`Home/dashboard_state.py:183-185`). A grep across
the package finds **no other occurrence of either name outside that file** —
`build_dashboard_input` never sets them, and no adapter produces them.
(Contrast `acp_ready`, which also defaults True but is genuinely set from the
ACP runtime manager in `home_screen.py` — live probes show `ACP: Blocked`.)

Consequences, all live-verified (dev @ 84e4b33f0, 2026-08-06):

- The Details section permanently shows `MCP: Ready` / "MCP ready", even with
  no MCP server configured or every server down.
- It permanently shows `RAG: Missing sources` / "RAG needs sources", no
  matter how complete the RAG setup is.
- `choose_next_best_action` branch 8 (`Search your Library` /
  "Search/RAG is ready over saved content.", `dashboard_state.py:384`) is
  gated on `rag_ready` and is therefore unreachable dead code.

## Acceptance Criteria (the what)

- [x] `mcp_ready` reflects real MCP state (or the field and its two Details
      strings are removed rather than shown as a constant).
- [x] `rag_ready` reflects real RAG state (or likewise removed), and the
      "Search your Library" suggestion becomes reachable or is deleted.
- [x] Tests cover at least one non-default value for each field reaching the
      Details strings.

## Implementation Plan

1. Premise-verify at origin/dev 9f8533686c: grep confirms zero producers for
   either field anywhere in the package; only the dataclass defaults, the
   two Details strings, and the rag_ready-gated dead branch consume them.
2. Investigate wiring a real producer (MCP: unified MCP control-plane /
   per-run console_mcp_tool_count; RAG: embeddings extras probe + index
   state). Neither offers a synchronous, always-populated, honest boolean
   for Home's compose-path Details: the Console MCP count is published only
   during Console agent runs; a local-hub inventory is constant-nonzero; a
   real RAG verdict needs index/config probes Home cannot run inline.
3. Take the AC's removal arm: delete both fields, their two Details
   segments (status summary + agent-readiness line), and the dead
   `search_library` branch; keep `acp_ready` (real producer) and
   `model_ready` (send-path readiness via TASK-31805).
4. Red-first: update the tests that pinned the constant strings to pin
   their ABSENCE, and add pure-state tests covering both polarities of the
   surviving fields plus the deleted branch's fallthrough.

## Implementation Notes

- Removal arm chosen: no honest synchronous producer exists for either
  field, so wiring one would either lie again (constant-true from the
  builtin hub catalog) or move slow probes onto Home's compose path. The
  fields, the `RAG:`/`MCP:` status-summary segments, the
  `RAG .../MCP ...` agent-readiness segments, and the unreachable
  `search_library` suggestion are deleted (with a comment at each site
  noting reintroduction requires a real producer).
- Behavior preserved: the suggestion ladder falls through exactly as
  before (the deleted branch was constant-False-gated); Details now reports
  only fields with producers (Model, ACP, Mode, Server, Active,
  Approvals).
- Tests (red-first: `test_status_summary_reports_only_fields_with_a_producer`
  failed pre-removal):
  - Tests/Home/test_dashboard_state.py: new summary test covering
    ready/blocked polarities of the surviving fields and absence of
    RAG/MCP; new `test_ladder_has_no_dead_search_library_branch` pins the
    fallthrough; the three ladder tests that incidentally passed
    `rag_ready=True` were updated (the kwarg no longer exists).
  - Tests/UI/test_home_screen.py: the readiness-summary and system-status
    tests now assert "RAG:"/"MCP:" absent (and "Agent readiness: Model
    ready, ACP blocked"); the ACP-runtime-state test constructs its input
    without the removed kwargs.
  - Tests/UI/test_product_maturity_phase6_recovery_docs.py: its Home-copy
    assertion updated for the removed segments. NOTE for the owner: that
    file is pre-existing red on dev (same fixture-setup
    RecoveryRequired("raw_source_selection_changed") class, but its own
    `_prepare_clean_environment` re-selects config, so the
    keep_bootstrap_profile enrollment does not apply) — the edit is
    consistency-only, verified by construction, not by a green run.
- Files: `tldw_chatbook/Home/dashboard_state.py`,
  `Tests/Home/test_dashboard_state.py`, `Tests/UI/test_home_screen.py`,
  `Tests/UI/test_product_maturity_phase6_recovery_docs.py`.
- Verified: `Tests/Home/` (142 passed) + test_home_screen.py +
  test_home_triage_rail.py + test_home_dashboard_seams.py +
  test_home_subscription_readiness.py (99 passed) at the fix commit.

ADR required: no
ADR path: N/A
Reason: display-honesty change confined to one module's display state; no
storage, sync, provider-boundary, or cross-module contract change (the
dataclass is ephemeral display state with no other constructors of the
removed fields).
