---
id: TASK-32682
title: Add qualified direct Streamable HTTP MCP transport
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:27'
updated_date: '2026-09-17 21:22'
labels:
  - plugins
  - implementation
  - mcp
dependencies:
  - TASK-32681
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-mcp.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Connect directly to generic MCP servers using explicit protocol profiles instead of treating the tldw_server wrapper as equivalent support.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Direct stdio and Streamable HTTP profiles qualify 2026-07-28, 2025-11-25 and 2025-03-26 behavior with correct per-request or initialize/session lifecycle and JSON/SSE handling.
- [x] #2 Detection uses side-effect-free discovery; unsupported versions, pagination failures and optional capabilities produce explicit readiness diagnostics.
- [x] #3 Cancellation and reconnect retain uncertain invocation outcomes without automatic replay, and connection readiness includes protocol and discovery success.
- [x] #4 HTTPS is required except explicitly selected loopback development origins; redirects cannot forward credentials or weaken transport, and legacy stdio profile storage remains readable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/111-mcp-remote-transport-and-client-dependency.md
Reason: Implement the accepted R4 direct transport boundary under existing MCPClient/httpx, with explicit partial supersession of ADR111 before wire code. Preserve existing permission, registry and audit owners.

1. Read task15 brief, actual TASK32682 and completed mcp-protocol-research.md primary-source pins/profile matrix. Recheck final M1 APIs and current store schema. Before code, amend ADR162 and ADR111 reciprocally for R4's narrow direct Streamable HTTP ownership decision and link it in Docs/superpowers/plans/2026-09-15-plugin-mcp.md and Docs/superpowers/specs/2026-09-15-managed-plugins-design.md. Preserve ADR111 history and unrelated OAuth/deprecated-SSE scope. No tldw_server publication prerequisite or new generic OAuth framework.
2. Establish behavioral RED through actual client profile connection with controlled loopback HTTP/JSON/SSE servers and a valid existing stdio control. Create protocol_profiles.py and streamable_http.py; use only the three exact qualified protocol profiles and exchanges from the brief/research. Unknown versions fail explicitly; discovery is side-effect-free and never probes tools/call. Record exact fixture/migration paths in the CLI plan before adding them.
3. Extend MCP/local_store.py closed discriminated profiles, MCPClient.connect_profile, local_control_service.py and Utils/optional_deps.py within the brief. Migrate old command records explicitly and qualify reopen/old stdio behavior. Preserve typed complete results and raw duplicate/framing/depth evidence from M1. Bound requests, responses, frames, pagination and pending ownership under the accepted limits; malformed or unsupported profiles remain honestly unready.
4. Implement each profile's per-request or initialize/session lifecycle, version/session headers, JSON/SSE results, notifications and unsupported server-request responses. Advertise only implemented capabilities. Qualify session expiry, catalog changes/cursor loops, disconnect, lost response and cancelled connect through controlled real transport entries. Readiness requires actual protocol/discovery success; cancellation/reconnect cannot replay an uncertain invocation or invent server completion.
5. Enforce reviewed origin and HTTPS except explicitly selected loopback development mode; no cross-origin credential forwarding, TLS weakening or fallback to legacy HTTP+SSE. Authentication challenges and unavailable methods remain explicit diagnostics; M3 later supplies supported binding resolution. Close actual requests/connection resources while preserving unresolved outcome identity and unrelated scopes.
6. Run the four exact brief test files plus controller-recorded affected typed-client neighbors using isolated profiles/null-keyring/network refusal except the task's explicit loopback servers. Verify worktree package/data-root provenance, freeze source/tests and retain actual commands/output. Run syntax/Ruff/Black with shared baseline parity and whitespace checks. Record APIs, three-profile evidence/limits and M3/M4/H6 handoffs in task-15-report.md and amended docs. Controller owns commit, independent review, four ACs, final notes and Done.

Verification handoff from H2 review F3: any child bootstrap/network/profile fixture needed by committed tests must be repository-owned and use the actual pytest temporary root. A private implementation runner may isolate the parent, but cannot be a hidden prerequisite for the ordinary exact test command. Reuse the final reviewed H2 fixture where appropriate; controlled HTTP tests must explicitly scope their own loopback allowance rather than disable network checks globally. Record extra helper paths before edits.

7. Activation must consume the final reviewed M1 API/report and R58. Its current client call_tool_result and strict raw decoder live in MCP/tool_results.py; the existing local/unified typed seam preserves one timeout/audit owner, while builtin/legacy mappings remain explicitly unqualified. Include the directly affected M1 regression files Tests/MCP/test_typed_tool_results.py, Tests/MCP/test_client_catalog_pagination.py, Tests/MCP/test_control_plane_tool_execute.py and Tests/Agents/test_mcp_tool_provider.py, alongside the exact four M2 files. These are read-only regression additions until a necessary fixture/source edit is recorded; the M2 fixture must not silently rewrite M1 expectations. Retain the unavailable optional mcp-unified exact-node evidence without installation or substitution. If M2 needs a change in tool_results.py/unified_control_plane_service.py or a new shared protocol helper beyond the brief, report the concrete seam for controller baseline/path recording before editing. No task activation is implied by this draft.

8. Activation consumes final M1 Fix1 at9de0ab3ef2/R59 and closeout74462424ac. Read m2-activation-precheck.md, m2-adr-reconciliation-draft.md and task-15-paths-draft.json. Root captures current Git BASE snapshots before the reciprocal R4 ADR/docs amendment; worker verifies that amendment before runtime edits. Four new files and the exact existing paths in the brief plus four named R4 docs are authorized; report additional shared helpers/readiness/fixture paths before editing. Preserve exact portable argv/env tokens where the new closed profile represents them, and distinguish legacy normalization. Task AC1 names both transports and three protocol profiles; actual qualification must identify and exercise each supported combination rather than infer stdio behavior from HTTP fixtures. Private parent runner refuses DNS/IPv4/IPv6 by default; committed controlled HTTP fixtures may grant only their explicitly bound loopback address/port, restoring refusal afterward. No hidden external fixture shim, blanket network exemption or real-profile access. Retain R59 audit claims and no automatic invocation replay.

9. Resolve the cross-era offer ambiguity as R60 before finalizing negotiation: an explicit modern connection that receives recognized modern error -32022 may select only a different qualified modern revision; if none exists, return an unsupported-version diagnostic without initialize, same-version retry or invocation replay. Explicit legacy initialize may counteroffer only the two qualified legacy revisions. Record the rationale/compatibility cost in ADR162, the plugin spec and MCP subplan; cover modern rejection with only legacy/unknown offers, repeated same-version offers, unknown legacy counteroffer, successful legacy counteroffer and unchanged same-profile controls on actual transports. Read m2-version-negotiation-ruling.md and retain the distinction between protocol-era qualification and optional future automatic detection.

10. Correct the disclosed blanket query ban under R61 before initial review: permit literal endpoint routing queries without parsing/reordering/dropping their parameters or assigning them credential-reference meaning. Preserve HTTPS/explicit loopback, userinfo/fragment refusal, reviewed origin, redirect refusal and fixed diagnostics. Never expand environment/placeholders or inject host credential material into a URL. Update only existing authorized profile/HTTP/store test paths and the existing ADR162/plugin-spec/MCP-plan docs; use actual endpoint request-target controls for duplicates, empty values and encoded parameters, profile save/reopen and changed-URL discovery invalidation. Qualify ordinary query and no-query success through the actual HTTP entry, plus the retained refusal controls; preserve superseded artifacts and refreeze affected source/tests/docs for final evidence.

11. Initial independent review found I1/I2 at commit5875926a9f. Repair only these findings in Fix1: HTTP resource cleanup must survive interruption of caller-task draining, retain honest cleanup ownership/retryability, and close actual HTTP resources within the existing bounded owner without inventing remote completion or replay. Reopened HTTP records must apply the same discriminated unsupported env/auth validation before any fields are normalized/dropped or a network connection is attempted; retain historical stdio behavior and valid HTTP controls. Read task-15-review.md/task-15-fix1-findings.md verbatim. Allowed repair paths are MCP/streamable_http.py, MCP/client.py, MCP/local_store.py, MCP/local_control_service.py and Tests/MCP/test_streamable_http.py, test_local_store.py, test_protocol_profiles.py, test_control_plane_lifecycle.py; use only those actually needed and report any additional path/contract change first. Add genuine same-entry regressions for in-flight disconnect with asynchronous caller cleanup, interrupted/repeated close and actual closed HTTP resources; raw schema2 reopen with unsupported env/auth fields must never become ready or dispatch, with valid HTTP and legacy stdio controls. Use the recorded eight-file covering set, private runner and exact scoped loopback fixtures; preserve original evidence, compare static against the23 captured Fix1BASE snapshots, freeze the final bytes and append the fix report. Root owns exact commit and fresh scoped rereview. Existing bootstrap noise remains final triage, not this fix.

12. Fix1 documentation may update only the existing M2 implementation-boundary sections in Docs/superpowers/plans/2026-09-15-plugin-mcp.md and Docs/superpowers/specs/2026-09-15-managed-plugins-design.md. Explain pending versus failed local cleanup, retained unready ownership, no false completion based on HTTPX closed flags, and no automatic replay/replacement. A cancelled wait can rejoin a still-running close; actual resource-close failure remains explicitly incomplete. This documents the existing truthful cleanup/uncertainty contract (ADR162/R49), not a new cleanup runtime or guarantee of remote termination. Both docs are already among the23 Fix1BASE snapshots and24 freeze paths; include them in final exact diff/static/whitespace/freeze evidence if changed.

13. After the independent Fix1 review accepts I1/I2, record the factual HTTPX closed-flag versus actual resource-cleanup incident in backlog/docs/lessons-testing-evidence.md. Keep this closeout documentation-only; the final frozen558-test result qualifies unchanged runtime/test bytes. Record reviewed fixes, existing ADR111/162/163 and R60/R61, exact verification limits, four ACs and final Implementation Notes through the Backlog CLI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented direct stdio and Streamable HTTP profiles for the three selected MCP revisions using the existing MCPClient, discovery, permission, typed-result and audit owners. HTTP handles bounded JSON/SSE exchanges, explicit version/session lifecycle, origin/TLS/redirect policy and literal endpoint queries. Profile schema2 explicitly migrates old stdio records; reopened unsupported HTTP fields now refuse before coercion or dispatch.

Fix1 gives HTTP request scopes their own completion evidence and retains one shielded cleanup task. Cancellation or the existing cleanup deadline cannot falsely release the owner: incomplete cleanup withdraws readiness, refuses new work and blocks replacement. A pending cleanup may later complete; an actual lower-close failure remains owned/unready and repeats its failure. Local closure never proves a remote invocation settled, and reconnect never replays it. Stdio teardown ordering is preserved.

Validation: final frozen eight-file covering run 558 passed in 36.94s, zero skips; all 24 source/test/doc/runner hashes and 23 exact Fix1BASE snapshots verified. Ten Python paths passed syntax; Ruff 169 baseline/current, no added diagnostics; Black no new edits and all four files introduced by M2 wholly clean; whitespace passed. Initial 534 tests overlap and are historical. Controlled actual Darwin IPv4 HTTP and stdio peers qualify the selected matrix; public HTTPS trust, external/vendor servers, IPv6, Linux/Windows and unavailable optional mcp-unified remain unqualified. Credentials, scoped plugin ownership and hook-effect acceptance retain their M3/M4/H6 owners. Bootstrap log noise remains final-review triage.

Independent review: Fresh scoped review accepted I1/I2, with no new Critical, Important or Minor findings at commit 4a4537427aa6710eed514b08e252100cd1e508a0. Initial findings I1/I2 and three scope boundaries are recorded in task-15-review-resolution.md; both Important findings are addressed with no new blocking breakage.

ADR check: existing ADR162/163 and reciprocal ADR111 partial supersession, R4/R49/R58/R59/R60/R61; no new ADR. Code/operation boundaries are documented in the MCP subplan and managed-plugin spec. Factual HTTPX cleanup incident lesson is recorded in lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
