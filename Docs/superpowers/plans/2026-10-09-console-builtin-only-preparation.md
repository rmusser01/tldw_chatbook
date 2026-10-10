# Builtin-only Console preparation (OPT-88)

Date: 2026-10-09
Task: TASK-34601, AC13
ADR required: yes, amendment to existing ADR-225.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md.
Reason: explicit user approval changes errors/audits for an unused external dependency.

The initial captured maximum stays fresh. Only later stock composition may omit external catalog work when the exact frozen tool-ID set cannot contain an external tool. Preserve required common policy reads, builtin inventory, no-widening ceilings, current-source refusal, native retirement and invocation gates.

## Agreed API and ownership

- Shared preparation (root): pure maximum_needs_external_catalog(maximum) in MCP/hub_tool_catalog.py, conservatively true for unknown/non-exact-frozenset/invalid IDs and false for empty or exclusively nonempty builtin:tldw_chatbook:: IDs. prepare_console_tools accepts optional maximum_tool_ids: frozenset[str] | None = None. Separate builtin inventory need from external read need, reusing existing captured-source owners and qualification. Bind whether the external catalog was included into the existing issued preparation. Own these files and Tests/MCP/test_console_tool_preparation.py.
- Controller/provider lane: pass the frozen maximum to preparation; apply the same helper to the stock ordinary fallback, retaining the original input's eligibility decision before existing normalization. Refuse adoption of a builtin-only preparation by a consumer needing external tools. Preserve custom callbacks and factory signatures. Own Chat/console_chat_controller.py, Agents/mcp_tool_provider.py and Tests/Chat/test_console_shared_tool_preparation.py.
- Baseline verification: read-only contract/test review and prior platform evidence. Root is the sole integration/native-test owner. No concurrent native runs.

## Implementation and qualification

1. Review the plan and caller/adoption/fallback boundaries. Record original-source failures for skipped external reads and intentionally ignored unused errors/audits.
2. Implement only this dependency refinement, with fresh policy/inventory and conservative external fallback. No cache, generic framework or deferred audit owner.
3. After both implementations are ready, run focused preparation/controller/provider controls. Cover builtin-only and empty ceilings; unknown/mixed/external ceilings; changed excluded definitions and unused corrupt catalog; fresh permission/kill-switch failures; schema-budget fallback; custom routes and actual invocation freshness.
4. Lint changed logic and review the integrated diff. Use existing native test owners and unchanged deadlines.
5. Run observer-free full-profile A/B/B/A sequentially against frozen c223d2c91d. Preserve exact revision/file manifests, cold/warm samples, persistence/trace completion and heartbeat evidence. Report work removal separately from timing, including regressions.
6. Update the optimization ledger and task evidence, then publish the reviewed candidate and targeted platform qualification. Do not mark latency acceptance or the overall task Done until its targets actually pass.

## Implementation outcome

Implemented after original-source RED and two boundary reviews. The controller/provider lane handed final test additions to root before execution; all product/test files were frozen for each integrated run. The defining-module helper anchor closes the reviewed first-import substitution case. Shared result completeness prevents adoption by an external-capable consumer.

Targeted qualification covers255 distinct cases; corrected new-test assumptions are documented in the optimization ledger. Quiet sequential full-profile ABBA measured warm3.742618→3.398968s (~9.18%) with all12 saved turns settled and stable source. No performance budget, durability policy or invocation gate changed. The approved ADR-225 refinement is retained; overall latency acceptance and remote candidate qualification remain open.


### CI contract reconciliation

Exact e3ee CI passes195 preparation/provider cases on each of Windows, Linux and macOS (585 executions). The separate startup-cohort selection exposed two old plugin-route assertions requiring an external catalog read with an empty MCP maximum. Both cases had already verified provider construction and the exact unavailable-plugin-authority refusal. The correction requires zero external reads while retaining those assertions, consistent with the approved empty-ceiling refinement. Targeted native Windows receipt empty-plugin-contracts-1 passes13 controls in34.13s, including both cases, unset/nonempty/custom/factory fallbacks, and fresh deny/kill-switch policy. Tested source and HEAD are stable; lint/format pass. Broader lifetime, startup and helper-count failures remain recorded in the optimization ledger; this is not an all-CI pass.
