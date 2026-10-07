# Current-dev controller approval integration

Integrated the reviewed approval capture and feedback additions into current dev's actual `InterruptRoundHost` owner. All seven conflict regions retain dev forwarding and named ownership structure. No commit, ref update, whole-repository staging, authority-policy change, guard bypass, dependency addition, or formatting rewrite was performed.

## Plan and ADR check

1. Trace dev stage2 and reviewed stage3 against the reviewed patch `0f8e97fe16..686cf8d84d`.
2. Resolve syntax with dev legs, adapt only bare reviewed fixtures to existing explicit test wiring, and establish a missing-hook RED through actual forwarded paths.
3. Port reviewed capture/receipt/settlement/finishing and review-stamp hooks into their current owners with three named live controller getters.
4. Verify exact affected files through the canonical guarded admitted private profile, compare inherited static findings, review the scoped diff, save source-bound receipts and stage only owned files.

ADR required: no new ADR.
ADR paths: `backlog/decisions/220-console-human-decision-coordination-ownership.md`; `backlog/decisions/221-console-approval-interaction-and-feedback.md`.
Reason: This is direct integration of those existing ownership and observational feedback decisions. No new authority contract, runtime boundary, schema, lifetime or policy is introduced.

## Changes

- `console_chat_controller.py`: retained module/method compatibility exports and dev patch namespaces; forwarded the optional revision; initialized the ephemeral feedback store and bridge binding; wired three explicit named live getters for feedback store, observation publisher and notifier. Existing reviewed local/virtual display metadata and observation delivery methods remain.
- `console_interrupt_rounds.py`: actual owner now captures payload view/revision, carries observation stamp metadata, binds feedback contexts/aliases, records optional receipt after Event release, reports final host arbitration, removes stale captured view/revision from finishing projections, and supplies reviewed builtin/lesson presentation plus builtin stamp observation scope.
- `Tests/Chat/console_interrupt_test_bindings.py`: explicit test-only wiring includes the same three named getters. No broad receiver was restored in production.
- `test_console_approval_feedback.py`: two pre-extraction fixtures now use actual dev host ownership; all existing assertions remain.
- `test_approval_payload_summary.py`: one bare finishing fixture now uses actual dev host registry/payload/lock aliases; all existing assertions remain.

Preserved dev's accepted-target consent guard, exact round matching and missing-ID no-op, original lock/registry authority, arm-time cancellation identity, hook deadline and cancellation final arbitration, denial reasons and unresolved denial stamps, owner scope, Event release ordering, optional reducer/observer failure isolation, FIFO/remount behavior and current actual owner lifetimes. Capture and feedback remain display-only.

## Verification

Interpreter: `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe`, Python3.12.10. Every test command used `Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py`. Each receipt verifies checkout module origins, installs original-home protection before HOME redirection, and reports `[true, "startup_allowed"]` before pytest. Full sweep was not run. Root added only the exact finite host-wiring target to the launcher allowlist.

Command shape: `<python> Docs/superpowers/qa/2026-10-05-console-approval-ux/private_control.py <target> [selection]`, with full stdout/stderr saved to the log named below.

| Phase | Target / selection | Output |
| --- | --- | --- |
| RED | `Tests/Chat/test_console_approval_feedback.py -k "controller_releases_event_before_receipt_callback or host_final_arbitration or captured_legacy"` | 5 failed,24 deselected,1 warning; missing receipt/store/legacy metadata on actual forwarded paths |
| GREEN | `Tests/Chat/test_console_approval_feedback.py` | 29 passed,1 warning |
| GREEN | `Tests/Chat/test_console_approval_scope_journeys.py` | 9 passed,1 warning |
| GREEN | `Tests/Chat/test_console_interrupt_rounds.py` | 29 passed,1 warning |
| GREEN | `Tests/Chat/test_approval_payload_summary.py` | 5 passed,1 warning |
| GREEN | `Tests/Chat/test_console_interrupt_host_wiring.py` | 12 passed,1 warning |

84 targeted checks passed. Scope journeys use real permission owners and harmless external transport doubles. Actual host tests cover receipt before callback, stale round no-op, final accepted/cancelled/timeout arbitration, failed reducer isolation, legacy alias custody and FIFO promotion. Public payload checks cover exact current dev row contract, revision forwarding, captured original arguments and finishing changed-call guard. Existing pytest-asyncio/Pydantic warning noise remains.

All five owned Python files compile. Ruff check passes on host/test bindings/feedback/payload; full host/test-bindings/feedback format checks pass. Only changed approval ranges were formatted; inherited controller/payload formatting was left in place. Full controller lint is not clean: 61 diagnostics match unchanged dev `6feb84c1d2203bc3c6d0eecd2823f4e2409b4b4e` exactly by code,message and source line; the private JSON includes every baseline diagnostic. Controller actual line count is29516 versus29470 on dev; host actual line count is6607. Owned `git diff --check` passes.

## Limits and handoff

This establishes source integration and targeted functional evidence only. Native/browser paint, latency timing qualification and actual Windows dispatch remain explicitly unqualified. No speed claim or full DoD claim is made. Current approval tasks remain In Progress. Root owns CSS/metadata/further PR verification and independent review. Only the five listed source/test files are staged by this integration agent; the report/logs are private.

## SHA256 source and evidence receipts

| Path | SHA256 |
| --- | --- |
| `tldw_chatbook/Chat/console_chat_controller.py` | `8cf43e5f2b3fce3b1b36bdf970acf77e730932886773eb870159c3a676143d18` |
| `tldw_chatbook/Chat/console_interrupt_rounds.py` | `ce0cf036e3591577ade9b36e7d36e5cdbdb59c2b36ad76715874c6c77bd7abe8` |
| `Tests/Chat/console_interrupt_test_bindings.py` | `1ffea891f04c99649bb15d7ec475c36ac8aeb9c63fe3ae0e4915625e91783931` |
| `Tests/Chat/test_console_approval_feedback.py` | `b704db1499244d8fed81b8db63f7295f5db5083178e831113180a95b29cd53a4` |
| `Tests/Chat/test_approval_payload_summary.py` | `c98a5cd11213066690eec5aa9c9c5fdbf22aa5faf5e1f85a65586024c3408329` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-feedback-green.log` | `01cb26aa4b6b80add3dcd7c45016d156229624f12aa8354e255861d6f776af10` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-interrupt-green.log` | `f2a504ddf8ebf9a57e339aba4cdbeba6c1dc97fe0c412cb7df7741f3b71d9c0b` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-lint-comparison.json` | `8c6f022197793f255ee3f9af016918d4d4b7c8c3af9612d8fd4262c28bbc6ecf` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-payload-green.log` | `0ab9a82774d2e31b8f862050eb6f6de0f7f2ce00b5e8eb499441ff42413557d6` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-red.log` | `da5ef81218d3ba0d4c6a91cf09808a2e2492d5990b273277b6fff4dc7ef77c8c` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-scope-green.log` | `989374202ec050ec7e6076e06cb6302e28ef7dbc94de33a18c5561e52d6fbd59` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-static-green.log` | `0ef6ec23267b84ed51934e9a9e5a33edcc7501f41d46e7907569515c31167449` |
| `.superpowers/sdd/2026-10-05-console-approval-ux-and-responsiveness/pr-controller-wiring-green.log` | `d1fe2bb3d8bb80579082b8b0c94ad8eae8629c11dd82d7430fe50ae5c42fa6a0` |
