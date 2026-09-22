# PR2722 current-dev inspector review

PR2721 merged at `3722a857480b94b30fd4755f3f8e3002bd163ec3`; its actual tree matches the approved/tested CI candidate. [Closeout](../../2026-09-18-mcp-audit-filters/merge-closeout/README.md). This fresh branch starts there and applies saved inspector commit `4b77c4b`.

Only the two review ledgers conflicted. Both histories were retained, with the saved inspector checkpoint labeled historical. Product and tests applied without conflicts; no choice discarded current dev behavior. The single inspector visibility owner now hides/restores the server badge, explanation and actions together while any tool, permission, Audit or finding detail is present. Routing, tokens, CSS and permission/runtime boundaries are unchanged. Existing ADR-150/161 and TASK-2270 apply; no new ADR is required.

## Targeted verification

201 distinct targeted cases pass: nine guidance cases, 17 existing inspector cases, ten workbench routing/transition cases, 25 governance cases and 140 shared runner admission cases. No full suite ran.

The [initial integration run](tests/32836-integration-tests.txt) had 53 passes and nine failures. Four saved Cancel fixtures omitted the now-required captured operation, leaving Cancel correctly disabled; four real-app fixtures called a removed Workbench helper. Those fixtures now supply the operation and use the current `inspector.clear_mode_view()` owner, preserving their assertions. The [intermediate run](tests/32836-guidance-fixture-check.txt) retains the first fixture correction; the [final nine-case run](tests/32836-guidance-final.txt) passes.

The remaining dimension-ratchet failure is inherited: twelve declarations are identical to actual merged dev in two unchanged CSS modules. [Comparison](tests/32836-upstream-dimensions.json). No allowances were weakened. Initial pytest cleanup warnings concern unrelated old temporary directories and remain in the logs.

The saved native runner's seven new invalid-argument cases [failed before modernization](tests/32836-admission-red.txt); all [140 admission cases now pass](tests/32836-admission-green.txt). The runner uses the existing shared parser, checkout pinning, app image-protocol helper, private logging, network guard, module/source provenance and positive painted geometry.

All nine artifact guards pass: [eight in the initial preflight](tests/32836-preflight.txt), with the Mermaid guard initially unable to fetch a declared input under restricted networking; its [single network-enabled retry](tests/32836-mermaid-retry.txt) reproduced all six outputs. No other checks needed rerunning. [Static analysis](tests/32836-static-analysis.json) adds no diagnostics; thirteen existing inspector diagnostics remain. New files/changed ranges [format cleanly](tests/32836-formatting.json). [Independent review](independent-review.txt) has no remaining findings.

## Native scope

All [16 current captures](GALLERY.md) were rendered and inspected: Servers → Audit → Tools → Servers, dark/light, 80×24 and 170×48. Guidance hides for the selected detail and restores when returning to Servers. Hidden actions are absent from focus order. Real private JSONL metadata and the real built-in catalog drive selection; no tools execute or servers connect. Permission/finding details and background readiness refreshes are test-covered, not claimed as native connected-runtime journeys. Explicit inspector scrolling qualifies content ownership, not automatic compact navigation.

[Native receipt](native/result.json), [lifecycle](lifecycle/lifecycle.json) and [source verification](source-verification.json): four passing cells, real LinuxDriver/TTY streams, normal App.run return, exit0, process absent, lock reacquired, ten healthy private databases, zero conversations/messages, unchanged defaults/sentinels, zero network attempts/errors/faulthandler output. All fourteen source hashes plus the runner match. Launch provenance records the working-source state honestly; no claim that the earlier launch commit alone contains later QA edits.

Each export directory contains a publication manifest with raw and normalized hashes. Historical receipts also have [path-normalization provenance](publication-normalization.json); original source hashes are retained. Old parent-directory qualification is historical.

## Approval boundary

TASK-32836 remains In Progress. This PR requires its own fresh visual approval, current-head CI, accumulated Qodo review and final dev/conflict review before merge. The two documentation conflicts retained both histories; no product conflict resolution changed the visuals. The wider component review stays open; consult the current top-of-file ledger before choosing another slice, as Audit navigation has since been reviewed separately.
