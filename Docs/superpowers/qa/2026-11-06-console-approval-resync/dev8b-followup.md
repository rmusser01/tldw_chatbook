# Current-dev rebase and Windows verification

The approval branch was rebased onto `8b762b52fa9c38be8c978f2b2175d9f070bc829a`, preserving published `3dc38f34` on `codex/pr3036-before-dev8b-20261107`. The 14 approval commits are retained. Incoming provider endpoint, DB ownership and Console breakpoint/readiness changes remain intact. The full current-dev Console guide is preserved; its only approval delta replaces the old button list.

The exact CI census is 1816: all 1809 incoming entries in order plus seven approval entries. All checker definitions, existing workflow targets and resource caps are preserved. Canonical CSS and diagnostic inventory reproduce without tracked changes. Both source builds refreshed their ignored CSS manifest normally; OS/font caches were not cleared.

Separate targeted phases, with overlap:

| Phase | Result |
| --- | --- |
| MCP / observation / scope / relay / feedback | 91 / 34 / 14 / 6 / 32 passed |
| Critical enrollment / queue pins / census / workflow / Console P0 | 1 / 119 / 84 / 63 / 175 passed |
| Approval UI: feedback / Details / privacy / lifecycle | 16 / 24 / 10 / 25 passed |
| Original UI/governance phase | 105 passed; one existing Windows CSS-path failure |
| CSS-path repair selection | 9 passed; includes regression and unchanged refusal controls |
| UI-ready module ratchet | 1038 at unchanged limit; six approval helpers absent |
| Eager-module ratchet | 688 fails strict `<688` on both feature and exact dev8b; no PR-added module |

The CSS pin helper used native paths for both output and its token-file exemption. A real temporary-fixture regression failed on Windows before two `as_posix()` conversions and passed afterward. The numeric-dimension regex, hard-zero floors, refusal assertions and actual baseline JSON are unchanged. Baseline SHA256 remains `0d25e64698d123481406446964025a80fd5613565124cd5ec2f8278df4373e37`.

All 85 changed Python files parse. Ruff has 181 current versus 186 baseline diagnostics, with no new diagnostic fingerprints. Existing formatter debt is unchanged. A fresh independent review of the complete `8b` to `f3cfffd9` product diff found no material findings in permission capture, lifetimes, grouped/raw decisions, stale actions, Details, feedback or phone arbitration. Root separately reviewed the verification-tool delta.

There are still zero qualified product timing samples. A genuine native Console frame was inspected, but real workspace-binding setup exceeded the diagnostic bound before any approval request. A separate guarded single original protected-directory lookup took 1.578 seconds and loaded 570 additional dependency names while returning four private paths. It did not reproduce the full-app 60-second wait. The unnecessary RAG import edge is being investigated separately; no overall prompt-speed or budget claim follows from this probe.

Original source-bound receipts, failed attempts and commands remain private under `.superpowers/pr3036-resync-20261106/`: `provider-post8b-*`, `ci-resync-8b-*`, `provider-boot8b-*`, `css-rebuild-dev8b-*`, `css-baseline-dev8b-*`, `css-path-tdd-*`, `static-dev8b-governance/` and `sensitive-helper-probe-*`. The adjacent manifest binds the checked source bytes. No full suite, model calls or external benchmark network traffic was used. The PR remains draft with auto-merge off pending physical qualification and current-head review/CI.