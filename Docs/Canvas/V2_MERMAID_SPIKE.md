# Canvas Mermaid compatibility spike — retained findings

Date: 2026-09-06

Status: Completed disposable feasibility investigation, not production qualification.

## Recommendation and scope

Use pinned Mermaid grammar with an explicit syntax subset and a Canvas-specific
renderer. Do not enable the full Mermaid browser API or relax V1 isolation.
The follow-up [design](../superpowers/specs/2026-09-06-chatbook-canvas-v2-mermaid-design.md)
and [ADR-124](../../backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md)
define the separately reviewed product scope.

The probe ran against Canvas commit `a87aab61c8e30e10c717a17c0f77de8cc9bbf67a`,
the reviewed head merged in [PR #2459](https://github.com/rmusser01/tldw_chatbook/pull/2459).
It changed no product source, dependencies, limits, conversations or databases.
All code remained disposable. Its loopback server and isolated browser session
were closed after inspection. This tracked summary preserves the observations;
it is not a committed reproducible benchmark harness.

## Inputs and provenance

- `mermaid@11.17.2`: `sha512-V6K3C8EBdEsPFZXSKMJe6ppQOENxuHARr9GvHX4hh47lAbhMRD9qf4oEK7LoaRQxULMa80/qt5gHO73aCleBBg==`.
- `@mermaid-js/tiny@11.17.2`: `sha512-PmFr1c5cSfNf5AngLLR98DrbvR4U5KeD7I+hQkGqG5w5Veu3QT4dG0qu9f99cpn/sKj2XxRBK0q7EL6hr9NY/A==`.
- Disposable bundler: `esbuild-wasm@0.25.9`; registry packages installed with
  lifecycle scripts disabled and exact lockfile inputs, only during setup.
- Exact committed QuickJS-WASM 0.32.0 bundle and virtual-facade source, running
  guest code only in QuickJS. Node 26 hosted the direct probes.
- Real-browser probe: Chromium 152.0.7977.76, unchanged production compiler,
  renderer/worker, response CSP and `sandbox="allow-scripts"`.
- Original local evidence directory: `/private/tmp/chatbook-mermaid-spike.yDDXeK`.
  It contains `REPORT.md`, probe scripts, lockfile, result JSON, generated fixtures
  and the inspected screenshots. It is temporary and not a product dependency.

## Observations

| Probe | Observed result |
| --- | --- |
| Published Tiny/full bundle size | 2,555,146 / 3,572,661 bytes versus V1's 262,144-byte script ceiling. |
| Original bundles, 32 MiB heap and 250 ms interrupt target | Guest interruption; evaluation returned after approximately 0.67–1.09 s. The direct probe did not model the production worker backstop and does not prove a 250 ms wall deadline. |
| Larger-budget diagnostic, unmodified bundles | Both fail on native `window.addEventListener` during autoload registration. |
| Tiny with exactly one autoload registration removed, diagnostic 5 s target | Loads in approximately 1.12–1.24 s with about 10.9–11.0 MB retained guest memory. Not an admitted production configuration. |
| Adjusted Tiny strict parse/render | Simple sequence parses; labeled flowchart fails in DOMPurify-dependent sanitization. Both renderers fail in native D3 `ownerDocument`/`documentElement` paths. Removing autoload is insufficient. |
| Grammar-only code plus narrow custom adapter | 77,817 bytes. Two-node top-down rectangular flow: load 15.13 ms, parse/draw 3.76 ms, 42 patches, 606,299 guest bytes. Two-party sequence: load 10.76 ms, parse/draw 1.51 ms, 82 patches, 616,101 guest bytes. Unchanged runtime limits. |
| Negative controls | Malformed flow, click/link directive and HTML-bearing sequence label refused. The click refusal was a missing adapter callback, not an acceptable production semantic-validation contract. |
| Actual browser | Both fixtures visibly rendered; status ready, scripts_disabled=false, engine=quickjs-wasm. Flow and sequence screenshots inspected. |
| Sequence reload HTTP observation through ready | Five expected loopback GETs; no observed foreign or post-execution-start HTTP requests. This benign case is not full adversarial zero-egress qualification. |
| Source round-trip | JSON restoration yielded exact source and equal compiled plans. This was not Chatbook archive export/import. |

Five final direct probe expectations and semantic-model assertions passed.
The targeted existing baseline was 84 compiler/runtime-asset tests passed,
3 reproducibility tests deselected and one existing RequestsDependencyWarning.
These are historical spike results, not fresh tests of V2 product code.

## Limitations and lessons carried forward

Only tiny linear flow and two-party messages worked in this probe. Branches,
rejoining, note placement, cycles, subgraphs, long/Unicode labels, text wrapping
and general routing were not established. Timings are single-host samples,
not worst-case bounds or portable performance promises. Eighty-two patches for
one small sequence already consume a meaningful fraction of the 500-patch
document operation budget.

Upstream generated Jison modules are internal implementation details. The probe
extracted code using comment markers; production needs a pinned, reproducible
build, exact input/output integrity, license notices and grammar/adapter tests.
Neither disabled sanitization nor native DOM access is an acceptable workaround.

The harness initially passed the model's `compatibility_issues` diagnostic as a
seventh wire field; the production renderer correctly rejected it. Correcting
the fixture to V1's exact six-field wire schema enabled the test. Other probe-only
corrections concerned QuickJS handle lifetime, whitespace normalization, template
syntax and the browser CLI invocation; they are not product defects.

Relevant upstream references: [Mermaid usage](https://mermaid.js.org/config/usage),
[security-level schema](https://mermaid.js.org/config/schema-docs/config-properties-securitylevel.html),
and [upstream releases](https://github.com/mermaid-js/mermaid/releases). Conclusions
above are based on the installed pinned sources and recorded probes, not claims
about a moving latest upstream version.
