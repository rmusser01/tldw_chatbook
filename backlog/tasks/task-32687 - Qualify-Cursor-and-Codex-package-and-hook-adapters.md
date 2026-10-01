---
id: TASK-32687
title: Qualify Cursor and Codex package and hook adapters
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:30'
updated_date: '2026-10-01 10:13'
labels:
  - plugins
  - implementation
  - delivery
dependencies:
  - TASK-32686
documentation:
  - Docs/superpowers/plans/2026-09-15-plugin-marketplaces-and-ui.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make supported foreign packages useful without claiming runtime equivalence for unsupported semantics.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Version-pinned portable/OpenAI overlay, standalone Codex and Cursor fixtures cover deterministic interpretation, explicit-path replacement and execution-affecting catalog overlays.
- [x] #2 Vendor metadata, variables, manual-only behavior and rule/agent constraints map explicitly or remain unsupported with preserved deliberate exclusions.
- [x] #3 Hook mappings qualify timing, payload, matchers, input/output, cwd, shell-free argv and timeout together; required unknown guards block affected behavior.
- [x] #4 Each fixture records upstream revision and license with independently expected inventory; parsing evidence is separated from exercised Chatbook behavior and original-host comparison.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md
Reason: implements the accepted versioned dialect and conservative foreign-semantics qualification contract.
1. Pin official packaging/hook references and curate independent synthetic fixtures with upstream revision/license and expected inventory.
2. Establish RED for overlay precedence, explicit-path replacement and foreign guard preservation through inspection.
3. Reuse bounded capture/native validation and current authority owners. Normalize supported instruction metadata; retain unknown constraints and deliberately unsupported source hooks. Capture executable catalog inputs in immutable package material so review/recovery can re-inspect them without a new storage owner.
4. Exercise actual Console content/readiness alongside parser controls; qualify every proposed hook mapping across its full contract and refuse incomplete mappings rather than borrowing native output semantics.
5. Run targeted/static checks, record unsupported/original-host/platform limits, update docs/ACs and commit only I2-owned paths.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented pinned OpenAI/Cursor interpretation through the existing bounded native inventory and authority owners. Inline overlays replace compatibility wholesale; explicit vendor paths and empty exclusions replace defaults. Retained catalog execution fields and chosen dialect survive protected review, materialization and registry-loss recovery. Supported instruction metadata uses the actual Console user lane; unsupported constraints, variables/apps and undocumented Codex presets remain unavailable. All foreign hook runtime mappings remain deliberately unsupported qualification proposals because full payload/cwd/timing/output/timeout contracts are not qualified; unknown/required root/group/handler guard scope fences affected material. This replaces the plan prototype HookHandler return without claiming original-host compatibility. ADR-162/163 and specs/plan/authoring/evidence docs updated. Targeted evidence: 117 parser/capture/adapter cases and 127 native Console/coordinator/recovery neighbors pass without skips; authored final 25 pass plus 2 post-lint controls. New Python lint/format, changed syntax/fixture JSON/whitespace and no-new-shared-lint checks pass. No full suite, GUI, original-host, Windows/Linux or Keychain/OAuth certification. Fixture provenance records independent AGPL bytes and exact upstream revisions/license context. I3+ remains outside this requested scope.
<!-- SECTION:NOTES:END -->
