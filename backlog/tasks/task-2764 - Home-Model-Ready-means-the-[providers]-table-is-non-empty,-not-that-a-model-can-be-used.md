---
id: TASK-2764
title: 'Home ''Model: Ready'' means the [providers] table is non-empty, not that a model can be used'
status: To Do
assignee: []
created_date: '2026-08-06'
labels: [home, bug, honesty]
dependencies: []
---
## Description (the why)

`model_ready = bool(providers_models)` (`Home/active_work_adapter.py:175`),
where `providers_models` is the `[providers]` config table — which the
shipped default config populates with ~20 providers. So on a fresh install
with zero API keys, Home's header reads `Home | Ready · Local` and Details
reads `Model: Ready` / "Model ready" — while Console's own readiness
(`console_ready`, which does check real provider readiness) can be False.

Consequence: `choose_next_best_action` branch 1 (`Set up Console model` /
"Console needs a working model before live AI tasks.", route `settings`) —
the only Home path into Settings ▸ Providers & Models — can effectively
never fire, and a user believes model setup is complete until a send fails
in Console.

Found in the guide-G5 verification (code re-verified at dev @ 84e4b33f0;
header live-verified "Ready" on a key-less profile).

## Acceptance Criteria (the what)

- [ ] `model_ready` derives from actual provider readiness (shared with, or
      equivalent to, the `console_ready` computation), not table presence.
- [ ] On a key-less default config, Home shows Blocked (and the
      "Set up Console model" suggestion becomes reachable).
- [ ] A test pins the key-less-config case.
