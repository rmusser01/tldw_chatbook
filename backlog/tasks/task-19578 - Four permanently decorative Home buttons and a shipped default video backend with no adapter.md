---
id: TASK-19578
title: >-
  Four permanently decorative Home buttons and a shipped default video backend
  with no adapter
status: To Do
assignee: []
created_date: '2026-08-21 20:24'
labels:
  - ux
  - honesty
  - video
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 6 (UX coherence / error handling /
honesty). Both re-verified at this branch base. Filed together as the residue
of the lane's central theme — *the app presents capability it does not have* —
and deliberately ranked below TASK-19550, because unlike the fake backup
checkbox, **neither of these lies silently**.

Important calibration the lane established, which keeps this scoped: the
dead-control layer is **genuinely clean** — only **two live silent no-ops
across 1,954 controls**. The Study and related cleanups in earlier programmes
actually worked. These two are the known, honest-but-unfinished exceptions.

**A — Home's Approve / Reject / Pause / Resume can never work.**
`tldw_chatbook/Home/active_work_adapter.py:188-207`
(**note: the review cited `UI/Home/...`; there is no such directory**) —
`UnavailableHomeActiveWorkAdapter.handle_control` returns
`HomeControlResultStatus.UNAVAILABLE` unconditionally for every action.

Only two adapters are ever constructed in production (`app.py:5529`, `:6125`
fallback, `:6910`). The richer one,
`LocalNotificationHomeActiveWorkAdapter.handle_control:324-418`, special-cases
**only** OPEN_DETAILS and OPEN_IN_CONSOLE (for watchlist runs, chatbook
artifacts and ingest jobs); **APPROVE / REJECT / PAUSE / RESUME / RETRY all
fall through to the unavailable base.** `RecordingHomeActiveWorkAdapter` exists
only in `Tests/UI/`.

**This is not a silent no-op**, and that distinction should survive into the
fix: `app.py:6142` raises a warning toast — *"… is not connected to an active
run service yet. Open details or Console to inspect the work."* And
`Docs/User_Guide/home.md:74` documents it plainly: *"**Decorative today.** Each
shows the same warning toast … and changes nothing."* Good doc, unfinished app.
The defect is that four controls are rendered at all when none of them can ever
act.

**B — the shipped default video backend has no adapter.**
`Video_Generation/config.py:26` — `DEFAULT_BACKEND = "stable_diffusion_cpp"`,
applied at `:345`. But `Video_Generation/adapters/` contains only
`__init__.py`, `base.py`, `comfyui_video_adapter.py`,
`minimax_video_adapter.py`. `adapter_registry.py:29` maps
`stable_diffusion_cpp` at a module path
(`...adapters.stable_diffusion_cpp_video_adapter`) that **does not exist
anywhere in the tree**.

**The mitigation holds today, and the honest framing depends on it:**
`enabled_backends` defaults to `[]` (`config.py:346` → `:312-314`) and
`_is_enabled` (`adapter_registry.py:70-73`) returns False on an empty list, so
`resolve_backend` returns `None` before any import is attempted. So this is
**latent, not a live crash** — it becomes reachable the moment a user enables
the default backend the config names. The registry docstring already concedes
the design: *"enabling a not-yet-shipped backend fails cleanly at generation
time rather than at import time."*

Per the owner's standing ruling, the durable answer for both is to stop
advertising what does not exist — remove the controls and re-point the default
— rather than to ship hurried implementations behind them.

## Acceptance Criteria

- [ ] The Home Approve / Reject / Pause / Resume controls are either wired to a
      real run service, or **not rendered** — four buttons whose only behaviour
      is a warning toast should not occupy the surface
- [ ] If they are removed, `Docs/User_Guide/home.md:74` and its troubleshooting
      entry are updated to match
- [ ] If they are wired, each action is verified end-to-end against a real
      in-flight run, not only by unit test
- [ ] `DEFAULT_BACKEND` names a backend that actually has an adapter, **or**
      `stable_diffusion_cpp_video_adapter` ships
- [ ] A test fails if any entry in `adapter_registry`'s map points at a module
      path that does not exist — this class of defect should be caught at
      import-graph level, not at generation time
- [ ] A test fails if `DEFAULT_BACKEND` is not resolvable, so the shipped
      default can never again name a backend the app cannot construct
- [ ] The mitigation is not relied upon as the fix: enabling the documented
      default backend must produce a working generator or a clear, immediate,
      accurate error
