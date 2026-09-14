# Console native-pause retry — independent review

APPROVED for the bounded source and tests below. No remaining actionable findings. No reviewer source edits.

The deferral requires exact RecoveryRequired(storage_locally_paused) before the outer config operation has entered. A body exception with that same reason still propagates, as do selector, native-config, UI and cleanup errors. The earlier error-preservation behavior and independent database ownership remain intact. No raw/admission predicate or authority changes, and no config operation spans an await.

A single coalescing flag owns the screen timer. Repeated refusals schedule one delayed 0.2-second retry rather than immediate refresh recursion. Fresh retries discard the prior rail snapshot. When an async Console pass defers, explicit False stops its subsequent native reads; a separate flag assigns the whole-pass replay to the same timer. Finally, local resume, and external sync requests respect that ownership. A running sync postpones replay; changed requests are recomputed by the eventual fresh pass. Teardown refuses new scheduling, and the actual Textual screen timer is canceled on unmount. This is a bounded concurrent timer, not a claim that persistent native refusal will eventually become successful.

Independent execution: 8 passed, 13 deselected in 9.99 seconds, /private/tmp/uat-console-pause-independent.log. Selection covers body-pause error identity, real screen-owned timer execution/unmount, and native whole-worker replay with actual pause, in-progress work, teardown and external changed settings. The author's complete 21-case run passed in 24.71 seconds, /private/tmp/uat-console-native-pause-final2.log. Existing error/config/selector cases remain present. The author reports zero new Ruff/Bandit findings against the same baseline.

These are native admission/config tests with narrow UI collaborators plus a real Textual timer; final full installed/Windows lifecycle acceptance remains separate. No product timeout or test deadline changed.

Verified frozen SHA-256:

- tldw_chatbook/UI/Screens/chat_screen.py: b831c80ff48b278b5e1e63eb7a8d2a9658308df54935e73dd48c1389eb305179
- Tests/Backup_Recovery/test_console_config_sync_lifetime.py: 43c2a011ba6e76d14e608b4d9dcf73b02d7c85139e41a0f1f18ccbadbd464e7f
