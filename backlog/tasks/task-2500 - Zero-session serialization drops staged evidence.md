---
id: TASK-2500
title: Zero-session serialization drops staged evidence
status: To Do
assignee: []
created_date: '2026-08-04 21:52'
labels:
  - console
  - rag
  - reliability
dependencies: []
priority: low
---

## Description

`ChatScreen._serialize_native_console_state` (`tldw_chatbook/UI/Screens/chat_screen.py`, ~line 16005) returns `None` as its very first branch whenever the native Console store has no sessions (`if store is None or not store.sessions(): return None`). Its caller, `save_state()` (~line 16426), only writes `state["native_console_state"]` when this method returns non-`None`, so a `None` return means nothing about the native Console survives that navigation-away at all.

Before PR-T1, that early return only cost sessions and messages (state the app can plausibly reconstruct or accept losing on a genuinely empty store). PR-T1 Task 3 (`b3114dd88`, task-2372) moved the persisted staged-evidence launch context and the "evidence sent" notice into this same serialized dict (read at ~line 16028-16029, *after* the early return) so that they would survive navigation. Because those two reads happen after the zero-sessions check, the early return now discards them too — the single point of loss for the staged-evidence-survives-navigation guarantee PR-T1 shipped.

task-2372's Implementation Notes already flagged this window during review and left it unfixed: "the 'zero active sessions' path, while non-reachable via this PR's own state, is a pre-existing latent risk (a persistent bootstrap failure would drop ALL native console state)." The PR-T1 whole-branch review re-confirmed the same finding (recorded there as minor M5) and triaged it to be filed rather than fixed in that review's fix wave. It was not filed alongside task-2370 through task-2374/2375/2376/2377, so this task records it.

Non-reachable by the mechanism actually verified in this PR: the compose-time claim of the launch context runs before the async session-bootstrap tick that would otherwise leave the store briefly empty, so a plain navigation round trip never lands `save_state()` in this window. The gap only opens if some other path calls `save_state()` while the store is genuinely still empty (e.g. a bootstrap failure that never creates a session, or a future caller that serializes before session creation).

## Acceptance Criteria

- [ ] `_serialize_native_console_state` no longer discards a persisted staged-evidence launch context or the "evidence sent" notice purely because the session store is momentarily empty
- [ ] A regression test drives the zero-sessions path directly (not only the navigation round trip PR-T1 already covers) and asserts staged evidence and the sent notice are preserved, or are honestly reported as absent rather than silently dropped
- [ ] The existing PR-T1 navigation round-trip behavior (staged evidence and sent notice surviving a normal Console <-> Library round trip) is unchanged
