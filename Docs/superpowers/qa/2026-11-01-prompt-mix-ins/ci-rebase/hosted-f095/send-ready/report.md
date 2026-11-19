# f095 UI3 attachment readiness fixture

Only Tests/UI/test_console_send_ready_attachment.py changed. No product code, budget, source gate, authority check or readiness oracle changed. Task38544 fresh UI3 plan read before edits. Parent handles Git/publication/manifests.

Hosted UI3 observed Workbench disabled and composer enabled at the final assertion. Exact original node passed locally (exact:1pass4.81s), so the hosted interleaving itself was not reproduced byte-for-byte. Source inspection found that the fixture ignored `_sync_console_control_bar()` returning False: checked configuration refresh is asynchronous on a cold/busy/expired proof and owns a0.2s coalesced replay. A single pilot.pause does not promise that future timer ran. The independent draft-message Workbench refresh can already show the new image blocker while the composer's sync still retains its previous cached blocker.

The fixture now waits at most4s for an actual successful checked projection, yielding through the real pilot while it refuses. It still asserts both button states identically for every staged kind and for a vision-blocked image with text. No expected boolean was relaxed. The existing image-only Send test still verifies actual provider payload image bytes.

A new real-lock control makes the relevant deferral deterministic. A separate thread holds the actual config rebuild lock. It waits1.05 real seconds for the existing one-second checked proof to expire, then stages the newly blocked image. A projection attempt returns False. The bounded wait must raise while the lock stays held, and both buttons must still show their previous enabled state; this prevents a vacuous post-release assertion. After release, the test only pumps the pilot and observes the production timer's convergence, without explicitly invoking projection. Both buttons become disabled, Send refuses, and the image stays staged. No fake maintenance/projection flags, clock patches or admission bypasses.

Evidence phases (overlapping, not aggregate counts):
- exact: original node1passed.
- control: initial investigative control failed because a fresh checked proof correctly renders without entering the held native lock. Retained as evidence; corrected by allowing the real proof to expire.
- control-expired: initial expiration control1passed.
- final-module: all5tests passed16.69s, including the original4readiness/provider oracles and the first expiration control.
- natural-replay:1passed after requiring automatic replay rather than direct post-release projection. baseline-natural-replay:1passed on ba1 production.
- final-replay-control: strengthened nonvacuous final control1passed6.43s.
- baseline-final-control: identical final test1passed7.11s on untouched ba1cafc55d production, with only this test copied into the private archive. Original archive test restored afterward. This proves the deferral contract predates extraction; it does not claim the hosted intermittent scheduling was identically replayed.

The original four oracle tests were unchanged after final-module; only the added control was strengthened and rerun. Ruff check/format and scoped diff check pass on final bytes. Independent reviewer requested the natural-only and nonvacuous proof improvements; both implemented. No remaining source finding. All mounted checks stopped before parent's final performance cohort.

Exact argv/cwd/durations/statuses are retained per-run JSON plus log/XML; final-test.py and source-inventory.json qualify current bytes. No historical test source or intermediate result is relabeled final.
