# Closed-chat Send custody regression

The shared manual Send path formerly interpreted a missing explicit destination
as uncertain draft custody, masking the established informative closed-chat
response. The bounded fix reuses that response only after the existing drain
fails and the explicit target is absent. Live custody, origin, hook, provider,
recovery and sibling isolation gates remain unchanged. Existing ADR219 applies;
no new ADR or public interface is required.

Real-store RED: three closed-target failures and one review-required custody
pass. A call-through withdrawal with an actual close pins the awaited boundary.
The earlier parameter named `live` was automatically skipped by collection;
its renamed review-required control actually executes and is not credited from
the skipped attempt. GREEN: 57 checks including all 52 unchanged size rows.
The complete controller/per-session/new-control selection passes 385 tests in
195.61s, with RuntimeWarning and PytestUnraisableExceptionWarning promoted.
No pytest warnings appear in the parent or its actual private-profile child.

All eleven artifact guards pass. New regression lint/format passes; changed
inherited Ruff code/message multisets add no finding. The controller shrinks
from 29270 to 29266 lines and its ceiling is lowered accordingly. Only obsolete
dispatch explanation is shortened; no helper, dependency or policy is added.
Self-review traced every shared submission caller and verified zero provider
resolution, unchanged live refusal, and untouched sibling transcript/run state.

Full raw diagnostics remain LOCAL under
`/private/tmp/closed-send-qualification-UH3Eup`; no environment dump is published.
Two unimported generated helper files were removed after the full affected green
gate, with recoverable copies retained under that root. No other checkout/chat
was altered. Broader qualification, review and integration remain separate;
this receipt is not a parent-task Done, native/Windows or participant claim.

The subsequent broader SQLite/physical-app/Interrupt/runtime selection passes
343 tests in 273.62s with the same strict coroutine warning policy and no pytest
warnings in the parent or actual private children. This includes original
closed-cursor assertions, mounted Interrupt executor teardown and three repeated
real-app sessions whose owned handles/FDs are checked after physical executor
join and before test cleanup. It does not certify every activation workflow,
native keyboard/ordinary quit, Windows Terminal or participant qualification.

| Local receipt | SHA256 |
| --- | --- |
| `red.log` (includes collection skip) | `dfeb749ca00e377b4732e6a8ddd30bf0c6ed1d10b4c4808ff3ad143f4e02be2b` |
| `red-verified.log` | `ce5fd3479174d25ae474298275a7994ee9539503c2ad22954d1f697d6ec2c9ea` |
| `green.log` | `92b772b9127d3e0b27b1c17aaba03d18778f421a07c5acd93eb28387382da0be` |
| `complete.log` | `62be0b711455aee0e4962c3181d30cd168aee242dc52adf083053e9a5b7c7a55` |
| `preflight.log` | `3cf56fe09c2712662654b8776405e6f79404f6172cfc6bd97232b5ee14f7ab0d` |
| `native.log` | `93cadee4e004e696bd0157306a11a305a87f20b8ed09beedff4efc13b4a36bbe` |

The broader follow-up subsequently exposed a real active Files callback at app
retirement; the earlier343-pass claim is not clearance for that route. Existing
physical-worker settlement and exact fixture ownership now pass the actual21
app/Interrupt and142 Workspace/resource gates. See [current audited outcomes](../task-34661/broader-qualification-2026-10-08.md).
Closed-Send production remains byte-identical to its committed385-test checkpoint.
