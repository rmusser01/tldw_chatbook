# Fresh DeepSeek and Kokoro UAT, 2026-09-05

The repaired early Stop boundary passed a fresh real DeepSeek run:
[provider-20260905-f666185a7f](provider-20260905-f666185a7f/provider.json).
The original status-only gate saw `streaming` with zero response characters;
Stop reached `stopped`, showed `Response stopped.`, and returned Buddy to idle.
[Post-exit SQLite inspection](provider-20260905-f666185a7f/persistence.json)
confirms the exact assistant persisted `stopped`, version 3, empty content, and
no dispatch checkpoint. Its user-stop transcript marker remains in memory only.
The [execution receipt](provider-20260905-f666185a7f/execution.json) binds wrapper
20100 and child 20108, exit 0, unchanged normal config, and identical production
diff/controller byte hashes at launch and exit. HEAD was
`64ce47a04f8b541818270c10c1a6f93b83a18080`; the production diff SHA-256 was
`72fff5947be564d1f15b7905caf812ac11051cfeea0314db47c95f5ab25af364`.
The elapsed Stop fields include polling and the harness's 0.2-second UI settle;
they are not pure cancellation latency. No microphone or additional voice run
was used. [Deterministic/static verification](early-stop-verification.json)
records the targeted results and three reproduced baseline failures.

TASK-31585. The earlier completed-provider and Kokoro runs imported the source checkout at
`64ce47a04f8b541818270c10c1a6f93b83a18080` on `codex/migu-live-voice-uat`.
Each launch receipt records the Git revision, dirty state, run ID, wrapper PID,
child PID, runtime directory, and source directory. The only dirty file at launch
was the task's UAT plan update. The same wrapper reaped its child and recorded
its return code, exit timestamp, and matching child-receipt identity.

| Check | Final result | Evidence |
| --- | --- | --- |
| DeepSeek reply | Expected synthetic reply; completed in 1.33 s; Buddy idle → thinking → speaking → idle | [Provider receipt](provider-20260905-ad64a006d7/provider.json), [execution](provider-20260905-ad64a006d7/execution.json) |
| DeepSeek Stop | Waited for an actual response character, then invoked the visible Stop action; final status `stopped`, copy `Response stopped.`, in-memory user-stop marker, Buddy idle | [Provider receipt](provider-20260905-ad64a006d7/provider.json) |
| Kokoro playback | 128,000 PCM bytes drained; Buddy idle → speaking → idle; presentation cleared; 4.68 s | [Voice receipt](kokoro-20260905-434174cc5b/kokoro.json), [execution](kokoro-20260905-434174cc5b/execution.json) |
| Voice ownership after context switch | Second real playback reached `playing`; switching Console context received an accepted `stopped` acknowledgement for that request. A separate synthetic voice lease survived, and releasing it returned Buddy to idle. Duplicate terminal transitions were rejected. | [Lifecycle events](kokoro-20260905-434174cc5b/kokoro.json) |
| Exit and config | Both final child processes exited 0, reported no app exception, matched wrapper/run/PID identity, and left the normal config file's byte fingerprint unchanged | Both execution receipts |

Provider run `provider-20260905-ad64a006d7` used wrapper PID 5410 and child PID
5415. Voice run `kokoro-20260905-434174cc5b` used wrapper PID 5682 and child PID
5687. Their runtime/profile directories were unique children of `/private/tmp`;
no two processes shared a profile. Credentials were read into child memory and
never persisted in receipts or the disposable config. The null keyring backend
prevented this UAT from writing to the normal OS keyring.

These runs used the mounted Textual application through `app.run_test`, real
DeepSeek network calls, and real local Kokoro/audio-sink playback. They do not
certify physical Terminal input, microphone capture, or OpenAI realtime. No
microphone was opened; OpenAI was not attempted without its required configured
credential. AC9 therefore remains open. The second voice owner was a synthetic
lease used to test ownership preservation, not a second physical audio session.

## Harness and earlier attempts

The [earlier launcher source](launcher-source.txt), [after-text provider harness source](provider-harness-source.txt),
and [voice harness source](kokoro-harness-source.txt) are frozen audit
artifacts for the earlier provider/Kokoro runs. Both harness byte hashes were checked against their launch
receipts. Their host-specific source, model, dependency, and profile-source
paths describe this run; these text files are not portable launch commands.
The existing Python virtual environment supplied dependencies. Kokoro's model
and additional speech packages stayed in their existing temporary locations.
The final runs used approved host network/audio access after sandbox limitations
were observed.

Earlier raw receipts remain alongside the final results:

- [Sandbox provider attempt](provider-20260905-8570ea5b48/provider.json) surfaced HTTP 502; a separate direct models read failed with `ConnectError` in the sandbox. This does not establish a DeepSeek outage.
- [Stale voice harness attempt](kokoro-20260905-a288b418b4/execution.json) exited 1 before playback: the old helper was on `ChatScreen`, while the current implementation owns it on the app. Only the disposable harness was corrected.
- [Sandbox voice attempt](kokoro-20260905-0378ba6e8f/kokoro.json) observed visual transitions but produced no sink receipt, so it is not playback evidence.
- [Status-only Stop attempt](provider-20260905-5d742494f3/provider.json) completed its reply, but Stop was requested before actual response text was established and ended `blocked`. Read-only inspection of its retained isolated SQLite database confirms a real race: the empty assistant remains `dispatch_started`, with checkpoint revision 2, after clean exit. The earlier after-text Stop check waited for provider text and verified the terminal copy and in-memory marker; the repaired status-only run is recorded at the top of this page.
- [Initial host voice attempt](kokoro-20260905-d84d4bb37e/kokoro.json) proved the first drain, but cancellation omitted a second pump result. The final harness observed the real lifecycle transition method, forwarding unchanged behavior and recording content-free request IDs, states, and acceptance results.

The earlier provider/Kokoro runs, including provider-20260905-ad64a006d7 and
kokoro-20260905-434174cc5b, preceded the production repair. Their publication
changed only receipts and documentation, so Bandit was not applicable to that
earlier evidence-only work. Their JSON, result assertions, process identity,
source revision, harness hashes, links, and whitespace checks passed under
existing ADR-037 trusted speech and ADR-074 Buddy lifecycle ownership.

The repaired early Stop run provider-20260905-f666185a7f at the top of this page
tested the changed production controller. Its [static and targeted verification](early-stop-verification.json)
records passing fatal Ruff checks and changed-test formatting, unchanged full
lint baselines, and 17 identical Bandit findings on current code and HEAD with
none added. Those repaired-run receipts follow the fix and include matching
production hashes plus independent post-exit SQLite verification.

The successful after-text Stop persisted `assistant_generation_state=stopped` and removed its dispatch checkpoint; its system stop marker existed only in memory. See [retained database observations](stop-persistence-observations.json). Four deterministic regressions cover both direct and agent execution, before and after the dispatch CAS commits, repeated Stop, and preservation of another running session. They verify terminal state through an independent SQLite connection after the transaction thread finishes. This repair follows existing [ADR-079](../../../backlog/decisions/079-console-library-conversation-authority.md).

Rebased onto dev `c14dadd77`: production and regression patches are range-diff
equivalent; both independent testing-lesson additions were retained. The same
eight durability/recovery modules passed **120 tests**, with the three documented
baseline failures deselected. The dependency warning and fixture file-descriptor
growth warning remain. Independent review found no actionable cancellation or
ownership issue. Live receipts identify their original source base; they are
not a claim that provider/audio UAT was repeated after rebase.
