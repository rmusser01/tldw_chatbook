# Migu live UAT after both merges — 2026-09-05

Chatbook PR2428 merged as `66a1cbf8fdabf2f54e3222e64895bea472d42d76`; server PR2902 merged as `84b6928dcfc48fdd7b424939a9ba52a82c37612c`. These fresh runs imported the merged Chatbook revision. The only source worktree change during launch/exit was TASK31585 documentation; each receipt records identical controller hashes and source revision at exit.

| Scenario | Result | Evidence |
| --- | --- | --- |
| DeepSeek response | Expected reply; completed in1.55s, Migu idle → thinking → speaking → idle | [Provider](provider-20260905-0d3335b27f/provider.json) |
| Stop during actual response | One response character observed; terminal stopped, visible Response stopped, Buddy idle | Same provider receipt |
| Reply after Stop | New assistant identity, same session, expected recovery reply completed, Buddy idle | Same provider receipt |
| Kokoro output | Real audio sink drained128000 PCM bytes in4.71s; Migu idle → speaking → idle | [Kokoro](kokoro-20260905-57a73fbbf4/kokoro.json) |
| Voice context switch | Active playback acknowledged stopped; presentation cleared; separate synthetic voice lease survived and released to idle | Same Kokoro receipt |

Provider wrapper87330/child87336 and Kokoro wrapper87001/child87013 exited0 with no app exception. Normal config byte hashes were unchanged. No microphone was opened. Credentials remained in child memory and were not written to these receipts or the scratch config. Each run used its own disposable profile and null keyring backend. These are mounted Textual `run_test` interactions with real network/audio output, not physical Terminal or human microphone acceptance.

OpenAI realtime remains untested: neither modern nor environment OpenAI configuration was present at preflight. Intentional human microphone participation was requested and remains pending. TASK31585 AC9 stays open.

The [initial recovery probe](provider-20260905-e36c7b3c4c/provider.json) read the prior Stop terminal state immediately after queuing the new send. It did not wait for a new assistant identity, so its false recovery result is a harness defect. The corrected probe requires that identity before accepting a terminal state and passes without application changes. Both original and corrected receipts are retained.

Harness `.txt` files are frozen, non-executable audit snapshots. Their hashes match their launch receipts. The new harness validates environment-derived output/profile paths beneath its fixed scratch root before use. These snapshots are not supported portable launchers. No logs, keys, recordings, or normal-profile databases are included.

Verification: receipt JSON, identity/hash invariants, exact result assertions and whitespace checks. No production code changed; no new automated test suite or Bandit run applies. Existing ADR037 trusted speech and ADR074 Buddy leases govern this evidence.
