# Console marker preparation verification — 2026-10-07

The transcript previously called the Change Review SQLite reader on the UI loop. Removing unused history columns reduced work but left that blocking call intact. The existing async transcript refresh now awaits the original narrow query in a finite worker before reading current messages.

The stock path captures its runtime, coordinator, bridge, database, conversation and publication revision. It preserves direct/custom synchronous readers, validates captured sources between the original queries, and refuses obsolete publication. Existing preparation ownership retains the exact callback through repeated cancellation and Runtime disposal. Newly acquired worker connections retire on their creator thread; borrowed connections and unrelated replacement caches remain intact. Existing ADR-126 documents this ownership boundary.

## Targeted native evidence

- Original narrow SQL regression: the native connection/lease and actual query were held briefly. Rendering remained correct, but the caller loop made no progress. A preceding fixture run used the wrong assistant anchor and is excluded.
- With finite preparation: the unchanged native hold allows loop progress and the exact acquired handle, registration and lease retire before completion.
- All 11 native controls pass: normal completion, cancellation, repeated cancellation, chat/revision changes, Runtime disposal, borrowed transactions, concurrent refresh coalescing, foreign cache replacement, database replacement and original-reader default drift.
- Six existing custom/query controls and the original real-Git byte-identical rendering test pass. The original rendering fixture now closes its creator-owned runs database and newly acquired default registry handle while preserving borrowers.
- Final affected integration: 10 passed in 42.78 seconds (52.5-second contained driver): original real-Git rendering, three original transcript repaint controls and all six original history-lifetime controls. The original global history drain passes unchanged. The three detached transcript tests declare their existing collection-bound bootstrap-profile prerequisite; their bodies are unchanged.
- Sources and HEAD remained unchanged during every accepted run. New modules pass Ruff and formatting; edited modules introduce no lint findings. The platform workflow includes all 11 native marker controls.

The new worker controls have been verified locally on Windows. The previous seven query/rendering cases passed on Windows, macOS and Linux; cross-platform verification of this new asynchronous boundary remains pending CI. These are functional ownership and responsiveness results, not acceptance of whole Send/startup latency. The broader task and draft PR remain in progress.

## Cross-platform result

At committed `c7e20dd64a`, all 18 marker native/query/rendering checks pass with no skips or errors on Windows, macOS and Linux in [CI run 37674018770](https://github.com/rmusser01/tldw_chatbook/actions/runs/37674018770). This verifies the asynchronous boundary on each actual platform. Original startup liveness still fails and Linux/macOS pending-close still reports its live CharactersRAGDB worker connection; neither broader issue is claimed fixed by the marker change.
