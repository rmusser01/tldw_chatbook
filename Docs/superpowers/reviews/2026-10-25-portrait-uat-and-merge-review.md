# PR #3244 UAT and independent review — 2026-10-25

Requested scope: full portrait feature UAT with real Pocket TTS, fix findings,
rebase current dev, independent requesting-code-review and merge with the live
image-provider qualification gap explicit. The rebase was already up to date at
`139333920d2a87cb011e4f3aa5f8319b59c44a5c`.

The independent read-only reviewer used the requesting-code-review template with
precise requirements/base/head and no session history. Coverage included connected
source/CAS, compositing/recovery, draft/generation/region ownership, Save
reconciliation, Console/avatar lifetimes, sink/TTS and recovery boundaries; it did
not exhaustively review every test, QA artifact or portable-export implementation.
It found malformed optional-region source JSON blocking manual Prepare. The fix
(`671431bf33`) has six stored-shape controls plus mounted import/Apply/Remove, all
red before and green after. The reviewer checked the repair remains behind source,
transactional recheck and exact CAS, and found no further issue.

A repeated actual-device UAT then exposed a pre-existing empty-context native
callback race, independently confirmed by a plain-thread Markdown reproducer.
`2a1711ea60` captures the admission context for both native and owned-file started
callbacks. Both deterministic observer-task controls fail on old code and pass
with the fix. The reviewer checked validators, cancellation and retained-context
lifetime, and found no further issue. No architecture boundary or dependency changed;
existing ADR-242/243 apply. Final disposition: no remaining confirmed code finding,
subject to final hosted CI and the declared live-image qualification exception.

[Final UAT matrix and evidence](../qa/portrait-compositing/2026-10-25-uat/README.md)
records 54 exact production file matches in both rebuilt installed wheels, passing
installed/native journeys, final 208 speech checks, 54 storage/authoring checks,
24 boot/token checks, real Pocket/source-device equality/content/guarded sample,
actual hardware timing and the initial 451-case feature matrix. Counts overlap.

**Live image qualification remains blocked:** SwarmUI is the only configured image
backend and cannot provide this workflow's reference-image edits. Zero live image
requests were sent. External ChatGPT sample artwork and deterministic adapter
fixtures are explicitly separate. TASK-36555 remains In Progress with AC5 open.
No full repository suite, acoustic loopback, microphone/STT, other OS/device,
native graphics-terminal hardware or independent human acceptance is claimed.

Qodo's exhausted workspace credits and Cubic's exhausted allowance remain unavailable
hosted review coverage. The queue requires a Qodo review on current head; an explicit
per-PR exception was requested from the user before arming it. Required CI and open
threads must still be verified after publication. No merge or paid quota bypass is
implied by this review record.
