# Chatbook Buddy UAT — 2026-10-03

## Result

Four scoped Chatbook repairs are verified on `codex/chatbook-buddy-uat`, based on dev `01a2020981c6197e5cd9945e5287567ad977edfe`.

1. **Home Resume opened the previous Console selection.** A reused Console now consumes the exact saved navigation target through its existing ordered startup worker before restoring ordinary view state. The original saved identity/history, unrelated Console draft, and independent Buddy owner/draft are preserved.
2. **Restart discarded Buddy's saved owner and Static choice.** The production config loader now copies the existing `[buddy_interaction]` table into application settings. Missing tables remain absent. The real visual controller now uses the existing strict preferences parser, so malformed `animated=0`, empty strings and empty lists retain Dynamic defaults; global reduced-motion still wins.
3. **Opening management first after restart marked the saved owner unavailable.** Management recognizes only its already-bound saved local owner through metadata without loading its transcript or constructing execution services. Presentation-only Apply preserves that binding. Missing, deleted, remote, temporary and repurposed targets remain unavailable; Persona changes still require a live target.
4. **Resume presentation and queued rebuilds could outlive shutdown.** Application shutdown cancels and drains only `console-sync` and `console-resume-navigation-startup` before runtime disposal. The retained shutdown task fences rollback sync, focus, startup and re-arming. Direct shutdown also sets Textual's existing exit flag, matching ordinary Quit and preventing queued rebuilds from mounting after child message pumps stop. Accepted execution keeps its existing runtime disposal policy.

These repair existing behavior under [ADR-139](../../backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md), [ADR-147](../../backlog/decisions/147-conversation-archive-and-exact-resume.md), and [ADR-094](../../backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md). No new schema, provider contract or visual token is introduced. Tasks: TASK-32108.1–32108.4.

## Rendered Chatbook verification

The supported Textual browser runtime served the actual Chatbook application on loopback port 18770. HOME, config, data, cache and keyring were isolated before imports. Only synthetic conversations were used; no provider request or microphone capture occurred.

- Independent Pixel Migu selection, Dynamic/Static presentation, Home/Library/Console navigation, mouse movement and grip resizing were exercised. Geometry initially saved as x118/y21, width40/height15; later browser resizing visibly kept Buddy available.
- Before repair, Home named saved B but opened the prior empty Console selection. After repair, cold and reused Home Resume selected B and painted B's original saved history.
- Buddy remained bound to saved A while Console displayed B. The exact unsent phrase `Draft A stays with Buddy UAT A.` survived modal closure, screen navigation and reopening on the final build.
- On a fresh final process, management was opened first from Home, before Buddy conversation or Console. It showed **Static**, **Conversation: Buddy UAT A · fb38008c**, and **Current Persona: None** with the live-target instruction. Apply succeeded without model setup. Artwork and geometry were retained.
- The final process's six production module hashes match static verification. Four final screenshots record management-first restart, warm Home Resume, independent Buddy draft behavior and the final visible Buddy.

![Saved owner and Static mode immediately after restart](artifacts/buddy-uat-20261003/buddy-final-cold-settings.png)

![Warm Home Resume opens B with original history](artifacts/buddy-uat-20261003/buddy-final-warm-resume.png)

![Buddy keeps A and its exact unsent draft](artifacts/buddy-uat-20261003/buddy-final-independent-draft.png)

## Automated verification

**64 unique focused checks passed on final production sources:** 63 in the final combined run, plus the corrected queued-shutdown regression separately. The combined run's one failure expected two empty mounts; Textual correctly skipped mounting entirely under the exit flag. The corrected outcome asserts no mounts, two completed automatic callbacks and no app exception. Product source did not change between these runs.

The final checks cover cold/reused Home identity/history and independent drafts, held ordered Resume during runtime shutdown, queued automatic rebuilds across actual child removal during app shutdown, real-TOML preferences/controller behavior, real-SQLite cold management, management modal/journey behavior, and existing mouse/focus capture guards.

Earlier stages passed 225 Buddy checks, 88 motion/config checks, and 33 Persona-assignment checks. They are recorded as stage-specific evidence, not claimed as one final full-suite run. The held-Resume regression first failed because shutdown returned with its worker RUNNING. The automatic-remount regression first reproduced the exact collapsed-feature `DuplicateIds`, then passed with existing Textual exit admission closed.

All nine changed test files and the management module pass scoped Ruff and formatter checks. All six changed production modules compile. Ruff comparison against base reports 487 existing diagnostics and zero introduced diagnostics after normalizing shifted line references. Whole legacy modules were not mechanically reformatted. `git diff --check` passes.

Independent review found the malformed-motion case and both shutdown gaps, verified the resulting repairs and found no remaining actionable issue. Cold management review also exercised runtime/profile changes and loaded/repurposed slots during metadata I/O; admission failed closed and worker DB connections were closed.

## Coverage limits and retained attempts

- TASK-32108 remains In Progress: native Terminal interaction, physical microphone/playback and application-configured OpenAI realtime acceptance remain open under TASK-31585. Server React Buddy success contributes no Chatbook acceptance evidence.
- Draft preservation covers navigation and modal closure. No unsent draft durability across process exit is claimed.
- Two older worker-group guards failed in the scoped lifecycle run; unchanged base source reproduces both failures (an existing ungrouped worker and a moved summarize dispatch). The other 11 checks passed, with one existing expected failure. Those unrelated checks are not claimed green.
- Older cold-resume expectations reproduced seven failures with the original base method. Their exploratory edits were removed. The full suite was not run.
- An initial direct initial-mount/recompose experiment reproduced a separate collision, but independent production-path review disproved it as the recorded shutdown fix. Its proposed widget override and test were removed. Final widget code is unchanged.
- Early harness/config-admission, temporary-directory and manually controlled mount attempts were corrected and retained in local logs. The first held-Resume harness blocked a different startup sync worker and was interrupted; the corrected regression holds only the actual ordered worker once so cancellation rollback can finish.

## Receipts

[Source, screenshot and verification hashes](artifacts/buddy-uat-20261003/verification.json) record final sources, earlier-stage attribution, targeted run counts and local log hashes. Screenshots contain synthetic fixtures. Config contents, bootstrap nonce, credentials, raw audio and raw application logs are excluded.

Raw attempts remain under `/private/tmp/chatbook-*20261003*`. Final stages: `chatbook-buddy-final-repairs`, `chatbook-queued-shutdown-final`, `chatbook-buddy-final-static`, and `chatbook-worker-guard-base-control`.
