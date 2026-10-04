# Chatbook Buddy UAT — 2026-10-03

## Result

Four scoped Chatbook repairs are verified on `codex/chatbook-buddy-uat`, based on dev `01a2020981c6197e5cd9945e5287567ad977edfe`.

1. **Home Resume opened the previous Console selection.** A reused Console now consumes the exact saved navigation target through its existing ordered startup worker before restoring ordinary view state. The original saved identity/history, unrelated Console draft, and independent Buddy owner/draft are preserved.
2. **Restart discarded Buddy's saved owner and Static choice.** The production config loader now copies the existing `[buddy_interaction]` table into application settings. Missing tables remain absent. The real visual controller now uses the existing strict preferences parser, so malformed `animated=0`, empty strings and empty lists retain Dynamic defaults; global reduced-motion still wins.
3. **Opening management first after restart marked the saved owner unavailable.** Management recognizes only its already-bound saved local owner through metadata without loading its transcript or constructing execution services. Presentation-only Apply preserves that binding. Missing, deleted, remote, temporary and repurposed targets remain unavailable; Persona changes still require a live target.
4. **Resume presentation and queued rebuilds could outlive shutdown.** Application shutdown cancels and drains only `console-sync` and `console-resume-navigation-startup` before runtime disposal. The retained shutdown task fences rollback sync, focus, startup and re-arming. Direct shutdown also sets Textual's existing exit flag, matching ordinary Quit and preventing queued rebuilds from mounting after child message pumps stop. Accepted execution keeps its existing runtime disposal policy.

These repair existing behavior under [ADR-139](../../backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md), [ADR-147](../../backlog/decisions/147-conversation-archive-and-exact-resume.md), and [ADR-094](../../backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md). No new schema, provider contract or visual token is introduced. Tasks: TASK-32108.1–32108.4. Command qualification harness repair: TASK-32108.5.

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

All ten changed test files and the management module pass scoped Ruff and formatter checks. All six changed production modules compile. Ruff comparison against base reports 487 existing diagnostics and zero introduced diagnostics after normalizing shifted line references. Whole legacy modules were not mechanically reformatted. `git diff --check` passes.

Independent review found the malformed-motion case and both shutdown gaps, verified the resulting repairs and found no remaining actionable issue. Cold management review also exercised runtime/profile changes and loaded/repurposed slots during metadata I/O; admission failed closed and worker DB connections were closed.

## Workspace and command qualification

**The final targeted group passed all 89 checks in 189.68 seconds.** Fresh and schema 69→75 upgraded profiles render independent artwork without a Persona, retain exact conversation attachment, exercise complete Dynamic frame cycles and Static presentation, and clear an existing workspace Persona default to explicit None. Workspace inbox tests cover read-only opening, focus/selection retention, exact acknowledgement, newer unread receipts, moved owners and failed refreshes. Mounted command tests cover exact-owner sends, drafts, retained questions/approvals, stale owners, busy/attachment refusals, and late voice completion after closure.

The first group passed 79 checks and failed 10. Several controller and full-app cases redirected imported config participants after import. Controller harnesses now retain the admitted bootstrap profile; the three full-app cases use the existing private-profile child before imports and production config writes. The profile correction passed 45 of 47 command checks. The two draft tests supplied only a stalled `stream_chat` gateway, while the default enabled agent bridge used a separate seam; they now explicitly select native streaming through production settings. One saved-owner case then pressed a still-disabled Send button between completed restoration and the next visible projection. Waiting for the actual enabled button preserves the user action and every original assertion. The two focused sends passed, followed by all 89 passing together. No product source or admission guard changed in this phase.

Actual browser interaction also created the disposable **Buddy UAT Workspace**, applied its Buddy binding with default Persona None, and opened its read-only empty inbox. Opening and closing did not acknowledge results. This browser evidence qualifies the empty inbox; populated and running-work cases use controlled controller/receipt fixtures. The two full-app sends prove accepted native-streaming work survives closing Buddy while both Console drafts and the unrelated active selection remain intact. They use a gateway double, so no real provider or default agent execution is claimed.

![Applied workspace binding retained on Home](artifacts/buddy-uat-20261003/buddy-final-workspace-binding.png)

![Read-only workspace inbox](artifacts/buddy-uat-20261003/buddy-final-workspace-inbox.png)

A separate same-process browser check returned from the workspace binding to saved A. Its original history and exact unsent phrase `Draft A stays with Buddy UAT A.` remained visible. The temporary wide viewport was reset afterward.

![Saved A keeps its draft after the workspace detour](artifacts/buddy-uat-20261003/buddy-workspace-detour-draft.jpg)

![Buddy remains visible at the restored default viewport](artifacts/buddy-uat-20261003/buddy-visible-qualified.jpg)

These 89 checks and the 64 four-repair checks have no duplicate test identities: **153 unique passing targeted checks on the final product sources**, aggregated across recorded runs. Independent review found no actionable issue in the profile/mode/readiness repairs. The final test module passes Ruff, formatting, compile and whitespace checks. Six separate `command-*.svg` exports record headless fresh/upgrade Static, Dynamic and inbox views; their original/export and repository hashes are recorded separately.

## Coverage limits and retained attempts

- TASK-32108 remains In Progress: native Terminal interaction, physical microphone/playback and application-configured OpenAI realtime acceptance remain open under TASK-31585. Server React Buddy success contributes no Chatbook acceptance evidence.
- Draft preservation covers navigation and modal closure. No unsent draft durability across process exit is claimed.
- Two older worker-group guards failed in the scoped lifecycle run; unchanged base source reproduces both failures (an existing ungrouped worker and a moved summarize dispatch). The other 11 checks passed, with one existing expected failure. Those unrelated checks are not claimed green.
- Older cold-resume expectations reproduced seven failures with the original base method. Their exploratory edits were removed. The full suite was not run.
- An initial direct initial-mount/recompose experiment reproduced a separate collision, but independent production-path review disproved it as the recorded shutdown fix. Its proposed widget override and test were removed. Final widget code is unchanged.
- Early harness/config-admission, temporary-directory and manually controlled mount attempts were corrected and retained in local logs. The first held-Resume harness blocked a different startup sync worker and was interrupted; the corrected regression holds only the actual ordered worker once so cancellation rollback can finish.

## Receipts

[Source, screenshot and verification hashes](artifacts/buddy-uat-20261003/verification.json) record final sources, earlier-stage attribution, targeted run counts and local log hashes. Screenshots contain synthetic fixtures. Config contents, bootstrap nonce, credentials, raw audio and raw application logs are excluded.

Raw attempts remain under `/private/tmp/chatbook-*20261003*`. Final stages: `chatbook-buddy-final-repairs`, `chatbook-queued-shutdown-final`, `chatbook-buddy-final-static`, `chatbook-worker-guard-base-control`, and `chatbook-buddy-workspace-command-qualified` (all with the 20261003 suffix).
