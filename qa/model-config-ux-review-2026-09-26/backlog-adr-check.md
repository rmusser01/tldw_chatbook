Checkout: `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/73fb7a69-fb3c-49ea-81ff-f711a75d6336/scratchpad/review-wt` (origin/dev @ c4225b5d38). All paths below are relative to it. I did not fetch, so the refs are from the last fetch at 2026-09-26 09:38.

**Read this first: duplicated frontmatter.** Seven task files have two `status:` keys, and a YAML last-key-wins reader will see the second one. Tasks 30010, 30011, 30012, 30013 and 30014 each read `status: Done` at :4 and `status: In Progress` at :9. Task 31552 has three status lines. Task 26042 reads Done at :4 and To Do further down. Tasks 30010–30014 are the previous Conversation settings redesign (PR #2365). All their ACs are checked, so treat them as Done. Several findings below are regressions against their checked ACs.

## 1. Open tasks that already cover findings (status is the first frontmatter key)

**Console model surfaces**
- **task-194**, To Do (:4), "console_model_popover uses the shared provider display-name catalog". The popover shows raw provider keys. Covers the provider-list and label drift findings, popover part only.
- **task-14812**, In Progress (:4), but all 7 ACs are checked, so the status is stale. "Unify Console model selection into a searchable picker". Its AC#6 (:30) says "cannot retain a model from the previous provider", so "provider switch keeps the old model" is a regression against it.
- **task-3600**, In Progress (:5), ACs all checked. Retired Anthropic models in the Console dropdown. Covers model-list authority.
- **task-18922**, To Do (:4), "Model popover one-turn model override". Adjacent to the A/B toggle and recents finding.
- **task-18931**, To Do (:4), council presets as virtual models (picker additions). **task-18921**, To Do (:4), rank slash suggestions by usage. Only a weak match for recents/favourites.
- **task-32859**, To Do (:4), "Unify the console provider selection builder". The selection algorithm exists twice (`console_chat_controller.py:593-720` and `chat_screen.py:9618+`). This is the root-cause area for "four editing surfaces" and provider-switch drift.

**Console display surfaces**
- **task-338**, To Do (:4). Streaming is not shown in the rail Model section. Covers the five display surfaces and the streaming inconsistency.
- **task-32811.7**, In Progress (:4). The rail Model section's Temperature and Max tokens rows never update after compose. Covers stale display surfaces and is adjacent to "saved defaults don't reach the open chat".
- **task-25887**, To Do (:4). The Inspector's blocked-state guidance falls below the fold at 128x40. Covers setup and readiness display.
- **task-32572**, To Do (:4). Tab does not cycle the Get started card's buttons. Covers the setup/Get started card.

**Readiness, Test Provider, credentials**
- **task-2764**, To Do (:4). Home "Model Ready" means the `[providers]` table is non-empty. Probably partly superseded by task-31805 (Done, merged in #2461), but `Home/active_work_adapter.py:175` still reads `model_ready=bool(providers_models)`, so verify before relying on either.
- **task-2523** and **task-2524**, To Do (:4 each). The readiness and spend credential lookups diverge (normalized key, `api_key_env_var`). Covers false Ready.
- **task-32806.1**, In Progress (:4). Placeholder or whitespace-padded keys pass the readiness gates. This is the placeholder part of "Test Provider says complete for fake keys".
- **task-32806.2**, In Progress (:6). Credential precedence; the ADR amendment is already recorded at ADR-012:100.
- **task-26837**, To Do (:6). The wizard's connection test succeeds but no `api_settings` block is written. Covers the wizard and Test-false-positive class.
- **task-486**, To Do (:4). Custom-named credential query params appear in the provider Test evidence. Covers the Test Provider "pipe-dump evidence" finding, redaction only.
- **task-24454**, To Do (:4), readiness is recomputed on every keystroke. **task-32804.3**, In Progress (:4), the idle 0.25 s credential poll rebuilds readiness. **task-32010**, In Progress (:4), ACs checked, a readiness exception leaves the send busy. These are performance constraints on any new readiness display.
- **task-2117**, To Do (:5). `api_base_url` is dropped for every non-llama.cpp provider. Covers the Endpoint/Base URL finding.

**Settings screen**
- **task-1378**, To Do (:4), split `settings_screen.py`. **task-31202**, To Do (:4), give it a size-ratchet row. **task-32809.2**, In Progress (:4), god-module budgets. Structural prerequisites.
- **task-1379**, To Do (:4). Re-run the Settings critique. The 14/40 score is effectively that re-run.
- **task-2831**, To Do (:4). Category focus intent is lost in recompose races. Adjacent to the focus and Tab-order findings.
- **task-25890**, To Do (:4). The boundary note escapes the impact pane under production CSS. Covers inspector geometry.
- **task-15512**, To Do (:4). Red Settings provider-default and stale-endpoint contract tests on dev. These are baseline reds for this area.
- **task-16503**, To Do (:4). Remaining `Select.BLANK` usages, including `compact_model_bar.py`.
- **task-31738**, To Do (:4, deferred). Automatic llama.cpp prompt cache. Relates to the Prompt-cache snapshots block at the top of the provider card.

**Cross-cutting**
- **task-32465**, To Do (:6). Radio groups show the selected option by colour only. It is radios only: checkbox colour-only in the catalog and discovered list is **not** covered.
- **task-32864**, To Do (:4). Modal dismissal adoption. It explicitly lists `console_settings_modal` as out of scope (:27), so "Esc gives no unsaved-draft prompt" is **not** covered.
- **task-32644**, To Do (:5). dev is over the `_ui_ready` ratchet; it names `console_model_popover` and `settings_screen` as boot-path importers (:31).
- **task-17655**, In Progress (:4). Console inline-REPL design; owner approval is required before implementation.

**Findings with no open task.** Card order and length. Model is free text. Discover goes to "Save selected". Saved defaults don't reach the open chat, and there is no "applies to" copy. Nested-frame and border contrast. 111- and 136-column inputs. Checkbox colour-only. OptionList clips the first character (fixed height 6). Settings rail focus is invisible. F6 is broken in Settings. Tab order. Placeholders shown instead of effective values. Config keys and jargon in the UI.
- "Console Defaults" is not a category: the copy is at `settings_screen.py:1975,5486,17214`, but it is absent from `settings_config_models.py:13-42`.
- "Override current Console model" is actually reasoning replay (`settings_screen.py:18036`, added by f043db12b7 / #2575, task-32273).

Also uncovered: Streaming as Select vs Checkbox. Keyless providers get credential fields written. `/model` ignores its argument. No recents or favourites. Unadvertised Ctrl+Enter. Unsupported params shown for Anthropic. Missing help text. The setup card has no return from F4. The wizard shows an auth failure as a radio option. The starfield card. Label drift between surfaces. Esc gives no draft prompt.

Checked ACs on the Done tasks that these findings contradict or that constrain them:
- 30011 AC#2 (:32): "Ready to send" means no known local blocker, never provider acceptance.
- 30011 AC#6 (:36): the rail, setup card, modal and Settings Test must not overclaim one another.
- 30012 AC#3 (:34): controls with unknown support stay visible under Advanced. This is why Anthropic shows unsupported params.
- 30012 AC#6 (:37): the scope copy requirement.
- 30014 AC#1 (:32): Test connection appears only where a non-generating probe exists.
- 30014 AC#5 (:36): the save accelerator must be discoverable.

## 2. ADRs that constrain the redesign

**Settings vs conversation scope**
- **ADR-006** (006:10, :29, :31). Settings owns persisted global and per-model defaults; Console owns effective session resolution. Precedence is per-model, then `chat_defaults`, then provider, then fallback. An unsupported control must show "unavailable" copy or be omitted.
- **ADR-095** (095:10-28, :65-66, :74-79, :189-191). Apply changes only the originating conversation. There are two explicit default actions: "Save as model default" and "Make default for new chats". Blank new chats use the saved default. **Existing and open conversations do not rebase (:26)**, so the P0 "saved defaults don't reach the open chat" is the decided behaviour: fix it with "applies to" copy or amend the ADR. Both surfaces share one Apply orchestration. A provider change clears unsupported reasoning fields. The quick popover's field mask is temperature and streaming. **Full Settings represents streaming as Inherit/On/Off (:79)**, so the Select is the decided form and the Checkbox is the drift.
- **ADR-052** (052:20, :31-35). Console Behavior owns memory and compaction defaults. The modal writes only provider, sampling and streaming defaults. This is where the "When limit nears" / "Behavior" drift lives.
- **ADR-147** (147:19-20). The six-layer parameter stack includes `[console.provider_defaults.<provider>]` and registry `params`. Collapsing the "four places" must keep these layers.

**Settings screen IA and commit models**
- **ADR-033, Settings commit models** (033-settings…:7-11). Keep three honestly labelled commit models; staged draft is the default; instant-apply must carry the "applies immediately" label. "Draft — save with s" is the ADR-mandated label for each category's save model (`settings_screen.py:9411-9419`, task-1717), not a dirty flag. It can be reworded but must stay truthful.
- **ADR-033, application state** (033-application…:13-23). No new root state object; handoffs go through revisioned single-slot handoffs. The setup card's missing return path must reuse `PendingHandoffStore`, as task-30010 did.
- **ADR-015** (015:30). `DestinationHeader` is the standard identity header.
- **ADR-011** (011:42, :46). Shared primitives never call provider services. Readiness and primary actions must be visible on screen, not only in the command palette.

**Credentials and Test Provider**
- **ADR-012** (012:10, :29-33, :75-99, :100). Settings owns credentials. Providers & Models must distinguish "saved", "from env var" and "missing". Console recovery must open the exact credential control. There is **no provider-specific secret validation** (:33), so detecting fake keys in Test Provider needs a new decision. The Anthropic subscription rule omits credential fields from writes (relevant to keyless providers). The 2026-09-19 amendment ranks a stored key above the env var.

**Provider catalog and aliases**
- **ADR-002** (002:10-12). Discovery is manual, and selected IDs must be explicitly persisted into `[providers]`. Discover feeding "Save selected" is by design. Persistence is refused when normalized keys collide.
- **ADR-020** (020:10-27, :54-67). Cloud catalogs auto-refresh; the endpoint snapshot is authoritative; the active session's model is always preserved. A consent gate applies, and recording consent through the Settings toggle was explicitly rejected. The auto-refresh checkboxes are ADR-033's instant-apply exception.
- **ADR-146** (146:14-19). Custom endpoints appear as `custom-ep:<slug>` instances in the picker. The built-in `custom` and `custom_2` slots are retained unchanged, so they cannot simply be dropped from the provider list. Sampling settings are never copied into templates.
- **ADR-066** (Proposed; 066:26, :32). The alias families `llama_cpp`/`local_llamacpp`/`local-llm` and `vllm`/`local_vllm` are live identities. Legacy aliases can be hidden, not deleted.
- **ADR-117** (117:11-17) and **ADR-114**. Settings owns durable endpoints and new-chat defaults; Console owns session adoption.
- **ADR-090** and **ADR-119**. They own reasoning replay and prompt-cache snapshots, the two misplaced provider-card sections.

**Console rail IA**
- **ADR-017** (017:10). The left rail uses text-only bordered sections, including Model.
- **ADR-083** (083:20-28). The Model section's natural height is capped at 15 rows, with an overflow hint.
- **ADR-077** (077:12-17). Inspector sections are bounded at 20 rows.
- **ADR-043**. Rails collapse below 150 and 100 columns.

**Keys**
- **ADR-031** (031:7-8, :10, :18, :20). F6 is app-global "next pane", so broken F6 is a violation. No terminal-convention keys. Footer hints must match working bindings (so a kept Ctrl+Enter must be advertised). Console modals need a safe Escape with modal-owned dirty guards (task-16211), which silently losing the draft on Esc conflicts with. The F-key destination layer comes from task-32458.

**Control height and CSS ratchets**
- **Three-row control height was not decided in an ADR.** It is a design-language token: `$ds-control-height: 3`, with `$ds-control-height-compact: 1` already available (`backlog/docs/design-language.md:73-75`, `css/core/_variables.tcss:176-177`). Related rules: "left borders cost a column, never a row", "focus is sacred", "readable beats decorative" (design-language.md:126-137). `MODAL_CONTROL_HEIGHT = 3` is a local constant (`console_settings_modal.py:168`, `console_endpoint_template_modal.py:89`), pinned by `Tests/UI/test_console_session_settings.py:4555-4560`.
- **ADR-150/161** (161:11-28). New geometry must come from tokens. Raw dimension literals in `.tcss` sheets are banned outright (`Tests/UI/test_component_pattern_governance.py:266-289`). Ad-hoc Python `styles.*` writes have a zero floor (:291). Hex colours are ratcheted (`Tests/UI/test_design_token_governance.py:170`). A universal modal base is rejected.
- **ADR-097** (097:43-61). Ratchets never rise. Current values: boot CSS bytes 608,090 (`Tests/Performance/test_boot_css_byte_budget.py:117`) and `_ui_ready` 1,031 (`Tests/Performance/test_ui_ready_module_census.py:150`). The popover is on the boot path. The ADR's ledger stops at 1022; later raises are recorded only in comments at census :130-149.
- **Module size ratchet.** `console_settings_modal.py` is pinned at 7,807 lines, exactly its current size, so it has zero headroom (`Tests/Architecture/test_module_size_ratchet.py:68`). `settings_screen.py` (32,167 lines) is deliberately unratcheted (:21) pending tasks 1378 and 31202.

## 3. Recent merged work on these surfaces (first-parent, since 2026-08-15)

There are 93 merges in total; the ones below touch model configuration.

**Settings ▸ Providers & Models**
- 0053a0a40f #2831 (09-25): task-32926, keyring reads moved off the UI thread (`settings_screen` +301).
- 11b3202b6c #2461 (09-06): task-31805, readiness-honesty fix for Home and Settings Overview.
- dcf53e13f6 #2623, 8aa867b525 #2617 (09-11): custom endpoint registry, tasks 32307/32308 (ADR-146).
- 96a8f37aa7 #2646 (09-12): custom-endpoint UX round 2, no task id (see the commit messages).
- ec55a0ed3b #2419 (09-05): task-31552, prompt-cache snapshots added to Settings (+507, ADR-119).
- d867fa15fd #2391 (09-04): tasks 31387–31392, vLLM redesign (+610, ADR-117).
- e9d3d63e27 #2575 (09-10): task-32273, reasoning replay, including the "Override current Console model" collapsible.
- 149acda36b #2707 (09-18): tasks 32766–32773, Settings aligned with runtime (+964/−401, `_settings.tcss` +40).
- 3840187988 #2188 and a3a80d7459 #2170 (08-28): Settings UX critique burn-down and the dirty-state badge, tasks 23104/23108/23109/23110/23191/23192 and 1716/1717.
- be380a1a6f #2608 (09-11): tasks 32306/32458, F-key renumbering of destinations.

**Console popover and modal**
- 939dee8dc2 #2365 (09-04): tasks 30010–30014, the Conversation settings redesign (modal +3589/−566, `settings_screen` +768).
- f6896176c8 #2385 (09-04): task-31301, post-merge UAT gaps.
- 6a3092ad39 #2201 (08-29): commit 7cf89de6c0, which consolidates a long-lived branch. It carried task-22515 / ADR-095 (Apply persistence): popover +877, modal +1378.
- 1c0327b3bb #2703 (09-16): tasks 32707–32711. Preserves settings across repeated Apply, keeps subscription readiness off the UI thread, and amends ADR-012 and ADR-052.
- 8a6ba98c0d #2670 and a4a054a61f #2672 (09-13): wide responsive tiers for the modal and popover, no task id.
- 1103b28f71 #2668 (09-13): custom-endpoint fixes CE-001/003/004.
- 030fd9935d #2736 (09-19): tasks 32533/32534/32566/32829.
- c418d4516c #2397 (09-04): current and next-send spend in the popover, closed out by task-31591.
- b47f6da425 #2140 (08-27): tasks 18932.x, thinking settings (ADR-090).
- 2660d4de29 #1721 (08-16): task-16319, local thinking controls (ADR-066).
- 18384c80d1 #2204: task-575.
- 1549f2bef6 #1700 and 00240d0f1f #2678: tasks 16502/16503/32533, `Select.BLANK` crash fixes.

**CSS**
- e89f28d751 #2704 (09-17): the component-pattern library. It **created** `_settings.tcss` (+1438) and `_console_panels.tcss` (+2278); task-32596 and others.
- d6e2a46384 #2725 (09-18): tasks 32826–32828, `_console_panels` +104.
- cebe68148b #2729 (09-19): ratchet drift fix, `_console_panels` ±19.

## 4. Current max backlog task id

I ran `git ls-tree` over all 564 refs, including `backlog/tasks`, drafts, completed and `archive/{tasks,drafts}`.

| Scope | Max id | Where |
|---|---|---|
| origin/dev (c4225b5d38) | **task-32950** | Voice-Cloning dependency gate |
| Any ref | **task-32959** | `feat/wizard-omnivoice-tts`, local and origin |
| On disk, uncommitted (not on any ref) | **task-33000** | `.worktrees/personal-context-purge`, branch `codex/chatbook-personal-context-purge` |

The next free id by the ref sweep is **32960**. Use **33001** if the uncommitted 33000 should count too.