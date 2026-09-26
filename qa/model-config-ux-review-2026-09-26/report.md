Method: three isolated reviewers (A1 Settings live, A2 Console live, B detector + static evidence), then a workflow with 4 refute-first fact-checkers, 1 backlog/ADR checker, 2 competing designers, and 1 judge. Tested at dev c4225b5d38, full screen only: 211x44 (ThinkPad) and 235x52 (MacBook). All runs used throwaway profiles; the real config is byte-identical.

# Model configuration UX review — Settings ▸ Providers & Models + Console model surfaces

## Design Health Score

| # | Heuristic | Settings | Console | Key issue |
|---|---|---|---|---|
| 1 | Visibility of system status | 1 | 2 | Test result is a pipe-joined dump that appends evidence twice. A known-failed connection test never reaches the Console header. Save copy never says what it applies to. |
| 2 | Match with the real world | 1 | 1 | Config keys in the UI (`api_settings.openai.api_base_url`, `chat_defaults.streaming is canonical…`). Raw Top P/Min P/Top K/Presence/Frequency with no help. "Console Defaults" is not a real category. |
| 3 | User control and freedom | 3 | 3 | Drafts survive and Revert confirms (good). But Esc in Chat settings discards edits without asking, which conflicts with ADR-031 / task-16211. |
| 4 | Consistency and standards | 1 | 1 | 4 editing surfaces, 3 of them titled "Conversation settings". Streaming is a Select, a Checkbox and 2 toggle buttons. Settings and the Console modal invert Input/Select heights. Labels drift between surfaces. |
| 5 | Error prevention | 2 | 1 | A provider switch inherits the global default model (a local model under Anthropic). Anthropic shows 4 sampler fields it silently drops. |
| 6 | Recognition over recall | 1 | 1 | The current model is not marked in either picker. No recents. Temperature is below the fold in the "quick" popover. |
| 7 | Flexibility and efficiency | 1 | 1 | Alex's switch-and-tune task takes 45 actions. The A/B toggle takes 16–20 keys. In Settings, Model is Tab stop #23. F6 is broken in Settings. `/model` ignores its argument. |
| 8 | Aesthetic and minimalist design | 1 | 1 | Only 4–8% of modal cells hold text. The popover's rows are 56% blank. A collapsed section costs 7 rows. The Settings card is about 115 rows (3.5 viewports at 211x44). |
| 9 | Error recovery | 1 | 2 | An unreachable server leads with "configuration is complete". The failure is the 8th pipe segment, with no next step. |
| 10 | Help and documentation | 2 | 1 | No inline explanation of sampling parameters. The inspector's help is written in config-key terms. |
| **Total** | | **14/40** | **14/40** | **Poor** on both surfaces |

## Design specificity verdict

The mechanics are specific to this product and good: provenance (Inherited/Edited), readiness rows, per-model drafts, discovery, and staged drafts with revert. The presentation is a generic web form stretched to fill a terminal. Every control is 3 rows tall and 110–136 columns wide for a 4-character value. There are three nested frames whose borders render at 1.0–1.05:1 contrast, so the structure is invisible while still costing columns. The complaint is accurate and measurable: at full screen the density goes to chrome and blank rows, not content.

The deterministic scan is blind here. The detector is web-only and skips `.py`, and TCSS never matches its rules, so its zero findings mean nothing. All evidence comes from live captures and code tracing.

## Verified corrections (the reviewers were wrong about these)

- **The "P0: Settings saves don't reach the open chat" is BY DESIGN** (ADR-095:26, TASK-22515 AC#10, settings.md:244). Open chats keep their own settings. The real defects:
  - The UI never says so (`settings_screen.py:30215`, `:30242`).
  - An untouched chat on a keyless provider never converges (`session.py:3783-3786`).
  - The first-run handoff fires only for the "Start chatting" exit (`FirstRunSetupWizard.py:9936`).
- **"Ready" for an untested local endpoint is BY DESIGN** (TASK-30011 AC#2, ADR-114). The defect: test evidence lives in 3 local stores (`console_settings_modal.py:1304`, `settings_screen.py:2949/13316`), so a known failure never reaches the Console header (`chat_screen.py:9865`). That violates TASK-30011 AC#6.
- **Test Provider not checking cloud keys is BY DESIGN** (ADR-012:33, TASK-386). This is a labelling problem ("Test" promises more than it checks). The double-appended evidence IS a bug (`settings_screen.py:15150` + `:15462`).
- **Discover → "Save selected" is BY DESIGN** (ADR-002). Typeahead of discovered IDs already exists (TASK-369). Friction remains only when a model is already set.
- **Model-chip truncation at 25 characters was user-requested** (TASK-1671). **Legacy aliases are intentional** (task-180, ADR-066), and may be hidden but not deleted.

## Priority issues

1. **[P1] Full-screen space goes to blank rows and over-wide fields.**
   - Console modal:
     - Inputs are 3 rows tall (`MODAL_CONTROL_HEIGHT=3`, `console_settings_modal.py:168`).
     - Three leaked global rules make a collapsed section cost 7 rows (`_conversations.tcss:298-303`, `:313-316`, `_evaluation_unified.tcss:59-63`).
     - Inputs are about 125 columns wide at 211 columns.
     - At 211x44 only the Connection block is visible without scrolling.
     - The "▼ more" hint shows even at the bottom of the scroll (`:3861`).
   - Popover: 170x32 with 18 blank rows. The 2x2 button grid uses 3-row slots for 1-row buttons (`console_model_popover.py:253-261`).
   - Settings card: about 115 rows, with Prompt-cache snapshots ahead of Connect.
   - **Fix:** a compact control-height token, scope the leaked CSS, content-sized widths, one frame level, and core-first ordering.
2. **[P1] Legibility: you can't see boundaries, focus, or state.**
   - Field borders are 1.05:1 and the modal frame is 1.01:1.
   - Settings rail focus is 1.1:1 (`_settings.tcss:198`) and the select highlight is 1.12:1.
   - Placeholders are 2.8–3.5:1, and a placeholder is indistinguishable from a value.
   - A focused primary button gets *darker*.
   - Checkbox state is colour-only in the model catalog and discovered list.
   - The OptionList focus border clips the first character of every provider ("nthropic").
   - **Fix:** boundary and focus tokens at ≥3:1, On/Off words, a reverse-video focus bar, effective values shown instead of placeholders, and padding on the OptionList.
3. **[P1] One job, four editors, four vocabularies.**
   - Four editing surfaces: popover, Chat settings modal, Settings, wizard.
   - Three surfaces are titled "Conversation settings".
   - Streaming has 4 control types.
   - Labels drift: Budget strategy/Budget mode, Think budget/Budget, Endpoint/Base URL, When limit nears/Behavior.
   - Underneath: three field-support projections and three test-evidence stores.
   - **Fix:** three named surfaces with one job each, one field table, one evidence owner.
4. **[P1] Model identity breaks on provider switch.**
   - Root cause: `resolve_effective_chat_configuration` ranks `chat_defaults.model` above the target provider's model (`console_session_settings.py:1269-1275`). This is a regression against task-14812 AC#6.
   - The current model is never marked in `ModelSearchPicker` (`:643`, `:697`, `:868`).
   - **Fix:** guard the default by provider; pair-only selection; a ● CURRENT mark.
5. **[P1] Status is unscoped or overclaims.**
   - Save doesn't say "applies to new chats".
   - A known-failed test never reaches the Console header.
   - Test results are a pipe dump with duplicated evidence.
   - "Test" implies a key check that never happens.
   - **Fix:** an "Applies to" line on every commit surface; shared evidence; labelled result rows; readiness words "Ready · not tested / verified HH:MM / reachable HH:MM / Not ready · reason".
6. **[P2] The keyboard path is long and partly broken.**
   - Alex's task takes 45 actions; A/B takes 16–20 keys.
   - In Settings, Model is Tab #23 and Generation defaults about #48.
   - F6 is broken in Settings (a documented limitation, `settings.md:877`, but it violates ADR-031).
   - `/model` ignores its argument, and Ctrl+Enter is unadvertised.
   - Esc silently discards the draft.
   - **Fix:** the switcher (14 keys; A/B 2), F6 wiring, a dirty guard on Esc, `/model <query>`.
7. **[P2] Jargon and dishonest fields.**
   - Raw sampler names with no help.
   - Config keys in UI copy.
   - "Override current Console model" actually holds reasoning replay.
   - Anthropic shows Min P, Seed, Presence and Frequency, which are silently dropped (`console_chat_controller.py:774-802` treats every sampler as universal).
   - **Fix:** one help line per field, and supported fields = capability ∩ `PROVIDER_PARAM_MAP`.

## Persona red flags

- **Jordan (first-timer):**
  - Cloud path: 3 actions, but a fake key reads "configuration is complete".
  - Local path: about 14 action groups, 2 scroll trips and 2 dead ends. Test is blocked until a model ID is typed. Discover doesn't set the model. The error hides in segment 8 of the pipe dump. The provider list clips first letters.
  - The setup card teleports to F4 with no way back.
  - The wizard renders an auth failure as a selectable radio option.
  - The starfield card is on the PRODUCT.md anti-reference list.
- **Alex (power user):**
  - Console: 45 actions across 2 modals and a disclosure; Tab ×13 to reach Apply; A/B 16–20 keys with no recents.
  - Settings: rail focus is invisible, F6 is broken, and about 60 keystrokes with no word on which chat the change affects.
  - The provider switch silently changed the model; temperature silently reverted on the switch back.
- **Sam (keyboard-only, low vision):**
  - Every boundary and focus cue is under 1.2:1.
  - Checkbox state is colour-only.
  - The ghost suggestion is 2.8:1 and is accepted with an undiscoverable → key.

## Minor observations

- "1 models available" (`model_search_picker.py:594`). Live catalogs are silently capped at 20 (`:640`).
- A toast covers the footer actions. Context-tab Select labels sit one row below their controls. "Base URL" label with no input for cloud providers.
- Inputs clear to their placeholder on focus, hiding the committed value while editing.
- The 14 auto-refresh checkboxes all show green while "Refresh on startup" is off.
- `SettingsURLInput` inserts a zero-width space after `http` (`settings_screen.py:1318`), so copied URLs may be corrupted.
- Keyless providers get `api_key_env_var` / `credential_source` written on save.
- Hover tooltips leak into the row above the footer.

## Recommended redesign: "Switchboard with field truth" (judge: 42 vs 41; mockups in mockups-211x44.md)

Three surfaces, one job each:
- **Switch model** (Alt+M, `/model <query>`): provider·model pairs grouped PREVIOUS / RECENT / READY / NEEDS SETUP, with a one-row value strip.
- **Chat settings** (Ctrl+O): Core first, Sampling/Connection as 1-row disclosures, a Source column, a help line per field, and "Applies to this chat only".
- **Settings ▸ Providers & Models** (F4): Connect / Default model / Model defaults / Advanced. The card drops from about 115 rows to about 25, and Model moves from Tab #23 to #5.

Budgets: Alex 14 keys, A/B 2, change model only 5, first run with a local server 2, first run with a cloud key about 11.

Phases:
1. Root-cause fixes.
2. Field table and honest copy.
3. Density tokens and CSS leaks.
4. Switcher.
5. Shared readiness evidence and the key check.
6. Chat settings layout.
7. Settings card reorder.
8. First-run connect in place.
