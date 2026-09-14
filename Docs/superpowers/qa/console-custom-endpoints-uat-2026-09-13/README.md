# Console custom endpoint registry — live UAT

Date: 2026-09-13 (session ran 2026-09-12 23:13 – 23:46)

Build: origin/dev tip `96a8f37aa7` ("Merge PR #2646 fix/custom-endpoint-ux-round2"), run from a
detached worktree at `/tmp/uat-dev`.

Server: the user's REAL llama.cpp server at `http://127.0.0.1:9099` (gemma-4-26B-A4B
Q4_K_M, native `/v1/models`). **No mocks anywhere** — every network interaction in this
UAT hit that server.

## Goal

Exercise the custom endpoint registry (ADR-146) end to end through the real Textual app
against the real server: first-run discoverability, the template creation flow, duplicate
management, real model discovery, a real send through a registry entry, switching across
entries, and F9 rename/edit/delete lifecycle.

## Method

1. Safety recipe per `backlog/docs/lessons-live-verification.md`: mock leftovers killed
   (ports 8231/8232 dead, only 9099 answering), app run exclusively from
   `/tmp/uat-dev` with `TLDW_CONFIG_PATH=/tmp/uat-scratch/config.toml`.
2. Scratch profile pre-seeded with `[general] users_name = "uat_ce"` and
   `[paths] data_dir = "/tmp/uat-scratch/data"` (both required), plus
   `[splash_screen] enabled = false` and `[model_catalog] auto_refresh_enabled = false`
   to keep the run offline except for the real server. The scratch config was validated
   with `tomllib` after every boot (the app rewrites it); snapshots kept per scenario.
3. tmux `-L uat`, 235x52, stderr attached to the pane (plus a tee that preserves the
   pane render while capturing crash tracebacks).
4. Keyboard-first driving; clicks injected as raw SGR mouse sequences at columns
   computed with python `line.find` (character columns). Note for reproduction:
   `tmux send-keys MouseClick` did NOT deliver events to the app in this environment;
   `ESC[<0;<col>;<row>M` / `...m` pairs did.
5. Isolation proofs: real config fingerprint before/after, marker-file check under
   `~/.local/share/tldw_cli`, and `lsof` handle checks (captures 02 and 26).

Severity: **S1** blocks the core task; **S2** major error risk/confusion; **S3**
avoidable friction; **S4** polish.

## Scenario results

| ID | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| FT-1 | First-run orientation: reach Console, open conversation settings, provider grouping, sentinel, `/endpoint` offered | **Pass** (one S3, one S4 observation) | 03–09 |
| FT-2 | Create endpoint #1 "Local llama" (llama.cpp family) at `http://127.0.0.1:9099` via the template flow; calm dialog; auto-switch; read-only URL hint; real discovery | **Fail (S1)** — creation + persistence + calm dialog pass; post-create auto-switch crashes the app (CE-001), so the switch/read-only-hint/entry-scoped discovery states were never reached | 10–14 |
| FT-3 | Create endpoint #2 "Gemma relay" from the entry provider (same-family starter, blank URL) | **Pass via substituted path** — the Console entry-provider start point is unreachable (CE-001); the identical same-family starter was exercised through F9 → Custom endpoints → Duplicate, which seeds from the entry. Starter preselects family only, blank name/URL, H4 placeholder shown | 15–17 |
| FT-4 | Endpoint #3 via Duplicate: "(copy)" prefill, rename "Gemma spare", keep URL, three entries | **Pass** | 18, 19, 19b |
| FT-5 | Real send "Reply with exactly: UAT-OK-9099" through endpoint #1 with its discovered model | **Blocked for the registry entry (CE-001); control executed** — with the built-in llama.cpp provider pointed at 9099 (provider-level endpoint), real discovery listed the served model ("Served by this endpoint now", Endpoint · Reachable, Model · Confirmed) and the real gemma server replied exactly `UAT-OK-9099` in ~6 s | 23, 24 |
| PU-1 | Switch across all three entries; readiness/model handling per entry | **Blocked (CE-001)** — every selection of a registry entry in Conversation settings crashes the app | 12, 13 |
| PU-2 | F9 management: rename (display name only); Edit URL to `.../v1` (equivalent after normalization) with re-resolve copy; Delete blocked listing the conversation; Detach references | **Partial** — rename, Edit URL (with the re-resolve status copy), and unreferenced Delete all pass. The referenced-conversation Delete block and Detach-references branch could not be exercised: no conversation can reference an entry because selecting one crashes (CE-001) | 20–22 |
| Final | One more send after the lifecycle flow to prove the surviving conversation works | **Pass** (on llama_cpp@9099): "Confirm still working: reply OK" → `OK` | 25 |

## Findings register

| ID | Sev | Finding | User impact | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- |
| CE-001 | S1 | Selecting a custom endpoint — or the automatic switch onto one after Create — raises an unhandled `InvalidSelectValueError: Illegal select value 'custom_ep:<slug>'` and the app exits. Root cause: the provider Select's option values use the dashed registry id (`custom-ep:<slug>`), but the rebase path (`_apply_rebased_state`, console_settings_modal.py:5102) assigns the canonicalized underscored form (`provider_config_key` rewrites dashes to underscores). Chain: `_endpoint_created` (4577) / `_provider_picker_selected` (5198) → `_switch_provider` (5250) → `_rebase_to` (5074) → `_apply_rebased_state` (5102). The registry entry itself persists correctly, so the entry survives the crash but can never be selected. Reproduced twice (create-switch crash #1; explicit selection crash #2, full traceback captured). The F9 Settings creation path does NOT crash — its `EndpointCreated` handler only refreshes the section. | The core value of the feature (use a named endpoint in a conversation) is unreachable: creation appears to fail violently, every later selection kills the app and loses the session. Blocks FT-2 post-create UX, PU-1, entry-scoped sends, and the referenced-delete branch. | 12, 13; config persisted `[custom_endpoints.local-llama]` before the crash | **fixed+verified** (PR #2668; re-run R1/R2 below — 05b–17b, 19b/20b) |
| CE-002 | S3 | While the blocking Console setup card is up, Tab always re-focuses the primary "Set up provider" button (`_focus_console_setup_modal_if_blocking`), so the "Write a note in Library" and "Use detected llama.cpp (127.0.0.1:9099)" actions are mouse-only. | A keyboard-only first-run user cannot take the fastest path to a working local provider (the detected-server action) or the notes alternative. | 03, 04; chat_screen.py `_focus_console_setup_modal_if_blocking` | Backlog |
| CE-003 | S3 | A conversation-scoped Base URL set and applied via "Use for this conversation" did not redirect the send: the send resolved the provider-level `api_settings.llama_cpp.api_url` (still `http://localhost:8080`) and blocked with "llama.cpp server is not reachable at http://localhost:8080… update Console provider settings". Only saving the provider-level endpoint in F9 unblocked sends. Attribution caveat: driver-click uncertainty exists, but the field verifiably held 9099 and Apply verifiably closed the modal before the blocked send. Entry-scoped sends could not be compared (CE-001). | A user who points "this conversation" at a working URL can still be blocked by a stale provider-level URL; the modal's "Use: this conversation only" scope promise is not what the send consumed on the built-in path. | 24 (send sequence), app log "Chat API Call - Routing to endpoint: llama_cpp" only after the provider-level save | **fixed+verified** (PR #2668; re-run R3 below — 23b–25b) |
| CE-004 | S4 | The template flow seeded from the built-in llama.cpp template prefills Models with the literal string `None` and persists `models = ["None"]` into the registry entry (F9 row then shows "· 1 model"). | Bogus model id lands in the registry and is presented as a real model count. | 10, 11; `[custom_endpoints.local-llama] models = [ "None", ]` | **fixed+verified** (PR #2668; re-run R5 below — 32b–34b) |
| CE-005 | S4 | The Ctrl+P command palette has no endpoint-creation entry; `/endpoint` discoverability ships in the composer slash-command popup ("Create a named custom endpoint from a template") plus the provider-list sentinel. The UAT brief expected a palette entry. Also: at 52 rows the conversation-settings action row sits below the fold with no in-pane shortcut hint (Ctrl+Enter applies). | Minor: two strong discoverability surfaces exist, the palette is not one of them; the fold-hidden Apply cost a wrong-turn send during the UAT. | 09 (composer popup), palette "endpoint" filter → "No matches found" | Keep / Backlog |

## Positive patterns to preserve

- First-run journey is genuinely guided: the app-level wizard detected the real server
  ("Found a local endpoint: http://127.0.0.1:9099"), and the Console setup card offered
  "Use detected llama.cpp (127.0.0.1:9099)" which configured provider + model in one
  click (01, 04, 06).
- The provider list groups Cloud / Local / Custom & legacy and ends with the
  "New custom endpoint…" sentinel row; the filter surfaces it instantly (07, 08).
- The template modal's calm dialog (H8) holds: before any edit the error banner is
  absent while Create stays gated; the H4 URL placeholder explains both defaults
  ("llama-server default :8080 · Chatbook default :9099") (10).
- The H5 same-family starter behaves exactly as designed: preselects the family with a
  blank name AND blank URL (never the entry's URL or the family default port), with the
  "(duplicate)" full-copy row one below (15).
- Duplicate prefills "<name> (copy)" and carries the URL; renaming changes only the
  display name — the status says so explicitly ("slug 'gemma-spare' unchanged") (18, 20).
- Edit URL normalizes family-appropriately: `http://127.0.0.1:9099/v1` saved back as
  the equivalent `http://127.0.0.1:9099`, with the status copy "Saved endpoint
  'local-llama'; existing conversations re-resolve on their next send." (21).
- Deleting an unreferenced entry is one click with clear status ("Deleted endpoint
  'gemma-spare'.") and the config write is the exact scoped delete (22).
- Real discovery against the live server is honest about scope: "Tests this endpoint by
  listing models; this does not test generation", then "1 model listed", "Served by this
  endpoint now", Endpoint · Reachable, Model · Confirmed, Generation · Not tested (23).
- Real generation worked first try through the provider pointed at the server: the
  gemma model replied exactly `UAT-OK-9099`, and a follow-up turn still worked (24, 25).

## Scope and limitations

- ONE physical server exists. All endpoints point at 9099, so multi-server wire variance
  (different llama.cpp builds, ollama/openai_compatible families, auth, error shapes) is
  NOT covered by this UAT.
- No mocks were used at any point; the only network peer was the real server at
  127.0.0.1:9099.
- Cross-URL re-resolution after an Edit URL change is not UAT-coverable with one server
  (both URLs are equivalent post-normalization); it is regression-tested at the provider
  gateway.
- CE-001 prevented: the post-create auto-switch presentation, the read-only Base URL +
  "managed by this endpoint" hint, switching across entries, an entry-scoped real send,
  and the referenced-conversation Delete/Detach branch. These need re-testing after the
  fix.
- The blocked-send transcript rows from the wrong-turn send (before the provider-level
  endpoint was saved) remain in capture 24's precursor states; they are harness journey
  artifacts, not product defects beyond CE-003's attribution caveat.

## Isolation evidence

- Real config `~/.config/tldw_cli/config.toml` SHA-1 identical before/after the whole
  session: `6c2577fb39ab993eb01fa7c6a69900e36e9d59db` (26).
- No file under `~/.local/share/tldw_cli` changed after the pre-session marker
  (checked again at the end) (26).
- `lsof` on the app process showed zero real-profile handles; all data handles under
  `/tmp/uat-scratch/data/uat_ce/` (02, 26).
- The invalid captures from the aborted earlier mock-based attempt were deleted before
  this run; all captures here are from this session against the real server.

## Captures

Numbered `NN-scenario-step.txt` in `captures/` (27 files), including the crash
traceback (12), the app-log crash context (13), and config snapshots after FT-4 (19b)
and at the end (26b).

---

# Re-run (post-#2668)

Date: 2026-09-13 (session 09:13–09:33 PDT).

Build: origin/dev tip `1103b28f711` ("Merge PR #2668 fix/custom-endpoint-uat-ce001" —
CE-001 crash fix, CE-003 endpoint-draft fix, CE-004 None-prefill fix, plus the
send-path slug-mangle fix via `provider_identity_key`), run from a detached worktree
at `/tmp/uat2-dev` (removed after captures were extracted).

Server: the same REAL llama.cpp server at `http://127.0.0.1:9099` (gemma-4-26B-A4B
Q4_K_M). **No mocks.**

Scratch profile `/tmp/uat2-scratch` (`users_name = "uat_ce2"`, data_dir
`/tmp/uat2-scratch/data`, splash off, model catalog auto-refresh off), first_run
pre-seeded from the prior session's validated shape so the run started straight at the
Console. The provider-level llama.cpp endpoint was deliberately left **unset** for R1–R3
(resolves to the stale default `http://localhost:8080`), making R3's pre-state faithful
without needing a blocked control send. Scratch TOML validated with `tomllib` after
every boot; snapshots at 36b.

Method: same safety recipe — tmux `-L uat2`, 235x52, no stderr redirect, keyboard-first
with raw SGR mouse clicks at python `line.find` columns, Enter ≥0.4 s apart. Isolation
proofs at 01b and 35b.

## Scenario results

| ID | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| R1 (was FT-2 tail, PU-1) | Create "Local llama" (llama.cpp) at `http://127.0.0.1:9099` via the settings-modal sentinel template flow; post-Create auto-switch must NOT crash; then explicitly select each entry and switch across all entries | **Pass on the crash assertion** — post-Create auto-switch is calm: modal returns with read-only Base URL `http://127.0.0.1:9099` + "Managed by this endpoint entry — rename or edit it in F9 Settings › Providers & Models." hint, model carried, entry persisted (`models = []`); the same-family starter created "Gemma relay"; explicit selection of each entry works with no crash (Local llama → Gemma relay → Local llama). **One new S3 (CE-006)**: after the auto-switch (and after any rebase onto an entry inside one modal session) the Provider Select renders blank and its dropdown loses all options until the modal is cancelled and reopened — entry→entry switching required a reopen between switches | 04b–17b |
| R2 (was FT-5, entry-scoped) | With endpoint #1 selected and its model, send "Reply with exactly: UAT2-OK-9099" through the REGISTRY ENTRY | **Pass** — "Use for this conversation" applied (status line shows `Provider: custom-ep:local-lla…`); the real gemma server replied exactly `UAT2-OK-9099` in ~6.2 s; app log routes via the family key (`Routing to endpoint: llama_cpp`) with `provider_entry` succeeding — first live-verified entry-scoped send. Discovery evidence note: in the settled entry state the settings modal offers no connection probe ("No non-billable live connection check is available for this provider.", "Endpoint · Not tested") and the F9 row shows "0 models" — entry discovery evidence is not surfaced (known follow-up, unchanged) | 18b–20b, 27b |
| R3 (CE-003 fix) | On the BUILT-IN llama.cpp provider in Conversation settings, set Base URL to `http://127.0.0.1:9099`, apply "Use for this conversation", then send | **Pass (fix verified)** — provider-level URL stayed unset/stale (`Providers & Models` screen still showed `http://localhost:8080`, no `[api_settings.llama_cpp]` in config at any point), yet the send used the conversation-scoped 9099 and the real server replied exactly `UAT2-CE003-OK`. Pre-#2668 this same flow blocked with "llama.cpp server is not reachable at http://localhost:8080…" | 21b–25b, 27b |
| R4 (was PU-2 blocked tail) | With a conversation actively using an entry: F9 → Delete that entry → blocked copy listing the conversation → Detach references → entry deleted, conversation survives → one more send | **Pass** — Delete on the referenced "Local llama" surfaced "Detach references" and the blocked copy: "Delete blocked: 1 conversation(s) still use this endpoint (Reply with exactly: UAT2-OK...). Detach references to keep each conversation's current endpoint as conversation-only, then delete -- or switch those conversations' provider first." Detach deleted the entry (config retains only `gemma-relay`/`probe-spare`) with status "Detached the referencing conversation(s) -- their endpoints are now conversation-only -- and deleted 'local-llama'."; the surviving conversation sent "Confirm still working: reply OK" → `OK` (DB-verified; log completed in 1.6 s) | 26b–31b |
| R5 (CE-004 fix) | Create one more entry from the built-in llama.cpp template; Models prefill must NOT contain "None"; created entry's models list placeholder-free | **Pass (fix verified)** — the default llama.cpp provider model list verifiably resolves to `['None']` (same preconditions as the original finding), yet the template Models input shows only the "Comma-separated model ids" placeholder (no prefill); typing `None, gemma-probe` persisted `models = [ "gemma-probe",]`; zero occurrences of "None" in the final config; R1's entries persisted `models = []` | 32b–34b |

## New findings register (re-run)

| ID | Sev | Finding | User impact | Evidence | Disposition |
| --- | --- | --- | --- | --- | --- |
| CE-006 | S3 | After the post-Create auto-switch — and after any rebase onto a registry entry inside one settings-modal session — the Provider Select renders its NULL placeholder ("Choose or search providers") instead of the entry name, and the dropdown's option list is emptied: filters for "custom" (sentinel), "Local llama", or "relay" all return "No matching providers. Clear the filter.", so entry→entry switching inside the open modal is impossible. In the same settled state the "Test connection & list models" button disappears ("No non-billable live connection check is available for this provider."). Cancelling and reopening the modal restores the full option list including both entries, and explicit selection then works without crashing. No exception is logged; the draft state itself is correct (read-only Base URL, managed-by hint, model, dashed identity in the app log). | The switched-to state is functional but the Provider field looks empty and the entry list appears to vanish; a user must close and reopen settings to switch endpoints. Cosmetic/flow defect only — no crash, no data loss, sends unaffected. | 07b, 08b, 10b, 11b, 15b (blank label + "No matching providers"); 12b–14b (reopened modal lists entries, selection sticks) | Backlog |
| CE-007 | S4 | Late in the run, after several settings-modal open/apply cycles, tab-bar navigation away from the Console stopped working (four `nav-settings` button presses logged, zero "Navigation requested" lines) and Ctrl+P opened nothing while the composer input held focus (palette opened only after clicking a neutral area first). Recovered via palette → "Open Settings Tab" once focus moved. Not reproduced from a fresh boot in this session; attribution to the retained-modal sequence is plausible but unproven. | A long Console session may need a focus shuffle or palette route to change tabs. | app log 09:27:46–09:28:26 (nav-settings without navigation); 26b (hub reached via palette) | Backlog |

## Re-run isolation evidence

- Real config `~/.config/tldw_cli/config.toml` SHA-1 identical before/after:
  `6c2577fb39ab993eb01fa7c6a69900e36e9d59db` (01b, 35b).
- Zero files under `~/.local/share/tldw_cli` newer than the pre-session marker
  (unfiltered `find -newer`, 35b).
- `lsof` on the app process (18703): no handles on any real-profile path; cwd
  `/private/tmp/uat2-dev`; all data handles under `/tmp/uat2-scratch/data/uat_ce2/`
  (01b, 35b). Other concurrently running `tldw_chatbook.app` processes belonged to
  separate sessions' tmux servers with their own scratchpad `TLDW_CONFIG_PATH`s and
  were not touched.
- tmux `-L uat2` killed and `/tmp/uat2-dev` removed after captures were extracted;
  the scratch TOML validated after every boot (01b, 35b).

## Re-run captures

Numbered `NNb-*.txt` / `NNb-*.toml` in `captures/` (37 files): 01b isolation proof,
02b–17b R1, 18b–20b R2, 21b–25b R3, 26b–31b R4 (31b includes the DB evidence for the
surviving-conversation send), 32b–34b R5, 35b final isolation proof, 36b final config
snapshot.
