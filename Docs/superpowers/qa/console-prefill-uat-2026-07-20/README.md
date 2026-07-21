# Console `/prefill` UAT — 2026-07-20

Fresh-profile user-acceptance drive of the response-prefill feature (PR #729) in the real
TUI (tmux 235x52, scratch `TLDW_CONFIG_PATH` profile `verify_prefill`, llama-server :9099,
model gemma-4-26B heretic quant). Distinct from the engineering smoke: run as a new user,
UI affordances only.

## Results

| # | Leg | Result |
|---|-----|--------|
| 1 | Discoverability: unknown-command hint lists `/prefill` | PASS |
| 2 | Bare `/prefill` with nothing armed → "No prefill armed." | PASS |
| 3 | Bracket-containing one-shot arm → confirmation renders literally, **no `\[` artifact** (validates the escape-drop) | PASS |
| 4 | Prefilled send → literal continuation (`Sure! Here you go: ["alpha",` → `"beta"]`) | PASS |
| 5 | One-shot consumed after completed send | PASS |
| 6 | `/prefill pin` → subsequent send starts in voice | PASS |
| 7 | Inspector "What's in play" shows pinned row | **FINDING → FIXED** (below) |
| 8 | Pin restored across full app restart + rail resume (conversation metadata) | PASS |
| 9 | UI regenerate (select message, `r`) → variant starts with the *current* pin | PASS |
| 10 | `/prefill pin` with no text → usage error, draft kept | PASS (prior session) |
| 11 | `/prefill clear` → metadata key removed merge-safely in SQLite | PASS (prior session) |

## Finding (fixed in-branch, commit 0b596d067)

The prefill inspector rows rendered as **unrouted rows at the very bottom** of the
inspector (below "Selected Message", above "Chat Dictionaries") instead of inside the
"Selected Conversation" section — `ConsoleRunInspector` routes rows through hardcoded
label tables (`_ROW_ID_BY_LABEL`, `_ROW_GROUPS`) that the feature never registered its
labels in. In a 52-row terminal the rows sat below the fold, i.e. effectively invisible.
Fix: registered both labels with stable ids in the Selected Conversation group + widget
test (`test_prefill_rows_route_into_selected_conversation_group`). Verified live
post-fix: `Prefill (pinned)` renders directly under "Resume state".

## Observations (non-blocking)

- The test model (heretic Gemma quant) leaks `<|channel>thought` tokens into prefilled
  continuations — model-template noise, present in raw curl too, not feature-related.
- The message-action "Guide" line (`♻ Regenerate ---> Continue …`) is a legend, not
  buttons; the actual affordance is keyboard (`r` etc.) with a message selected. Fine
  once known, but a first-time user may click the legend expecting buttons (pre-existing
  UX, not prefill-specific).
- The two `_sync_console_*` calls in the `/prefill` handler's refresh tail do not rebuild
  the inspector; the inspector catches up on the next UI-sync tick (message append
  triggers one, so in practice the row appears immediately after the confirmation row).
