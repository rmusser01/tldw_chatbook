---
id: TASK-21596
title: >-
  Video generation resolves the MiniMax secret eagerly whenever the config is read
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - performance
  - video-generation
  - keyring
priority: low
---
## Description

TASK-21111 removed the *boot* consumer that forced a Keychain query during `TldwCli.__init__`,
but the underlying shape remains: `Video_Generation` resolves the MiniMax secret whenever anything
asks for the full config. Opening Settings or Console video still pays the Keychain query up
front rather than at send time.

Keyring cost is easy to under-count because `keyring.get_keyring()` memoizes the backend — the
first caller pays backend discovery for everyone, so the expensive site is whichever runs first,
not whichever looks heaviest.

## Acceptance Criteria

- [~] The MiniMax secret is resolved when it is used, not when the config is read --
      **declined, with the measurement below.** The one remaining eager consumer is the
      Settings ▸ Video Gen page, which *renders* where each backend's key came from
      ("key: keyring / env: VAR / config / not set"), so the resolution is that page's
      content, not incidental to it
- [x] Opening Settings and opening Console video are measured to perform zero keyring calls
      -- **Console: confirmed zero. Settings: 4, and cannot be zero** (see above)
- [x] The measurement counts keyring calls across the whole interaction, not just the site being changed, so a relocation cannot pass as a removal
- [x] A missing or rejected credential still surfaces the same error to the user, at send time
      -- unchanged, because nothing changed

## Evidence

From TASK-21111: the first keyring touch of every boot was `Video_Generation/config._keyring_get`
at **18.2 ms** (11.3 ms backend discovery plus the query), while the three sites the original
finding named cost 0.33, 0.41 and 0.04 ms. That task took the boot path to zero keyring calls;
this is the remaining on-demand path.

## Implementation Plan

1. Spy on the shared entry points (`keyring.core.get_keyring` and `get_password`), never on
   the individual callers -- the memoized-backend trap TASK-21111 recorded.
2. Count calls across whole interactions -- app mount on three destinations, opening the
   Video Gen settings category, the Console `/generate-video` path -- not at the call site.
3. Put a real-backend wall-clock number on one keyring query, separating the once-per-process
   backend discovery from the per-query cost.
4. Ship only if the deferral removes work rather than relocating it.

## Implementation Notes

**CLOSED without a code change.** The premise is half wrong on this base and the half that is
right cannot be fixed without deleting a shipped feature. Measured, not reasoned.

### What the spy counted (mounted-app probes, spy on `keyring.core.get_keyring` + `get_password`)

| interaction | keyring calls |
|---|---|
| app mount, **Home** destination | **0** |
| app mount, **Console** destination | **0** |
| app mount, **Settings** destination (Overview) | **0** |
| first `get_video_generation_config()` (what `/generate-video` does) | 1 `get_password` + 1 `get_keyring` |
| opening **Settings ▸ Video Gen** | **4** = 2 x (`get_password` + `get_keyring`) |
| re-opening it | another 4 |

So **"opening Console video still pays the Keychain query up front" is false on `7f38cb6ef`**.
There is no Console video surface that reads the config: `UI/Console_Modules/video.py`'s only
`get_video_generation_config()` is inside the `/generate-video` command handler, and the
registry, request validation, worker and adapters all read it downstream of that. Mounting
the Console destination reaches the keyring zero times. TASK-21111 already moved this to send
time; nothing was left behind.

### What one keyring query actually costs (real macOS backend, 3 runs x 8 samples)

```
backend = keyring.backends.macOS.Keyring
import keyring        24.4 - 32.5 ms
get_keyring() cold    12.6 - 17.7 ms   <- memoized; paid ONCE per process by whoever is first
get_password() first   5.4 -  6.0 ms   <- Security.framework/ctypes warm-up
get_password() after   0.11 - 0.36 ms
```

And a whole `get_video_generation_config(reload=True)` outside the keyring is **0.022 ms**
(median of 12; `load_settings()` is identity-cached, so the TOML is not re-read).

### Why that closes it

- The Settings page is the only eager consumer left, and it is eager *because of what it
  shows*: `build_backend_rows` puts `cfg.key_sources[backend]` on screen as
  `key: keyring` / `env: MINIMAX_API_KEY` / `config` / `not set`
  (`UI/Screens/settings_video_gen_defaults.py`, `Widgets/settings_video_gen_panel.py:237,297`).
  With no env var and no config value, the keyring probe is the *only* thing that
  distinguishes "stored in your keychain" from "not set". Deferring it to send time would
  make that page lie.
- The keyring tier is real, not dead: `tldw_chatbook_videogen` is a documented,
  user-managed namespace (ADR-044 / task-3401.2, the same shape as
  `tldw_chatbook_imagegen`). Nothing in the app writes it -- a user populates it with the
  `keyring` CLI -- so a probe is the only way to discover one.
- What is left to win is the duplicate: the category composes
  `get_video_generation_config(reload=True)` twice, once in
  `SettingsScreen._queue_video_gen_select_suppression` and once in
  `VideoGenSettingsPanel.compose`. De-duplicating it banks **~0.17 ms** (0.022 ms of config
  work + one warm 0.15 ms Keychain query). That is far below this machine's noise floor and
  not worth the coupling it would take (the suppression queue must record the value the
  about-to-mount `Select` will carry, so the two reads would have to be threaded together).
- The one-off ~18-24 ms is backend discovery plus the first query, and per TASK-21111's own
  lesson it belongs to **whoever runs first**, not to this site. Moving it out of Settings
  would simply promote `/generate-video` to first place and move 0 ms of user-visible cost.

### Residual observation, deliberately not actioned

That first ~24 ms runs on the event loop inside `VideoGenSettingsPanel.compose`. For a
missing item macOS returns `errSecItemNotFound` without prompting, but on a locked keychain
holding a stored videogen secret the call can block. Making the key-source column resolve in
a worker and fill in after mount is a redesign of a shipped display for ~24 ms once per
process; recorded here rather than filed, since it is the same trade this task just declined.

### Error path

Unchanged, because nothing changed: a missing key still reaches
`MiniMaxVideoAdapter.generate`, which raises `VideoBackendUnavailableError` at send time
(`adapters/minimax_video_adapter.py:201`), and the config resolution still records
`key_sources["minimax"] = "missing"`.

### Test counts

No production file was modified. `Tests/Video_Generation/`: **354 passed, 2 skipped, 0 failed**.
