# TASK-3401.15/.16 Video Settings Runtime Design

## Goal

Restore the existing Video Generation settings path so users can reach the panel and persisted `[video_generation]` values reach the video runtime used by Console generation.

## Confirmed failures

1. `SettingsScreen._category_summaries()`, the domain contract table, and the Video Gen panel branch all include `SettingsCategoryId.VIDEO_GENERATION`, but `_category_groups()` omits it from Domain Defaults. The rail and category filter therefore have no button to expose or open.
2. The bootstrap TOML parser retains `[video_generation]`, but `load_settings()` does not project that table into its returned mapping. `Video_Generation.config` reads only from `load_settings()`, so it receives built-in defaults and an empty enabled-backend list.

Both failures were reproduced through the real Textual Console UAT in TASK-3401.14 before any generation request reached the configured server.

## Design

### Settings navigation

Add `SettingsCategoryId.VIDEO_GENERATION` to the existing Domain Defaults tuple immediately after Image Generation. Keep the current explicit category ordering and collapse/filter behavior. Do not introduce a second registry or derive the entire rail from domain contracts in this bug fix.

The existing summary, domain contract, compose branch, draft validation, persistence, and runtime-reset paths remain the owners of their current responsibilities.

### Runtime configuration projection

Read `video_generation` from the already parsed and decrypted `toml_config_data` in `load_settings()` and return it in `config_dict`, matching the existing `image_generation` pass-through pattern. `Video_Generation.config._read_video_generation_toml()` remains unchanged and continues to use `load_settings()` as the single profile-aware configuration boundary.

This preserves config-path overrides, bootstrap merging, decryption, caching, and the existing `reset_video_generation_runtime()` behavior without adding another TOML reader.

## Data flow

```text
config.toml [video_generation]
  -> config.load_settings()
  -> Video_Generation.config.get_video_generation_config()
  -> VideoAdapterRegistry
  -> Console /generate-video worker
```

Settings saves continue to persist through the existing writer and invalidate both the video configuration cache and adapter registry only after a successful save.

## Error handling

No error contract changes. Invalid Settings drafts remain blocked by the existing validator. Missing or malformed video configuration values continue to use the video loader's existing coercion and default behavior. A configured but disabled backend remains a terminal Console error; this fix only ensures persisted enablement is visible to that decision.

## Verification

- A focused Settings regression must fail when Video Generation is absent from `_category_groups()`, then prove the expanded group count, category search, and panel opening behavior. Existing focused panel/data-layer coverage must also remain green for the curated ComfyUI controls, ownership guidance, invalid-draft blocking, successful persistence, and save-triggered runtime refresh required by TASK-3401.15 AC #3–#4.
- A real scratch-TOML regression must fail when `load_settings()` omits `video_generation`, then prove the global and nested values survive initial load and forced reload into `get_video_generation_config()` and registry resolution.
- Run only tests related to modified production and test files, plus targeted Ruff, `py_compile`, and diff/privacy checks.
- After TASK-3401.15/.16 implementation and automated verification are complete, resume TASK-3401.14 as follow-up UAT—not additional implementation scope for either defect—and execute Base and Spectrum through the actual Console command against the configured trusted server.

## Scope boundaries

- No new dependency or abstraction.
- No domain-rail registry refactor.
- No direct TOML parsing inside `Video_Generation`.
- No workflow JSON changes.
- No prompts, server identity, credentials, generated media, or private source-workflow identity in tracked evidence.

## ADR check

ADR required: no new ADR

ADR path: `backlog/decisions/044-ephemeral-generated-video-storage-playback-and-streaming.md`

Reason: these are narrow reachability and configuration-projection corrections within the Settings-to-Console and ephemeral video boundaries already established by ADR-044.
