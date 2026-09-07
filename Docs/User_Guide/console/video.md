# Video generation, playback & streaming

The Console can generate short videos with AI backends, play them (in the
transcript or the full player), and stream videos from URLs — without ever
writing them into your chat database.

- **Generate**: `/generate-video [:backend] [@style] <prompt>`
- **Stream**: `/stream-video <url>`
- **Play**: select a video message → ▶ Play (full player) or click the
  in-card silent preview.
- **Configure**: Settings → Video Gen.

## Generating videos

```
/generate-video a kite flying over a harbor at dusk
/generate-video :comfyui a kite over the harbor
/generate-video @cinematic fog rolling over a suspension bridge
```

- `:backend` picks a non-default backend for one run.
- `@style` applies a style template (prompt language + sensible defaults
  for duration/fps/ratio). Built-ins: `cinematic`, `drone`, `timelapse`,
  `anime`. Your own styles live in config.toml under
  `[video_generation.styles.<id>]`.
- One generation runs per session at a time; the composer's **Stop**
  button cancels (a queued MiniMax task is cancelled remotely and does
  not bill to completion).
- **Paid backends ask first.** When `[video_generation]
  confirm_cost_estimate` is on (default), cloud backends show the
  billing shape (per generated second) and wait for confirmation.

### Backends

| Backend | Kind | Notes |
|---|---|---|
| `minimax` | cloud | MiniMax-H3 (768P/2K, 4–15 s). Needs an API key. |
| `comfyui` | local server | Your ComfyUI install + video workflow. |
| `stable_diffusion_cpp` | local binary | Wan2.x / MiniMax-H3 weights via the `sd` binary. |

Enable backends and set defaults in **Settings → Video Gen** (mirrors the
Image Gen page: default backend, per-backend enable/status/fields,
generation defaults, diagnostics for backend + ffmpeg/ffplay/yt-dlp
availability).

## Videos are ephemeral (by design)

**Generated videos are not stored permanently.** The bytes live in a
session-scoped folder (`generated_videos/<message-id>/`), never in the
database. The conversation keeps a `[video] <name>` marker and the
generation facts (prompt, backend, seed, duration, resolution) — nothing
else.

What that means in practice:

- **Restart the app and videos are gone** (default `retention =
  "session"`). The message stays as a named **tombstone** with all its
  facts and a working **♻ Regenerate** action — recreate the clip any
  time. This is intended behavior, not data loss.
- Want to keep clips longer? Set `retention = "ttl"` and
  `retention_ttl_hours` (default 24). A total store cap
  (`max_store_mb`, default 2048) always applies — the oldest videos are
  evicted first, even within a session.
- **"Save"** on a ready video card copies the file to
  `[chat.videos] save_location` (default `~/Downloads`) — the only way a
  video escapes ephemerality, and always an explicit act.
- MiniMax's download links expire quickly; the app downloads immediately
  and never keeps the URL.

## Playing videos

- **Silent preview**: click the ▶ Preview area in a ready video card
  (12 fps, capped at 30 s clips and 2K sources). Click again to pause;
  previews pause themselves when scrolled off-screen, and only one plays
  at a time. Needs the `video_playback` extra (`av`):
  `pip install "tldw_chatbook[video_playback]"`.
- **Full player**: select the message → **▶** — audio, seek (±5 s with
  `←`/`→`), `space` pause, `s` stop, `q` close, drift/dropped-frame stats.
  Renders in kitty/sixel/halfcell/ascii depending on your terminal.
  Needs `ffmpeg` + `ffplay` installed.
- **System player**: if ffmpeg/ffplay are missing, Play falls back to
  your OS default player with a one-line notice.

## Streaming videos

```
/stream-video https://example.com/clip.mp4
/stream-video https://youtube.com/watch?v=...
```

Streams play in the full player **without downloading**:

- Direct media URLs play as-is; page URLs are resolved with `yt-dlp` (a
  separate install, invoked as a subprocess only).
- Every URL — including every redirect hop and the resolved final URL —
  passes the app's egress (SSRF) policy before playback.
- Seek works when the server supports range requests; otherwise the
  seek keys are disabled and the footer says so.
- Streams are time-boxed (2 hours) and never written to disk.
- HLS/DASH streams with separate audio/video tracks are not supported
  yet (progressive single-URL streams only in v1).
