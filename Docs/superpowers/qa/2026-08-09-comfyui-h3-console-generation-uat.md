# ComfyUI H3 Console Generation UAT — 2026-08-09

## Scope

TASK-3401.14 tests the packaged Base and Spectrum H3 workflows through the real Textual Settings and Console surfaces. Evidence is intentionally prompt-free, host-free, credential-free, and media-free.

## Environment and safety

- The app ran with isolated home, config, and data directories under a disposable temporary root.
- A read-only prerequisite probe against the configured trusted server found both packaged workflow node sets and the required video-output class.
- The run was configured only with disposable profile paths. A post-run containment check found a byte fingerprint change in the separate default config, so the exact validated snapshot was restored as a precaution.
- No generated-media filename, prompt, credential, server identity, or private source-workflow identity is recorded here.

## Results

| Check | Result | Sanitized evidence |
|---|---|---|
| Server prerequisites | Pass | Base and Spectrum prerequisites plus the video-output class were present. |
| Reach Video Gen Settings | Pass | Category search opened the existing panel; curated ComfyUI values and ownership guidance were visible. TASK-3401.15 is live-verified. |
| Runtime settings projection | Pass | A workflow change saved through Settings and the same process used it for the next Console request. TASK-3401.16 is live-verified. |
| Base Console generation | Pass | The real command produced a ready card with 864×480, 24 FPS, and 5-second displayed metadata. |
| Spectrum Console generation | Pass | Spectrum was selected and saved through Settings, then the same command path produced a ready card. |
| Base stored bytes | Pass | MP4-family container; H.264 864×480 at 24 FPS; 124 frames; 5.167 seconds; AAC audio; observed MIME `video/mp4`. |
| Spectrum stored bytes | Pass | MP4-family container; H.264 864×480 at 24 FPS; 124 frames; 5.167 seconds; AAC audio; observed MIME `video/mp4`. |
| Cancellation | Pass | Stop cleared the running queue, rendered an explicit user-cancelled terminal message, left no pending card, and retained no partial file. |
| Full player | Pass | The real selected-message Play action opened the full Textual player, started its decoder/audio-bearing playback pipeline, rendered terminal video frames to completion, and reported a 5-second run with bounded drift and no player error. The source contained one H.264 video stream and one AAC audio stream with 163 audio packets. |
| Save copy | Pass | The real selected-message Save action wrote one MP4 copy. The copy was byte-identical to the managed artifact and independently probed as MP4-family, 864×480 at 24 FPS, 5.167 seconds, with H.264 video and AAC audio. |

## Defect evidence

- The generated-video directory key did not match the persisted video-message ID. The card was initially ready from the live in-memory state but could not resolve the same bytes after restart even under TTL retention.
- A Console lifecycle transition under session retention reapplied the startup sweep and removed a current-run generated video. Session retention must describe the app session, not each screen instance.
- Explicit inline-preview activation was followed by an app-level unhandled `AttributeError` and shutdown in the isolated persistent log. The log contained no prompt or media bytes.
- These issues are tracked atomically in TASK-3401.17, TASK-3401.18, and TASK-3401.19. No production fix was made inside this UAT task.
- A follow-up run after those tasks completed verified that identity persistence, preview lifecycle, remount retention, full-player launch, and save-copy no longer block this acceptance path.
- The follow-up session's containment check also found a byte fingerprint change in the unrelated default config: built-in default keys had appeared while existing values remained unchanged. Because unrelated concurrent activity existed, that observed delta did not identify its writer or prove that the isolated app's startup-to-approved-quit lifecycle wrote the file. Restoring the exact validated pre-run snapshot remained the appropriate precaution.
- TASK-15674 later ran a controlled current-development reproduction with distinct effective and decoy configs. Approved-quit persistence ran, selected the exact effective profile path, and left the decoy default config byte-identical. No product fix was required; regression coverage now locks that verified boundary.

## Cleanup

The isolated app session was stopped. Before deletion, the validated scratch root contained no symlinks and no partial downloads. Its disposable profile, managed generation, save copy, and diagnostic state were permanently removed. The real user config was restored from the validated pre-run snapshot and verified byte-for-byte before that recovery snapshot was removed. TASK-3401.14 now satisfies all acceptance criteria.
