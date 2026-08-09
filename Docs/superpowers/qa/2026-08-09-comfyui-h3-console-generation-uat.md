# ComfyUI H3 Console Generation UAT — 2026-08-09

## Scope

TASK-3401.14 tests the packaged Base and Spectrum H3 workflows through the real Textual Settings and Console surfaces. Evidence is intentionally prompt-free, host-free, credential-free, and media-free.

## Environment and safety

- The app ran with isolated home, config, and data directories under a disposable temporary root.
- A read-only prerequisite probe against the configured trusted server found both packaged workflow node sets and the required video-output class.
- No production profile path was targeted.
- No generated-media filename, prompt, credential, server identity, or private source-workflow identity is recorded here.

## Results

| Check | Result | Sanitized evidence |
|---|---|---|
| Server prerequisites | Pass | Base and Spectrum prerequisites plus the video-output class were present. |
| Reach Video Gen Settings | Blocked | The panel and contract exist, but the category is absent from the Domain Defaults navigation group. See TASK-3401.15. |
| Console generation submission | Blocked | Persisted video-generation settings were absent from `load_settings()` output, so the runtime reported no enabled backend before any submission. See TASK-3401.16. |
| Terminal error hygiene | Pass for observed blocker | The Console displayed a terminal system message and did not create a pending video card. |
| Partial-media hygiene | Pass | The isolated profile contained zero generated-video or partial-download files after the rejected command. |
| Base/Spectrum media validation | Not run | Submission was blocked before the server boundary. |
| Full player and save copy | Not run | No video was produced. |

## Root-cause evidence

The Settings failure is a navigation-registration defect: the Video Gen summary, contract, and panel branch exist, while the Domain Defaults category tuple omits the Video Generation category.

The Console failure is a configuration-projection defect: the persisted TOML contains the requested global and ComfyUI values and the process has the intended config override, but `load_settings()` does not include the `video_generation` table in its returned settings mapping. The video-generation loader consequently uses built-in defaults with an empty enabled-backend list.

Both defects were reproduced without changing production code and were filed separately, as required by the UAT task.

## Cleanup

The isolated app session was stopped. The validated scratch directory contained no symlinks and no generated or partial video files, then was removed. TASK-3401.14 remains In Progress with the live-generation and playback criteria open.
