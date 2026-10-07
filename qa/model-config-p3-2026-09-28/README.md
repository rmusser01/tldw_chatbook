# Phase 3 (TASK-33003) captures

These are evidence captures for the Phase 3 model-configuration PR (#2937). Each task's folder holds its captures; `final/` holds the re-capture at the rebased final head.

- **Live captures** (`live-*`, `base-*`, `head-*`): the real app ran in tmux at 211x44 and 235x52 (plus 140x42 and 120x40 where a task names them). Each run used a scratch profile: HOME, XDG and `TLDW_CONFIG_PATH` were scratch, with a throwaway `users_name`, a null keyring and the splash off. `.txt` is `tmux capture-pane -p` (text only). `.ansi.txt` is `capture-pane -p -e`; it carries the colours that the contrast figures were measured from.
- **Harness captures** (`task-1/*.before.txt` / `*.after.txt` and `task-1/composers/`): a test's app was forced to 211x44 and its screen was dumped at exit. A pair is byte-identical when the task records that surface as unchanged (TASK-33003.1 AC#3). A composer that is wider than the harness screen is cut at column 211. That cut is not a missing border.
- **Drivers are not kept.** The `.sh` launch and drive scripts, the probe and measuring scripts, and the pytest probe plugins were local one-offs. They had machine-specific paths, and the review of #2937 removed them. The task notes still name some of them to say how a capture was made.
- **Removed as evidence errors** (review of #2937): the four `*-120x40-*-scrolled-end` captures in `task-7/` were byte-identical to their at-rest twins, because the wheel never reached the overflowing pane. The `task-2/focus-{before,after}` crops mixed widgets from two different views.
