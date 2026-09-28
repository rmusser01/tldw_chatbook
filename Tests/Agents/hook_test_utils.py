"""Explicitly trusted execution targets for hook protocol tests only."""

from contextlib import nullcontext

from tldw_chatbook.Agents.run_hooks import HookTarget, RunHooksEngine, fingerprint_hook


def trusted_target(spec, key="one"):
    return HookTarget("protocol-test", key, fingerprint_hook(spec), spec, "test-only")


def trusted_launch_guard(target, *, tool_name=None):
    return nullcontext()


def trusted_hook_engine(config_provider, cwd_provider):
    def targets(event, tool_name):
        cfg = config_provider()
        return tuple(
            trusted_target(spec, str(index))
            for index, spec in enumerate(cfg.hooks)
            if cfg.enabled
            and spec.event == event
            and (
                spec.matcher is None
                or tool_name is not None
                and spec.matches_tool(tool_name)
            )
        )

    return RunHooksEngine(
        targets,
        cwd_provider,
        notification_targets=targets,
        launch_guard=trusted_launch_guard,
    )
