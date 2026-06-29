#!/usr/bin/env python3
"""Run a small Workbench route-switch responsiveness soak."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Iterable
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Tests.UI.test_screen_navigation import _build_test_app  # noqa: E402
from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen  # noqa: E402
from tldw_chatbook.Utils.ui_responsiveness import (  # noqa: E402
    UIResponsivenessMonitor,
)
from tldw_chatbook.Utils.ui_responsiveness_artifacts import (  # noqa: E402
    REQUIRED_RESPONSIVENESS_ARTIFACTS,
    write_responsiveness_artifacts,
)


FOCUS_TARGETS = {
    "console-left-rail",
    "console-transcript-surface",
    "console-right-rail",
    "console-native-composer",
    "workbench-action-settings",
}


def _query_matches(root: object, selector: str) -> list[object]:
    candidates = [root, getattr(root, "screen", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            matches = list(candidate.query(selector))  # type: ignore[attr-defined]
        except Exception:
            continue
        if matches:
            return matches
    return []


def _query_one(root: object, selector: str) -> object:
    matches = _query_matches(root, selector)
    if not matches:
        raise LookupError(selector)
    return matches[0]


async def _wait_for_selector(root: object, pilot: object, selector: str, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if _query_matches(root, selector):
            return
        await pilot.pause(0.1)
    raise RuntimeError(f"Timed out waiting for selector: {selector}")


async def _click_if_present(screen: object, pilot: object, selectors: Iterable[str]) -> None:
    for selector in selectors:
        try:
            if _query_matches(screen, selector):
                await pilot.click(selector)
                await pilot.pause(0.05)
        except Exception:
            continue


async def _run_soak(output_dir: Path, *, route_switches: int, idle_seconds: float) -> int:
    app = _build_test_app(configured_default="chat")
    if getattr(app, "ui_responsiveness_monitor", None) is None:
        app.ui_responsiveness_monitor = UIResponsivenessMonitor(enabled=True)

    route_failures = 0
    focus_failures = 0

    def fake_cli_setting(section: str, key: str | None = None, default: object = None) -> object:
        if (section, key) == ("splash_screen", "enabled"):
            return False
        if (section, key) == ("general", "default_tab"):
            return "chat"
        return default

    with (
        patch("tldw_chatbook.app.get_cli_setting", side_effect=fake_cli_setting),
        patch("tldw_chatbook.UI.Screens.chat_screen.save_setting_to_cli_config", return_value=True),
    ):
        async with app.run_test(size=(140, 42)) as pilot:
            await _wait_for_selector(app, pilot, "#console-shell")
            before_workers = app.ui_responsiveness_monitor.snapshot().active_workers

            for index in range(route_switches):
                target = ("library", "settings", "chat")[index % 3]
                try:
                    await app.handle_screen_navigation(NavigateToScreen(target))
                    await pilot.pause(0.05)
                except Exception:
                    route_failures += 1

            await app.handle_screen_navigation(NavigateToScreen("chat"))
            await _wait_for_selector(app, pilot, "#console-shell")
            await _wait_for_selector(app, pilot, "#console-native-composer")

            try:
                _query_one(app, "#console-native-composer").focus()  # type: ignore[attr-defined]
                await pilot.pause(0.05)
            except Exception:
                focus_failures += 1

            focused_id = getattr(getattr(app, "focused", None), "id", None)
            if focused_id is None:
                focused_id = getattr(getattr(app.screen, "focused", None), "id", None)
            if focused_id not in FOCUS_TARGETS:
                focus_failures += 1

            await _click_if_present(
                app,
                pilot,
                (
                    "#console-context-rail-collapse",
                    "#console-context-rail-open",
                    "#console-inspector-rail-collapse",
                    "#console-inspector-rail-open",
                ),
            )

            await pilot.pause(max(0.0, idle_seconds))
            snapshot = app.ui_responsiveness_monitor.snapshot()
            after_workers = snapshot.active_workers

    route_summary = (
        f"route switches: {route_switches}, failures: {route_failures}, "
        f"focus failures: {focus_failures}, workers before: {before_workers}, "
        f"workers after: {after_workers}"
    )
    write_responsiveness_artifacts(output_dir, snapshot, route_switch_summary=route_summary)

    missing_artifacts = [
        filename for filename in REQUIRED_RESPONSIVENESS_ARTIFACTS if not (output_dir / filename).exists()
    ]
    if route_failures or focus_failures or missing_artifacts or snapshot.stalled or after_workers > before_workers:
        details = [
            route_summary,
            f"stalled: {snapshot.stalled}",
            f"missing artifacts: {', '.join(missing_artifacts) if missing_artifacts else 'none'}",
        ]
        print("\n".join(details), file=sys.stderr)
        return 1

    print(route_summary)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Docs/superpowers/qa/chatbook-workbench-ui-foundation-console/artifacts"),
    )
    parser.add_argument("--route-switches", type=int, default=6)
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    args = parser.parse_args()
    return asyncio.run(
        _run_soak(
            args.output,
            route_switches=max(1, args.route_switches),
            idle_seconds=max(0.0, args.idle_seconds),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
