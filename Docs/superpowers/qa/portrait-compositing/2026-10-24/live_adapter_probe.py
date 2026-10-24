"""Opt-in real image-adapter qualification in a disposable admitted app profile.

Inventory is the default and sends no request. Execution requires explicit consent
for at most five image-edit requests (three components, small-mouth and eyelid retry).
The existing installed journey supplies the actual editor, pair viewer and Console;
TTS transport/device remain deterministic test boundaries, not live audio evidence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import tomllib
from pathlib import Path

from tldw_chatbook.Image_Generation.capabilities import (
    resolve_backend_reference_image_capability,
)

HERE = Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    """Inventory configured capability or explicitly execute the five-request journey."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path.home() / ".config/tldw_cli/config.toml"
    )
    parser.add_argument("--output", type=Path, default=HERE / "live-adapter")
    parser.add_argument("--accept-five-image-requests", action="store_true")
    args = parser.parse_args(argv)
    section = tomllib.loads(args.config.read_text()).get("image_generation", {})
    backend = section.get("default_backend")
    enabled = section.get("enabled_backends", [])
    supported = resolve_backend_reference_image_capability(backend).supported
    report = {
        "default_backend": backend,
        "enabled_backends": enabled,
        "reference_capable_default": supported,
        "image_requests": 0,
        "live_qualified": False,
    }
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    if backend not in enabled or not supported:
        report.update(
            status="unavailable",
            reason="Configured default is not an enabled reference-image backend. No fallback selected.",
        )
    elif not args.accept_five_image_requests:
        report.update(
            status="available_not_run",
            reason="Use --accept-five-image-requests to authorize the paid live qualification.",
        )
    else:
        import toml

        from Tests.Backup_Recovery.test_home_citation_retirement import _run
        from Tests.ProductionApp.test_console_radio_portraits import _JOURNEY

        if any((args.output / name).exists() for name in ("artifacts", "journey.json")):
            parser.error(
                "Existing live evidence: choose a new --output directory before sending requests."
            )
        script = _JOURNEY
        replacements = {
            "images=QualificationImages()": "images=QualificationImages(live=True)",
            "selector.chmod(0o600)": "selector.write_text(selector.read_text()+(Path.home().parent/'image-settings.toml').read_text()); selector.chmod(0o600)",
            "install()\nfor name in ('sounddevice', 'pyaudio')": "install(); from Tests.network_guard import set_mode,NetworkMode; set_mode(NetworkMode.ALLOW_ALL)\nfor name in ('sounddevice', 'pyaudio')",
        }
        for before, after in replacements.items():
            assert script.count(before) == 1, before
            script = script.replace(before, after, 1)
        with tempfile.TemporaryDirectory(prefix="portrait-live-adapter-") as folder:
            root = Path(folder)
            settings = root / "image-settings.toml"
            settings.touch(mode=0o600)
            settings.write_text(toml.dumps({"image_generation": section}))
            try:
                _run(root, "portraits", "live", script=script, timeout=900)
                result = json.loads(
                    (root / "home/portrait-qualification.json").read_text()
                )
                assert result["real_image_adapter"] and result["image_calls"] == 5
                shutil.copytree(
                    root / "home/portrait-artifacts", args.output / "artifacts"
                )
                (args.output / "journey.json").write_text(json.dumps(result, indent=2))
                report.update(
                    status="executed_review_required",
                    image_requests=5,
                    live_adapter_exercised=True,
                    reason="Inspect retained actual viewer and Console output; structural pairs do not prove artwork semantics.",
                )
            except Exception:  # noqa: BLE001 -- never print private transport logs or retry uncertain paid requests
                artifacts = root / "home/portrait-artifacts"
                requests = artifacts / "requests.json"
                count = (
                    len(json.loads(requests.read_text())) if requests.is_file() else 0
                )
                if artifacts.is_dir() and not (args.output / "artifacts").exists():
                    shutil.copytree(artifacts, args.output / "artifacts")
                report.update(
                    status="failed_or_uncertain",
                    image_requests=count,
                    reason="Qualification did not complete; no automatic retry was made. Private transport logs were discarded.",
                )
    (args.output / "availability.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
