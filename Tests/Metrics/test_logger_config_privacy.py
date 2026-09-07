from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_legacy_metrics_setup_cannot_create_unfiltered_file_sinks(
    tmp_path: Path,
) -> None:
    application_log = tmp_path / "legacy-application.log"
    metrics_log = tmp_path / "legacy-metrics.json"
    script = (
        "from tldw_chatbook.Metrics.logger_config import setup_logger\n"
        "from loguru import logger\n"
        f"setup_logger(app_log_path={str(application_log)!r}, "
        f"metrics_log_path={str(metrics_log)!r})\n"
        "logger.info('METRICS-PRIVATE-SENTINEL-sk-not-a-real-key')\n"
        "logger.complete()\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert not application_log.exists()
    assert not metrics_log.exists()
