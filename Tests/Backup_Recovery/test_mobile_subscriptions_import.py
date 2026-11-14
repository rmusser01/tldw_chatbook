"""Cold recovery owner imports are inert; actual settings still use config."""

import json
import os
import pwd
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "args,kwargs",
    [
        (("subscriptions",), {}),
        (("subscriptions.auto_pause_after_failures", None), {}),
        (("subscriptions", "auto_pause_after_failures", None), {}),
        (
            (),
            {
                "section": "subscriptions",
                "key": "auto_pause_after_failures",
                "default": 7,
            },
        ),
    ],
)
def test_subscription_getter_forwards_exact_call(monkeypatch, args, kwargs):
    from tldw_chatbook import config
    from tldw_chatbook.DB import Subscriptions_DB as subscriptions

    observed = []
    answer = object()

    def actual(*received, **keywords):
        observed.append((received, keywords))
        return answer

    monkeypatch.setattr(config, "get_cli_setting", actual)
    assert subscriptions.get_cli_setting(*args, **kwargs) is answer
    assert observed == [(args, kwargs)]


def test_subscription_getter_preserves_native_exception(monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.DB import Subscriptions_DB as subscriptions

    failure = RuntimeError("native-config-refusal")

    def actual(*args, **kwargs):
        raise failure

    monkeypatch.setattr(config, "get_cli_setting", actual)
    with pytest.raises(RuntimeError) as caught:
        subscriptions.get_cli_setting("subscriptions", "auto_pause_after_failures", 10)
    assert caught.value is failure


def cold_native_owner(result_path):
    """Standalone fresh-process proof; no pytest conftest or home override."""
    from Tests import network_guard, real_profile_guard

    entry = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
    uid_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert entry == {"HOME": str(uid_home), "USERPROFILE": None}
    real_profile_guard.install()
    network_guard.install()
    assert Path(os.environ["TLDW_TEST_REAL_HOME"]) == uid_home
    assert (
        str(uid_home / ".config/tldw_cli").lower()
        in real_profile_guard.protected_roots()
    )
    assert "tldw_chatbook.config" not in sys.modules
    root = Path(
        tempfile.mkdtemp(
            prefix="native-subscriptions-", dir=Path(os.environ["TMPDIR"]).resolve()
        )
    ).resolve()
    selector = root / "config.toml"
    selector.write_text('[general]\nusers_name="shared"\n')
    selector.chmod(0o600)
    os.environ["TLDW_CONFIG_PATH"] = str(selector)
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    bootstrap.default_bootstrap_root = lambda: root / "bootstrap"
    report = {
        "scope": "standalone cold native class/schema owner",
        "root": str(root),
        "entry_home": entry,
    }
    owner = None

    def refuse_config(event, args):
        if event == "import" and args[0] == "tldw_chatbook.config":
            raise RuntimeError("cold_owner_imported_configuration")

    sys.addaudithook(refuse_config)
    try:
        from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

        assert "tldw_chatbook.config" not in sys.modules and not storage._startups
        owner = SubscriptionsDB(root / "subscriptions.db", "cold-native-subscriptions")
        assert owner.db_path == root / "subscriptions.db"
        assert "tldw_chatbook.config" not in sys.modules and not storage._startups
        # An explicit threshold performs native SQL without settings use.
        identity = owner.add_subscription(
            name="Native fixture",
            type="rss",
            source="https://example.invalid/feed",
            auto_pause_threshold=7,
        )
        assert owner.get_subscription(identity)["auto_pause_threshold"] == 7
        owner.close()
        owner = None
        assert (
            not storage._live_leases
            and not storage._operations
            and not storage._raw_operations
        )
        assert (
            not storage._holds and not storage._retiring_holds and not storage._startups
        )
        report["completed"] = True
    finally:
        if owner is not None:
            owner.close()
        report.update(
            profile_guard=real_profile_guard.take_violations(),
            network_guard=network_guard.blocked_attempts(),
            exit_home={key: os.environ.get(key) for key in entry},
            config_loaded="tldw_chatbook.config" in sys.modules,
            startups=len(storage._startups),
        )
        result_path.write_text(json.dumps(report, indent=2))
    assert report["entry_home"] == report["exit_home"]
    assert not report["profile_guard"] and not report["network_guard"]


if __name__ == "__main__":
    result = Path(sys.argv[1])
    assert result.is_absolute() and result.parent.is_dir() and not result.exists()
    cold_native_owner(result)
