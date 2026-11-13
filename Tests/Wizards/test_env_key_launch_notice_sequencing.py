"""TASK-34100.2 AC#7 / AC#11: the env-key launch notice is honest and waits.

On a fresh profile with an exported key, the app used to raise a 10 s
"Found ANTHROPIC_API_KEY — you're ready to chat" toast at the same moment as
the "Check model lists online?" consent modal: the claim was unchecked (new
chats still used an unkeyed OpenAI) and the toast expired behind the modal.
The notice is now built from the shared readiness verdict, held while the
consent question is open, and shown once after it is answered.
"""

from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import tldw_chatbook.UI.Wizards.first_run_setup_state as state_module
from Tests.app_module_patches import set_app_global
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML


@pytest.fixture
def anthropic_only_env(monkeypatch):
    for table in DEFAULT_CONFIG_FROM_TOML["api_settings"].values():
        if isinstance(table, dict) and isinstance(table.get("api_key_env_var"), str):
            monkeypatch.delenv(table["api_key_env_var"], raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-env-launch-sequencing")
    monkeypatch.setattr(state_module, "setup_recovery_action", lambda *_: "none")
    monkeypatch.setattr(state_module, "should_show_resume_toast", lambda *_: False)


def _fake_app(events: list[str]) -> SimpleNamespace:
    fake = SimpleNamespace(
        app_config=copy.deepcopy(DEFAULT_CONFIG_FROM_TOML),
        _first_run_startup_action_scheduled=False,
    )
    fake.notify = lambda message, **kwargs: events.append(
        f"notify[{kwargs.get('title')}]: {message}"
    )
    fake._persist_env_key_notice_flag = lambda: events.append("notice-flag-saved")
    fake._show_pending_env_key_notice = lambda: TldwCli._show_pending_env_key_notice(fake)
    fake.run_worker = lambda *args, **kwargs: events.append("catalog-refresh")
    fake._refresh_model_catalogs = lambda: None
    return fake


def test_the_launch_holds_an_honest_notice_instead_of_toasting(anthropic_only_env):
    events: list[str] = []
    fake = _fake_app(events)

    assert TldwCli._maybe_offer_first_run_wizard(fake) is False

    assert events == []  # nothing raised over the consent modal
    notice = fake._pending_env_key_notice
    assert notice.ready is False
    assert notice.message.startswith(
        "Found ANTHROPIC_API_KEY, but new chats use OpenAI, which has no key."
    )
    assert "ready to chat" not in notice.message


@pytest.mark.asyncio
async def test_the_consent_decision_comes_first_then_one_notice(
    anthropic_only_env, monkeypatch
):
    events: list[str] = []
    fake = _fake_app(events)
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config",
        lambda values: events.append(f"consent-saved:{sorted(values)}") or True,
    )
    TldwCli._maybe_offer_first_run_wizard(fake)

    await TldwCli._handle_model_catalog_consent(fake, True)

    assert events[0] == "consent-saved:['model_catalog']"
    notices = [event for event in events if event.startswith("notify[")]
    assert len(notices) == 1
    assert notices[0].startswith(
        "notify[Provider key found — setup needed]: Found ANTHROPIC_API_KEY"
    )
    assert events.index(notices[0]) > 0
    assert events[-1] == "notice-flag-saved"  # recorded only once shown


@pytest.mark.asyncio
async def test_a_declined_consent_still_gets_the_notice_once(
    anthropic_only_env, monkeypatch
):
    events: list[str] = []
    fake = _fake_app(events)
    monkeypatch.setattr(
        "tldw_chatbook.config.save_settings_to_cli_config", lambda values: True
    )
    TldwCli._maybe_offer_first_run_wizard(fake)

    await TldwCli._handle_model_catalog_consent(fake, False)
    TldwCli._show_pending_env_key_notice(fake)  # a later flush is a no-op

    key_notices = [event for event in events if "ANTHROPIC_API_KEY" in event]
    assert len(key_notices) == 1
    assert events.count("notice-flag-saved") == 1


def test_a_keyed_default_reads_ready_to_chat(monkeypatch, anthropic_only_env):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-env-launch-sequencing-openai")
    events: list[str] = []
    fake = _fake_app(events)

    TldwCli._maybe_offer_first_run_wizard(fake)
    TldwCli._show_pending_env_key_notice(fake)

    assert events[0].startswith(
        "notify[Provider key detected]: Found OPENAI_API_KEY — you're ready to chat "
        "with OpenAI · "
    )


@pytest.mark.parametrize(
    ("consent_pending", "headless", "shown"),
    [
        (True, False, False),  # the consent modal is up: hold the notice
        (False, False, True),  # no modal to wait on: show it at mount
        (True, True, True),  # headless: no modal is ever shown
    ],
)
def test_the_mount_shows_the_notice_only_when_no_consent_modal_waits(
    anthropic_only_env, consent_pending, headless, shown
):
    """Review round 1 (C4/F10): the mount-time decision itself is tested --
    flushing at mount while the consent modal is pending recreates the
    stacked toast-over-modal bug, and the fake-app tests above never ran it."""
    events: list[str] = []
    fake = _fake_app(events)
    fake._startup_model_catalog_consent_required = consent_pending
    fake._schedule_startup_model_catalog_refresh = lambda: None
    fake.is_headless = headless
    TldwCli._maybe_offer_first_run_wizard(fake)

    pending = TldwCli._schedule_catalog_refresh_then_env_key_notice(fake)

    assert pending is consent_pending
    assert any(event.startswith("notify[") for event in events) is shown
    assert (fake._pending_env_key_notice is None) is shown


def test_the_mount_schedules_the_consent_question_before_it_flushes_the_notice(
    anthropic_only_env, monkeypatch
):
    """Review round 2 (R2-5): the test above presets the consent flag by hand,
    so swapping 'schedule the catalog pass' and 'flush the notice' at the
    mount would toast over the consent modal again with every test green.
    The mount now makes one call, and this drives it with the REAL scheduler
    on a profile that has never answered the consent question."""
    events: list[str] = []
    fake = _fake_app(events)
    fake.is_headless = False
    fake.call_after_refresh = MagicMock()
    fake._push_model_catalog_consent_modal = MagicMock()
    fake._schedule_startup_model_catalog_refresh = MethodType(
        TldwCli._schedule_startup_model_catalog_refresh, fake
    )
    set_app_global(monkeypatch, "load_settings", lambda: {})  # no consent on file
    TldwCli._maybe_offer_first_run_wizard(fake)

    pending = TldwCli._schedule_catalog_refresh_then_env_key_notice(fake)

    assert pending is True
    assert fake.call_after_refresh.call_args.args[0] is (
        fake._push_model_catalog_consent_modal
    )
    assert not any(event.startswith("notify[") for event in events)
    assert fake._pending_env_key_notice is not None  # held for the answer


def test_the_mount_calls_only_the_ordered_scheduler():
    """Structural pin (an AST call-site check, not behavioural evidence; its
    behavioural sibling is the real-scheduler test above).

    Review round 1 (TE-4): the tests above call the scheduler directly,
    so reverting the mount to schedule the catalog pass itself (and read the
    consent flag) left the held notice never shown on a launch without a
    pending consent modal -- with every test green. Pin the call site: the
    mount (the method that offers the first-run wizard) makes exactly one
    call to the ordered scheduler, and app.py never schedules the catalog
    pass on its own at mount (the after-setup refreshes elsewhere are
    unrelated)."""
    import ast
    import inspect

    import tldw_chatbook.app as app_module

    tree = ast.parse(inspect.getsource(app_module))
    calls: dict[str, list[str]] = {}
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.setdefault(node.func.attr, []).append(function.name)

    (mount,) = calls["_schedule_catalog_refresh_then_env_key_notice"]
    assert mount in calls["_maybe_offer_first_run_wizard"]
    assert mount not in calls.get("_schedule_startup_model_catalog_refresh", [])
    assert calls["_maybe_offer_first_run_wizard"].count(mount) == 1


from Tests.private_profile import private_profile_test  # noqa: E402


def _flatten(table: dict, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    flat: dict[tuple[str, ...], object] = {}
    for key, value in table.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, (*prefix, key)))
        else:
            flat[(*prefix, key)] = value
    return flat


@private_profile_test
async def test_an_env_key_boot_writes_only_the_notice_flag_and_the_consent_answer(
    request, monkeypatch
):
    """Pass 3 review round 1 (TE-1): AC#7's 'the only boot write is the
    one-time notice flag; no provider setting or chat default is written'
    had no direct guard -- the tests above stub the writer on a fake, so a
    boot-time write through the shared provider-setup persistence (the
    'Use <provider>' commit's own path) went unseen. This runs the real
    launch decision, consent handler, notice and flag writer against this
    test's private profile and diffs config.toml semantically."""
    import os
    import tomllib
    from pathlib import Path

    from tldw_chatbook import config as config_module

    path = Path(config_module.get_cli_config_path())
    root = os.environ.get("TLDW_TEST_CONFIG_ROOT", "")
    assert root, "only a private profile"
    profile_root = Path(root).resolve()
    assert path.resolve().is_relative_to(profile_root), "only a private profile"
    assert Path(os.environ["HOME"]).resolve().is_relative_to(profile_root)
    for table in DEFAULT_CONFIG_FROM_TOML["api_settings"].values():
        if isinstance(table, dict) and isinstance(table.get("api_key_env_var"), str):
            monkeypatch.delenv(table["api_key_env_var"], raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-env-boot-writes")
    settings = config_module.load_settings(force_reload=True)
    before = _flatten(tomllib.loads(path.read_text(encoding="utf-8")))
    events: list[str] = []
    fake = SimpleNamespace(
        app_config=settings, _first_run_startup_action_scheduled=False
    )
    fake.notify = lambda message, **kwargs: events.append(
        f"notify[{kwargs.get('title')}]: {message}"
    )

    def run_worker(work, *args, **kwargs):
        if kwargs.get("thread"):  # the flag writer: run it, for real, now
            work()
        else:
            events.append("catalog-refresh")

    fake.run_worker = run_worker
    fake._refresh_model_catalogs = lambda: None
    fake._persist_env_key_notice_flag = lambda: TldwCli._persist_env_key_notice_flag(fake)
    fake._show_pending_env_key_notice = lambda: TldwCli._show_pending_env_key_notice(fake)

    assert TldwCli._maybe_offer_first_run_wizard(fake) is False
    await TldwCli._handle_model_catalog_consent(fake, True)

    after = _flatten(tomllib.loads(path.read_text(encoding="utf-8")))
    changed = {
        key: after.get(key)
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }
    assert any("Found ANTHROPIC_API_KEY" in event for event in events), events
    assert changed == {
        ("first_run", "env_key_notice_shown"): True,
        ("model_catalog", "refresh_consent_recorded"): True,
    }
    assert "sk-ant-test-env-boot-writes" not in path.read_text(encoding="utf-8")
