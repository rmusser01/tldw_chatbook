"""No-device ownership, identity and monotonic lease contracts."""

import importlib
import importlib.util
from dataclasses import replace

import pytest


def lifetime():
    name = "tldw_chatbook.Audio.voice_process_lifetime"
    assert importlib.util.find_spec(name), "process lifetime implementation missing"
    return importlib.import_module(name)


def test_lease_expiry_is_terminal_even_when_renewal_arrives_late():
    module = lifetime()
    now = [10.0]
    lease = module.ChildLease(clock=lambda: now[0])
    now[0] = 14.999
    assert lease.alive
    now[0] = 15.0
    assert not lease.renew()
    now[0] = 14.0
    assert not lease.alive


def test_live_lease_renewal_extends_only_from_current_monotonic_time():
    module = lifetime()
    now = [0.0]
    lease = module.ChildLease(clock=lambda: now[0])
    now[0] = 4.0
    assert lease.renew()
    now[0] = 8.99
    assert lease.alive
    now[0] = 9.0
    assert not lease.alive


@pytest.mark.parametrize(
    "field,value",
    [("root", "/wrong"), ("source", "0" * 64), ("native_abi", 999), ("version", 99)],
)
def test_identity_mismatch_cannot_advance_handshake(field, value):
    module = lifetime()
    identity = module.source_identity()
    gate = module.StartupGate(identity, clock=lambda: 0)
    with pytest.raises(module.VoiceProcessError):
        gate.hello(replace(identity, **{field: value}))
    assert not gate.permitted


def test_handshake_deadline_does_not_become_stt_deadline():
    module = lifetime()
    now = [0.0]
    identity = module.source_identity()
    gate = module.StartupGate(identity, clock=lambda: now[0])
    now[0] = 5.0
    with pytest.raises(module.VoiceProcessError):
        gate.hello(identity)
    gate = module.StartupGate(identity, clock=lambda: now[0])
    gate.hello(identity)
    now[0] = 124.99
    gate.stt_ready()
    gate.permit(current=True)
    assert gate.permitted


def test_no_permit_before_readiness_or_after_stale_activation():
    module = lifetime()
    identity = module.source_identity()
    gate = module.StartupGate(identity, clock=lambda: 0)
    gate.hello(identity)
    with pytest.raises(module.VoiceProcessError):
        gate.permit(current=True)
    with pytest.raises(module.VoiceProcessError):
        gate.stt_ready()
    gate = module.StartupGate(identity, clock=lambda: 0)
    gate.hello(identity)
    gate.stt_ready()
    with pytest.raises(module.VoiceProcessError):
        gate.permit(current=False)
    assert not gate.permitted


def test_stt_deadline_is_terminal_after_120_seconds():
    module = lifetime()
    now = [0.0]
    identity = module.source_identity()
    gate = module.StartupGate(identity, clock=lambda: now[0])
    gate.hello(identity)
    now[0] = 120.0
    with pytest.raises(module.VoiceProcessError):
        gate.stt_ready()
    now[0] = 1.0
    with pytest.raises(module.VoiceProcessError):
        gate.stt_ready()


class FakeJobApi:
    def __init__(self, *, fail_assignment=False):
        self.fail_assignment = fail_assignment
        self.active = 2
        self.events = []

    def create_kill_on_close_job(self):
        self.events.append("created")
        return 123

    def assign_process(self, handle, pid):
        assert (handle, pid) == (123, 456)
        self.events.append("assigned")
        if self.fail_assignment:
            raise OSError("private detail")

    def wait_for_job_empty(self, handle, timeout):
        assert handle == 123
        return self.active == 0

    def terminate_job(self, handle):
        assert handle == 123
        self.events.append("terminated")

    def close_handle(self, handle):
        assert handle == 123
        self.events.append("closed")


def test_windows_retains_job_until_descendants_absent():
    module = lifetime()
    api = FakeJobApi()
    tree = module.OwnedProcessTree(456, platform="nt", windows_api=api)
    tree.attach()
    assert api.events == ["created", "assigned"]
    assert not tree.observe_absence()
    tree.terminate()
    assert not tree.observe_absence()
    assert "closed" not in api.events
    api.active = 0
    assert tree.observe_absence()
    assert api.events == ["created", "assigned", "terminated", "closed"]


def test_windows_failed_assignment_fails_closed():
    module = lifetime()
    api = FakeJobApi(fail_assignment=True)
    tree = module.OwnedProcessTree(456, platform="nt", windows_api=api)
    with pytest.raises(module.VoiceProcessError):
        tree.attach()
    assert api.events == ["created", "assigned", "closed"]
    assert not tree.attached


def test_child_environment_excludes_credentials_and_configuration(monkeypatch):
    module = lifetime()
    monkeypatch.setenv("OPENAI_API_KEY", "private-key")
    monkeypatch.setenv("TLDW_CONFIG_PATH", "/private/config")
    monkeypatch.setenv("PYTHONPATH", "/injected/import")
    monkeypatch.setenv("HF_HOME", "/local/model-cache")
    env = module.child_environment()
    assert not {"OPENAI_API_KEY", "TLDW_CONFIG_PATH", "PYTHONPATH"} & env.keys()
    assert env["HF_HOME"] == "/local/model-cache"


def test_uncertain_group_observation_retains_ownership(monkeypatch):
    module = lifetime()
    monkeypatch.setattr(module.os, "getpgid", lambda pid: pid)
    tree = module.OwnedProcessTree(456, platform="posix")
    tree.attach()

    def denied(*args):
        raise PermissionError("private OS detail")

    monkeypatch.setattr(module.os, "killpg", denied)
    assert not tree.observe_absence()
    tree.terminate(force=True)
    assert not tree.absent
