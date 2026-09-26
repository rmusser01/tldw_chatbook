"""Console project-instruction control-state and identity contracts."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from tldw_chatbook.Chat.console_project_instructions import (
    EPHEMERAL_ORIGIN_KEY,
    LOCATOR_FINGERPRINT_DOMAIN,
    NOTICE_KEY_FINGERPRINT_DOMAIN,
    PROJECT_CONTEXT_VERSION,
    PROVIDER_DESTINATION_FINGERPRINT_DOMAIN,
    ProjectInstructionControlState,
    decode_project_context_json,
    encode_project_context_json,
    fingerprint_canonical_locator,
    fingerprint_provider_destination,
    project_instruction_notice_key,
    sanitized_destination_label,
)


def test_new_session_explicitly_enables_project_instructions() -> None:
    assert EPHEMERAL_ORIGIN_KEY == "_chatbook_ephemeral_origin"
    assert ProjectInstructionControlState.new_session() == (
        ProjectInstructionControlState(project_instructions_enabled=True)
    )


@pytest.mark.parametrize(
    "raw_state",
    [
        None,
        "",
        "not-json",
        "null",
        "[]",
        "{}",
        '{"version": 2}',
        '{"version": true}',
        '{"version": 1}',
        json.dumps(
            {
                "version": 1,
                "project_instructions_enabled": 1,
                "working_folder_binding_id": None,
                "working_folder_locator_fingerprint": None,
                "project_instruction_notice_key": None,
            }
        ),
        json.dumps(
            {
                "version": 1,
                "project_instructions_enabled": True,
                "working_folder_binding_id": 12,
                "working_folder_locator_fingerprint": None,
                "project_instruction_notice_key": None,
            }
        ),
    ],
)
def test_untrusted_or_legacy_state_fails_closed(raw_state: str | None) -> None:
    assert decode_project_context_json(raw_state) == (
        ProjectInstructionControlState.legacy_disabled()
    )


@pytest.mark.parametrize(
    ("duplicate_key", "duplicate_value"),
    [
        ("version", 1),
        ("project_instructions_enabled", True),
        ("working_folder_binding_id", "binding-7"),
        ("working_folder_locator_fingerprint", "locator-fingerprint"),
        ("project_instruction_notice_key", "notice-key"),
    ],
)
def test_duplicate_json_keys_fail_closed(
    duplicate_key: str, duplicate_value: object
) -> None:
    pairs = [
        ("version", 1),
        ("project_instructions_enabled", True),
        ("working_folder_binding_id", "binding-7"),
        ("working_folder_locator_fingerprint", "locator-fingerprint"),
        ("project_instruction_notice_key", "notice-key"),
        (duplicate_key, duplicate_value),
    ]
    raw_state = (
        "{"
        + ",".join(f"{json.dumps(key)}:{json.dumps(value)}" for key, value in pairs)
        + "}"
    )

    assert decode_project_context_json(raw_state) == (
        ProjectInstructionControlState.legacy_disabled()
    )


def test_control_state_round_trips_only_the_version_and_four_control_fields() -> None:
    state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-7",
        working_folder_locator_fingerprint="locator-fingerprint",
        project_instruction_notice_key="notice-key",
    )

    encoded = encode_project_context_json(state)

    assert json.loads(encoded) == {
        "version": PROJECT_CONTEXT_VERSION,
        "project_instructions_enabled": True,
        "working_folder_binding_id": "binding-7",
        "working_folder_locator_fingerprint": "locator-fingerprint",
        "project_instruction_notice_key": "notice-key",
    }
    assert decode_project_context_json(encoded) == state


def test_unknown_or_sensitive_fields_are_not_preserved_on_reencode() -> None:
    raw_values = {
        "version": 1,
        "project_instructions_enabled": True,
        "working_folder_binding_id": "binding-7",
        "working_folder_locator_fingerprint": "opaque-locator-fingerprint",
        "project_instruction_notice_key": "opaque-notice-key",
        "locator": "file:///Users/alice/private/repo",
        "source_path": "secret/AGENTS.md",
        "digest": "raw-instruction-digest",
        "endpoint": "https://user:password@example.test/private/v1",
        "body": "private instruction body",
    }

    decoded = decode_project_context_json(json.dumps(raw_values))
    reencoded = encode_project_context_json(decoded)

    assert decoded == ProjectInstructionControlState.legacy_disabled()
    assert set(json.loads(reencoded)) == {
        "version",
        "project_instructions_enabled",
        "working_folder_binding_id",
        "working_folder_locator_fingerprint",
        "project_instruction_notice_key",
    }
    for sensitive_value in (
        raw_values["locator"],
        raw_values["source_path"],
        raw_values["digest"],
        raw_values["endpoint"],
        raw_values["body"],
    ):
        assert sensitive_value not in reencoded


def test_fingerprint_protocol_domains_and_outputs_are_pinned() -> None:
    assert LOCATOR_FINGERPRINT_DOMAIN == (
        b"tldw_chatbook.console.project-instructions.locator.v1\0"
    )
    assert PROVIDER_DESTINATION_FINGERPRINT_DOMAIN == (
        b"tldw_chatbook.console.project-instructions.provider-destination.v1\0"
    )
    assert NOTICE_KEY_FINGERPRINT_DOMAIN == (
        b"tldw_chatbook.console.project-instructions.notice-key.v1\0"
    )
    locator_fingerprint = fingerprint_canonical_locator("file:///Users/alice/work/repo")
    assert locator_fingerprint == (
        "221fa1a5f342123e6dda9f409b35146dacfaf96978c64832130f0df73ecf7c1b"
    )
    destination_fingerprint = fingerprint_provider_destination(
        "OpenAI",
        "HTTPS://user:secret@API.Example.COM:443/v1/?api_key=secret#fragment",
    )
    assert destination_fingerprint == (
        "a7fd7f4ef42fa29cc07e2e712cce210bc427ea5e293384933c8feb96d13030b4"
    )
    assert (
        fingerprint_provider_destination("openai", "https://api.example.com/v1")
        == destination_fingerprint
    )
    assert (
        project_instruction_notice_key(
            locator_fingerprint,
            "OpenAI",
            "HTTPS://user:secret@API.Example.COM:443/v1/?api_key=secret#fragment",
        )
        == "ef9c37589f7117d0647e6fb350448e68725900920664d9425b9700d156f07e1c"
    )


def test_notice_key_tracks_provider_destination_but_not_model() -> None:
    locator_fingerprint = fingerprint_canonical_locator("file:///repo")
    endpoint = "https://api.example.test/v1"

    baseline = project_instruction_notice_key(locator_fingerprint, "openai", endpoint)
    assert "model" not in inspect.signature(project_instruction_notice_key).parameters
    assert project_instruction_notice_key(
        locator_fingerprint, "openai", endpoint
    ) == project_instruction_notice_key(locator_fingerprint, "openai", endpoint)
    assert (
        project_instruction_notice_key(locator_fingerprint, "anthropic", endpoint)
        != baseline
    )
    assert (
        project_instruction_notice_key(
            locator_fingerprint, "openai", "https://other.example.test/v1"
        )
        != baseline
    )
    assert (
        project_instruction_notice_key(
            locator_fingerprint, "openai", "https://api.example.test/v2"
        )
        != baseline
    )


def test_destination_label_shows_only_provider_and_custom_endpoint_origin() -> None:
    raw_endpoint = (
        "https://user:password@API.Example.test:8443/private/v1?api_key=secret#fragment"
    )

    label = sanitized_destination_label("OpenAI", raw_endpoint)

    assert label == "OpenAI (https://api.example.test:8443)"
    for secret_or_path in ("user", "password", "private", "api_key", "secret"):
        assert secret_or_path not in label
    assert sanitized_destination_label("OpenAI", None) == "OpenAI"


@pytest.mark.parametrize("control_character", ["\0", "\x1f", "\x7f"])
def test_destination_label_rejects_control_characters(
    control_character: str,
) -> None:
    endpoint = f"https://api.example{control_character}.test/private"

    assert sanitized_destination_label("OpenAI", endpoint) == (
        "OpenAI (invalid endpoint)"
    )


@pytest.mark.parametrize(
    "unsafe_endpoint",
    [
        "https://api.example.test\\@evil.test/private",
        "https://api.example\u202e.test/private",
        "https://api.example\u2066.test/private",
        "https://api.example\ue000.test/private",
        "https://api.example.test/private\n",
    ],
)
def test_destination_label_rejects_backslashes_and_bidi_controls(
    unsafe_endpoint: str,
) -> None:
    assert sanitized_destination_label("OpenAI", unsafe_endpoint) == (
        "OpenAI (invalid endpoint)"
    )
    with pytest.raises(ValueError, match="invalid provider endpoint"):
        fingerprint_provider_destination("openai", unsafe_endpoint)


@pytest.mark.parametrize(
    ("unicode_host", "punycode_host"),
    [
        ("münich.example", "xn--mnich-kva.example"),
        ("faß.de", "xn--fa-hia.de"),
    ],
)
def test_unicode_and_punycode_hosts_share_one_destination_identity(
    unicode_host: str,
    punycode_host: str,
) -> None:
    unicode_endpoint = f"https://{unicode_host}/v1"
    punycode_endpoint = f"https://{punycode_host}/v1"

    assert fingerprint_provider_destination(
        "openai", unicode_endpoint
    ) == fingerprint_provider_destination("openai", punycode_endpoint)
    assert sanitized_destination_label("OpenAI", unicode_endpoint) == (
        f"OpenAI (https://{punycode_host})"
    )
    assert sanitized_destination_label("OpenAI", punycode_endpoint) == (
        f"OpenAI (https://{punycode_host})"
    )


@pytest.mark.parametrize(
    ("endpoint", "expected_label"),
    [
        ("https://127.0.0.1:8443/v1", "OpenAI (https://127.0.0.1:8443)"),
        (
            "https://[2001:db8::1]:8443/v1",
            "OpenAI (https://[2001:db8::1]:8443)",
        ),
    ],
)
def test_destination_label_preserves_ip_address_hosts(
    endpoint: str, expected_label: str
) -> None:
    assert sanitized_destination_label("OpenAI", endpoint) == expected_label


def test_invalid_endpoints_cannot_produce_reusable_notice_keys() -> None:
    locator_fingerprint = fingerprint_canonical_locator("file:///repo")

    for malformed_endpoint in ("not a valid endpoint", "https://[broken"):
        with pytest.raises(ValueError, match="invalid provider endpoint"):
            project_instruction_notice_key(
                locator_fingerprint,
                "openai",
                malformed_endpoint,
            )


def test_blank_endpoint_is_the_valid_default_provider_identity() -> None:
    assert fingerprint_provider_destination(
        "openai", None
    ) == fingerprint_provider_destination("openai", "  ")
    assert sanitized_destination_label("OpenAI", "  ") == "OpenAI"


# ---------------------------------------------------------------------------
# Phase 3c (task 17): remote (ssh-filesystem) admission, exclusions, guards
# ---------------------------------------------------------------------------


def _ssh_binding(
    binding_id: str = "ssh-b1",
    *,
    locator: str = "ssh://devbox/srv/www",
    exclusions: tuple[str, ...] = ("secrets", "build/out"),
):
    """One registry row for an ssh-filesystem binding (fake registry shape)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        workspace_id="w1",
        binding_id=binding_id,
        binding_kind="ssh-filesystem",
        label=binding_id,
        locator=locator,
        status="ready",
        metadata={
            "access": "rw",
            "exclusions": [
                {"path": path, "kind": "directory", "added_at": ""}
                for path in exclusions
            ],
        },
    )


def _session_for_remote():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )

    return SimpleNamespace(
        workspace_id="w1",
        project_instruction_state=ProjectInstructionControlState.new_session(),
    )


def _ready_cache_with_identity():
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    cache.record_success(
        "ssh-b1",
        [["/srv/www", 99, 7, 16877], ["/srv", 98, 6, 16877]],
    )
    return cache


def test_ssh_binding_admitted_when_cache_ready() -> None:
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_locator import (
        locator_string,
        parse_remote_locator,
    )
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot, is_remote

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(),
        _ssh_binding(),
        status_cache=_ready_cache_with_identity(),
    )

    assert selection is not None
    assert is_remote(selection.root)
    assert isinstance(selection.root, RemoteRoot)
    canonical = locator_string(parse_remote_locator("ssh://devbox/srv/www"))
    assert selection.root.canonical_locator == canonical
    assert selection.locator_fingerprint == fingerprint_canonical_locator(canonical)
    assert selection.allow_write is True
    assert selection.root_identity == (
        ("/srv/www", 99, 7, 16877),
        ("/srv", 98, 6, 16877),
    )


def test_ssh_binding_missing_requires_reselection() -> None:
    """MISSING is a retarget-class failure: admission refuses (None), so
    the resolver raises binding_unavailable -> explicit re-selection
    (ADR-069 parity with a deleted local folder)."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    cache.record_missing("ssh-b1", "missing on host")

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(), _ssh_binding(), status_cache=cache
    )

    assert selection is None


def test_ssh_binding_blocked_admits_degraded_without_forcing_reselection() -> None:
    """BLOCKED is transient (flaky link, not a retarget): the selection
    still resolves -- degraded with a warning -- and never forces the
    ADR-069 recovery dialog."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_status import BindingState
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache
    from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind

    cache = RemoteBindingStatusCache()
    cache.record_transport_failure(
        "ssh-b1", TransportFailureKind.UNREACHABLE, "unreachable"
    )
    assert cache.status("ssh-b1").state is BindingState.BLOCKED

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(), _ssh_binding(), status_cache=cache
    )

    assert selection is not None
    assert selection.degraded is True


def test_ssh_binding_stale_identity_admits() -> None:
    """STALE_IDENTITY is not BLOCKED (spec): the pin re-captures on the
    next worker call, so admission keeps the binding."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = _ready_cache_with_identity()
    cache.record_pin_failure("ssh-b1")

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(), _ssh_binding(), status_cache=cache
    )

    assert selection is not None
    assert selection.degraded is False


def test_ssh_binding_cold_cache_admits_optimistically_without_ping() -> None:
    """Hot-path rule: admission reads the cache only; an empty cache is
    optimistically READY (a dead host costs one typed mid-run error)."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(), _ssh_binding(), status_cache=RemoteBindingStatusCache()
    )

    assert selection is not None
    assert selection.degraded is False


def test_ssh_binding_first_selection_ping_runs_once_when_identity_cold() -> None:
    """The ONE synchronous probe point at selection: a cold identity
    triggers the supplied ping factory exactly once; its recorded chain
    becomes the admitted identity."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = RemoteBindingStatusCache()
    calls: list[str] = []

    def ping() -> None:
        calls.append("ssh-b1")
        cache.record_success(
            "ssh-b1",
            [["/srv/www", 99, 7, 16877], ["/srv", 98, 6, 16877]],
        )

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(),
        _ssh_binding(),
        status_cache=cache,
        first_selection_ping=ping,
    )

    assert calls == ["ssh-b1"]
    assert selection is not None
    assert selection.root_identity == (
        ("/srv/www", 99, 7, 16877),
        ("/srv", 98, 6, 16877),
    )


def test_ssh_binding_first_selection_ping_not_repeated_when_identity_warm() -> None:
    cache = _ready_cache_with_identity()
    calls: list[str] = []

    def ping() -> None:
        calls.append("ssh-b1")

    from tldw_chatbook.Chat import console_chat_controller as controller

    controller._validate_project_instruction_binding(
        _session_for_remote(),
        _ssh_binding(),
        status_cache=cache,
        first_selection_ping=ping,
    )

    assert calls == []


def test_ssh_binding_reselection_survives_blocked_status_in_resolver() -> None:
    """End-to-end resolver posture: a SELECTED ssh binding with a BLOCKED
    cache resolves (degraded) instead of raising the recovery dialog."""
    from tldw_chatbook.Chat.console_chat_controller import (
        ProjectInstructionBindingRecovery,
        resolve_project_instruction_binding,
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
    )
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache
    from tldw_chatbook.Tools.remote_workspace_transport import TransportFailureKind

    class _Registry:
        def get_runtime_binding(self, binding_id):
            assert binding_id == "ssh-b1"
            return _ssh_binding()

        def list_runtime_bindings(self, workspace_id):
            return (_ssh_binding(),)

    state = ProjectInstructionControlState.new_session()
    state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="ssh-b1",
        working_folder_locator_fingerprint=fingerprint_canonical_locator(
            "ssh://devbox/srv/www"
        ),
        project_instruction_notice_key=None,
    )
    session = type("Session", (), {})()
    session.workspace_id = "w1"
    session.project_instruction_state = state
    cache = RemoteBindingStatusCache()
    cache.record_transport_failure(
        "ssh-b1", TransportFailureKind.UNREACHABLE, "unreachable"
    )

    selection = resolve_project_instruction_binding(
        session, _Registry(), status_cache=cache
    )

    assert selection is not None and selection.degraded is True
    # and MISSING still forces the recovery flow:
    cache.record_missing("ssh-b1", "missing on host")
    try:
        resolve_project_instruction_binding(session, _Registry(), status_cache=cache)
    except ProjectInstructionBindingRecovery as exc:
        assert str(exc) == "binding_unavailable"
    else:
        raise AssertionError("MISSING must force re-selection")


def test_remote_binding_authority_guard_compares_cached_identity() -> None:
    """`_workspace_binding_authority_is_current` for remote roots: registry
    read + cached identity chain (the worker pin stays the per-call
    guard); a swapped root (new chain) revokes."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cache = _ready_cache_with_identity()
    expected = controller._validate_project_instruction_binding(
        _session_for_remote(), _ssh_binding(), status_cache=cache
    )
    assert expected is not None

    class _Registry:
        def get_runtime_binding(self, binding_id):
            return _ssh_binding()

    assert (
        controller._workspace_binding_authority_is_current(
            workspace_id="w1",
            registry=_Registry(),
            expected_selection=expected,
            write=True,
            status_cache=cache,
        )
        is True
    )

    swapped = RemoteBindingStatusCache()
    swapped.record_success("ssh-b1", [["/srv/www", 111, 222, 16877]])
    assert (
        controller._workspace_binding_authority_is_current(
            workspace_id="w1",
            registry=_Registry(),
            expected_selection=expected,
            write=True,
            status_cache=swapped,
        )
        is False
    )


def test_remote_exclusion_paths_provider_fails_closed_on_shrink() -> None:
    """Live-read union keeps the high-water mark: a provider exception or
    registry loss can never NARROW the effective exclusion set mid-run."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

    class _Registry:
        def get_runtime_binding(self, binding_id):
            raise RuntimeError("registry exploded")

    provider = controller._exclusion_paths_provider(
        _Registry(),
        "ssh-b1",
        _remote_root_descriptor(),
        ("secrets",),
    )
    # First read fails but keeps the snapshot; second read stays at the
    # high-water mark.
    provider()
    assert provider() == (SensitiveExclusion("subtree", "secrets"),)


def test_remote_project_instruction_excluded_dirs_are_relative_and_unresolved(
    monkeypatch,
) -> None:
    from pathlib import Path as _Path

    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot

    def blow_up(self, *_args, **_kwargs):
        raise AssertionError("remote excluded dirs must never resolve on the laptop")

    monkeypatch.setattr(_Path, "resolve", blow_up)

    selection = type("Selection", (), {})()
    selection.root = RemoteRoot(
        alias="ssh-b1",
        canonical_locator="ssh://devbox/srv/www",
        root="/srv/www",
        binding_id="ssh-b1",
    )
    selection.binding = _ssh_binding()

    excluded = controller._project_instruction_excluded_dirs(selection)

    assert excluded == frozenset({_Path("secrets"), _Path("build/out")})


# ---------------------------------------------------------------------------
# Review fix (Important 2): cold-admission vs authority-guard consistency.
# The controller's call sites pass no cache yet (Task 18 wires it), so a
# cold admission carries an EMPTY identity chain -- the client-side guard
# must treat that as CURRENT (registry + fingerprint + status only; the
# WORKER's root pin is the per-call guard per the spec's split-authority
# rule). Local roots keep the exact opposite: an empty identity chain
# FAILS the local guard.
# ---------------------------------------------------------------------------


def test_remote_cold_authority_guard_passes_without_cache_or_identity() -> None:
    from tldw_chatbook.Chat import console_chat_controller as controller

    class _Registry:
        def get_runtime_binding(self, binding_id):
            return _ssh_binding()

    cold = controller._validate_project_instruction_binding(
        _session_for_remote(), _ssh_binding()
    )
    assert cold is not None
    assert cold.root_identity == ()  # cold: no cache, no captured chain

    assert (
        controller._workspace_binding_authority_is_current(
            workspace_id="w1",
            registry=_Registry(),
            expected_selection=cold,
            write=True,
        )
        is True
    )


def test_remote_guard_never_fails_on_identity_captured_after_admission() -> None:
    """Identity captured AFTER a cold admission (first worker call) must
    not trip the guard: missing identity on either side is not a
    mismatch; only two PRESENT-but-different chains revoke."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_binding_status import RemoteBindingStatusCache

    cold = controller._validate_project_instruction_binding(
        _session_for_remote(), _ssh_binding()
    )
    assert cold is not None and cold.root_identity == ()

    cache = RemoteBindingStatusCache()  # identity appears later
    cache.record_success("ssh-b1", [["/srv/www", 99, 7, 16877]])

    class _Registry:
        def get_runtime_binding(self, binding_id):
            return _ssh_binding()

    assert (
        controller._workspace_binding_authority_is_current(
            workspace_id="w1",
            registry=_Registry(),
            expected_selection=cold,
            write=True,
            status_cache=cache,
        )
        is True
    )


def test_local_guard_still_fails_on_empty_identity(tmp_path) -> None:
    """Local roots are unchanged: the client guard re-captures the laptop
    identity chain per call, and an empty expected chain can never match
    a real capture -- it stays a failure."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat import console_chat_controller as controller

    project = tmp_path / "project"
    project.mkdir()
    binding = SimpleNamespace(
        workspace_id="w1",
        binding_id="local-b1",
        binding_kind="local-filesystem",
        label="local-b1",
        locator=str(project),
        status="ready",
        metadata={"access": "rw"},
    )
    broken = SimpleNamespace(
        binding=binding,
        root=project.resolve(),
        locator_fingerprint="fingerprint-local",
        allow_write=True,
        root_identity=(),  # empty for a LOCAL root: must fail the guard
    )

    class _Registry:
        def get_runtime_binding(self, binding_id):
            return binding

    assert (
        controller._workspace_binding_authority_is_current(
            workspace_id="w1",
            registry=_Registry(),
            expected_selection=broken,
            write=True,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Review fix (Important 3): the remote exclusion provider must yield
# properly typed SensitiveExclusion entries -- both executor seams read
# ``entry.kind``/``entry.value`` (raw Paths raised AttributeError and
# fail-closed-refused every op). Pinned by consuming the provider's
# output through BOTH seams directly.
# ---------------------------------------------------------------------------


def _remote_root_descriptor():
    from tldw_chatbook.Tools.remote_root_types import RemoteRoot

    return RemoteRoot(
        alias="ssh-b1",
        canonical_locator="ssh://devbox/srv/www",
        root="/srv/www",
        binding_id="ssh-b1",
    )


def _ssh_registry(exclusions=("secrets", "build/out")):
    class _Registry:
        def get_runtime_binding(self, binding_id):
            return _ssh_binding(exclusions=exclusions)

    return _Registry()


def test_remote_exclusion_provider_yields_typed_entries(monkeypatch) -> None:
    from pathlib import Path as _Path

    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

    def blow_up(self, *_args, **_kwargs):
        raise AssertionError("remote exclusions must never resolve on the laptop")

    monkeypatch.setattr(_Path, "resolve", blow_up)

    provider = controller._exclusion_paths_provider(
        _ssh_registry(),
        "ssh-b1",
        _remote_root_descriptor(),
        ("secrets", "build/out"),
    )

    assert provider() == (
        SensitiveExclusion("subtree", "build/out"),
        SensitiveExclusion("subtree", "secrets"),
    )


def test_remote_exclusion_provider_maps_stored_file_kind() -> None:
    """A binding exclusion stored with kind "file" serializes as the
    narrower 'file' exclusion; unknown/snapshot entries stay 'subtree'
    (the fail-closed superset)."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

    row = _ssh_binding(exclusions=("secrets", "env.lock"))
    row.metadata["exclusions"] = [
        {"path": "secrets", "kind": "directory", "added_at": ""},
        {"path": "env.lock", "kind": "file", "added_at": ""},
    ]

    class _RegistryWithRow:
        def get_runtime_binding(self, binding_id):
            return row

    provider = controller._exclusion_paths_provider(
        _RegistryWithRow(),
        "ssh-b1",
        _remote_root_descriptor(),
        ("secrets", "env.lock"),
    )

    # sorted by rel: env.lock < secrets
    assert provider() == (
        SensitiveExclusion("file", "env.lock"),
        SensitiveExclusion("subtree", "secrets"),
    )


def test_remote_exclusion_provider_output_consumable_by_both_executor_seams(
    tmp_path,
) -> None:
    """The provider's entries feed BOTH exclusion consumers unchanged:
    the remote-mode ``WorkspaceToolExecutor`` request builder and the
    ssh/loopback transport executor's injection seam."""
    from tldw_chatbook.Chat import console_chat_controller as controller
    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceExecutionError,
        RemoteWorkspaceToolExecutor,
    )
    from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor

    provider = controller._exclusion_paths_provider(
        _ssh_registry(exclusions=("secrets",)),
        "ssh-b1",
        _remote_root_descriptor(),
        ("secrets",),
    )

    # Seam 1: remote-mode WorkspaceToolExecutor request builder.
    builder = WorkspaceToolExecutor(
        _remote_root_descriptor(),
        remote_identity_source=lambda: [["/srv/www", 99, 7, 16877]],
        remote_sensitive_exclusions=provider,
    )
    request = builder._build_request("fs_read", {"path": "ok.txt"}, intent="read")
    assert request.arguments["sensitive_exclusions"] == [
        {"kind": "subtree", "value": "secrets"}
    ]
    excluded_request = builder._build_request(
        "fs_read", {"path": "secrets/kv"}, intent="read"
    )
    # The builder serializes the set onto EVERY request (enforcement is
    # worker-side; seam 2 proves the refusal end-to-end).
    assert excluded_request.arguments["sensitive_exclusions"] == [
        {"kind": "subtree", "value": "secrets"}
    ]

    # Seam 2: the ssh/loopback transport executor's injection.
    standin = tmp_path / "remote-fs"
    standin.mkdir()
    (standin / "ok.txt").write_text("public\n", encoding="utf-8")
    (standin / "secrets").mkdir()
    (standin / "secrets" / "kv.txt").write_text("hidden\n", encoding="utf-8")
    prober = RemoteWorkspaceToolExecutor(
        standin, root_locator=str(standin), identity_chain_source=lambda: None
    )
    payload = prober.ping()
    chain = {
        "identity_chain": payload["identity_chain"],
        "canonical_path": payload["canonical_path"],
    }
    executor = RemoteWorkspaceToolExecutor(
        standin,
        root_locator=str(standin),
        identity_chain_source=lambda: chain,
        sensitive_exclusions=provider,
    )
    ok = executor.execute("fs_read", {"path": "ok.txt"}, intent="read")
    assert ok["outcome"] == "success"
    with pytest.raises(RemoteWorkspaceExecutionError):
        executor.execute("fs_read", {"path": "secrets/kv.txt"}, intent="read")



def test_remote_instruction_io_for_selection_builds_reader_or_degrades() -> None:
    """Task 19 dispatch wiring: a remote selection yields the executor-
    backed instruction reader; construction failure degrades to ``None``
    (ADR-069 posture: content-free warning, proceed)."""
    from types import SimpleNamespace

    from tldw_chatbook.Agents.project_instruction_resolver import RemoteInstructionIO
    from tldw_chatbook.Chat import console_chat_controller as controller

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(),
        _ssh_binding(),
        status_cache=_ready_cache_with_identity(),
    )
    assert selection is not None

    calls: list[tuple] = []

    class _StubExecutor:
        def ping(self):
            return {}

        def execute(self, tool, args, *, intent):
            return ""

    def _recording_factory(selection_arg, binding_id, **kwargs):
        calls.append((selection_arg, binding_id, kwargs))
        return _StubExecutor()

    reader = controller._remote_instruction_io_for_selection(
        selection,
        registry=None,
        status_cache=_ready_cache_with_identity(),
        executor_factory=_recording_factory,
    )
    # The reader wraps the factory's executor, built from the selection
    # under its binding id with the status cache and the typed
    # exclusions provider. (The REAL ssh factory is exercised at
    # integration level; constructing it here trips the test HOME's
    # config-recovery bootstrap, unrelated to this seam.)
    assert isinstance(reader, RemoteInstructionIO)
    assert len(calls) == 1
    assert calls[0][0] is selection
    assert calls[0][1] == "ssh-b1"
    assert set(calls[0][2]) == {"status_cache", "sensitive_exclusions"}
    exclusions_provider = calls[0][2]["sensitive_exclusions"]
    assert callable(exclusions_provider)
    entries = exclusions_provider()
    assert all(hasattr(entry, "kind") and hasattr(entry, "value") for entry in entries)
    assert {entry.value for entry in entries} == {"secrets", "build/out"}

    def _refuses_factory(*_args, **_kwargs):
        raise RuntimeError("binding unbuildable")

    assert (
        controller._remote_instruction_io_for_selection(
            selection,
            registry=None,
            status_cache=_ready_cache_with_identity(),
            executor_factory=_refuses_factory,
        )
        is None
    )

    local_root = SimpleNamespace(root=Path("/tmp/never"))
    assert (
        controller._remote_instruction_io_for_selection(
            local_root, registry=None, status_cache=None
        )
        is None
    )


def test_dispatch_path_unreachable_remote_warns_content_free_and_proceeds(
    monkeypatch,
) -> None:
    """Review Important 2: the dispatch-path guard (send path and preview
    share ``_resolve_remote_project_instruction_startup``) never lets an
    unreachable remote break the send — content-free warning, ``None``
    candidate, no exception text escapes."""
    import asyncio

    from loguru import logger as loguru_logger

    from tldw_chatbook.Agents.project_instruction_resolver import (
        ProjectInstructionResolver,
        RemoteInstructionIO,
    )
    from tldw_chatbook.Chat import console_chat_controller as controller

    selection = controller._validate_project_instruction_binding(
        _session_for_remote(),
        _ssh_binding(),
        status_cache=_ready_cache_with_identity(),
    )
    assert selection is not None
    guard = object.__new__(controller.ConsoleChatController)

    events: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: events.append(message), level="WARNING"
    )
    try:
        # (a) Reader construction raises (unreachable remote): the guard
        # catches, warns content-free, returns None.
        def _raising_reader(*_args, **_kwargs):
            raise RuntimeError("secret detail ssh://devbox/srv/www key")

        result = asyncio.run(
            guard._resolve_remote_project_instruction_startup(
                selection,
                registry=None,
                startup_max_bytes=32768,
                reader_for=_raising_reader,
            )
        )
        assert result is None

        # (c) Reader that yields a source-closed candidate (typed remote
        # failures): the send PROCEEDS with the content-free candidate.
        result = asyncio.run(
            guard._resolve_remote_project_instruction_startup(
                selection,
                registry=None,
                startup_max_bytes=32768,
                reader_for=lambda *args, **kwargs: RemoteInstructionIO(
                    _StubRefusingExecutor()
                ),
            )
        )
        assert result is not None
        assert result.source is None
        assert [
            (item.relative_path, item.code) for item in result.outcomes
        ] == [(".", "resolution_failed")]

        # (b) Reader built, resolution itself raises: same posture, and
        # the warning carries NO exception text.
        class _ExplodingResolver:
            def __init__(self, remote_io=None):
                pass

            def resolve_startup(self, **_kwargs):
                raise RuntimeError("secret body /srv/www/AGENTS.md")

        monkeypatch.setattr(
            controller, "ProjectInstructionResolver", _ExplodingResolver
        )
        result = asyncio.run(
            guard._resolve_remote_project_instruction_startup(
                selection,
                registry=None,
                startup_max_bytes=32768,
                reader_for=lambda *args, **kwargs: RemoteInstructionIO(
                    _StubRefusingExecutor()
                ),
            )
        )
        assert result is None
    finally:
        loguru_logger.remove(sink_id)

    warnings_text = "\n".join(events)
    assert "secret detail" not in warnings_text
    assert "secret body" not in warnings_text
    assert "srv/www/AGENTS.md" not in warnings_text
    assert "could not be read" in warnings_text or "unavailable" in warnings_text


class _StubRefusingExecutor:
    """Unreachable remote: typed errors on every executor call."""

    def ping(self):
        from tldw_chatbook.Tools.remote_workspace_executor import (
            RemoteWorkspaceExecutionError,
        )

        raise RemoteWorkspaceExecutionError("transport_failure", admitted=False)

    def execute(self, tool, args, *, intent):
        from tldw_chatbook.Tools.remote_workspace_executor import (
            RemoteWorkspaceExecutionError,
        )

        raise RemoteWorkspaceExecutionError("transport_failure", admitted=False)
