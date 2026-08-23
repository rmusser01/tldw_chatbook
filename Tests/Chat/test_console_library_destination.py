"""Conservative Console provider destination and disclosure contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleEgressClass,
    ConsoleResolvedDestination,
)
from tldw_chatbook.Chat.console_library_destination import (
    ConsoleLibraryDestinationRuntimeState,
    resolve_console_destination,
    settle_console_library_destination_runtime,
    update_console_library_destination_runtime,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution


def _resolution(
    endpoint: str,
    *,
    provider: str = "provider-a",
    model: str | None = "model-a",
    api_key: str | None = None,
) -> ConsoleProviderResolution:
    return ConsoleProviderResolution(
        provider=provider,
        base_url=endpoint,
        model=model,
        ready=True,
        api_key=api_key,
    )


@pytest.mark.parametrize(
    ("endpoint", "expected_class", "expected_identity"),
    [
        (
            "http://localhost:9099/v1",
            ConsoleEgressClass.ON_DEVICE,
            "http://localhost:9099",
        ),
        ("http://localhost:80/v1", ConsoleEgressClass.ON_DEVICE, "http://localhost"),
        (
            "http://127.42.7.9:8080",
            ConsoleEgressClass.ON_DEVICE,
            "http://127.42.7.9:8080",
        ),
        ("http://[::1]:8080/v1", ConsoleEgressClass.ON_DEVICE, "http://[::1]:8080"),
        (
            "unix:///private/tmp/model.sock",
            ConsoleEgressClass.ON_DEVICE,
            "unix://local",
        ),
        (
            "http+unix://%2Ftmp%2Fmodel.sock/v1",
            ConsoleEgressClass.ON_DEVICE,
            "http+unix://local",
        ),
        ("unix://", ConsoleEgressClass.UNKNOWN, "external/unknown"),
        ("http+unix://", ConsoleEgressClass.UNKNOWN, "external/unknown"),
        ("unix://user:secret@", ConsoleEgressClass.UNKNOWN, "external/unknown"),
        ("http+unix://user:secret@/v1", ConsoleEgressClass.UNKNOWN, "external/unknown"),
        (
            "http://10.20.30.40:8000/v1",
            ConsoleEgressClass.PRIVATE_NETWORK,
            "http://10.20.30.40:8000",
        ),
        ("http://172.16.0.1", ConsoleEgressClass.PRIVATE_NETWORK, "http://172.16.0.1"),
        (
            "http://192.168.4.5",
            ConsoleEgressClass.PRIVATE_NETWORK,
            "http://192.168.4.5",
        ),
        (
            "http://169.254.2.3",
            ConsoleEgressClass.PRIVATE_NETWORK,
            "http://169.254.2.3",
        ),
        (
            "http://[fe80::1%25en0]:8080/v1",
            ConsoleEgressClass.PRIVATE_NETWORK,
            "http://[fe80::1%25en0]:8080",
        ),
        (
            "http://[fd12:3456::9]:8080",
            ConsoleEgressClass.PRIVATE_NETWORK,
            "http://[fd12:3456::9]:8080",
        ),
        (
            "http://[::ffff:127.0.0.1]",
            ConsoleEgressClass.ON_DEVICE,
            "http://[::ffff:127.0.0.1]",
        ),
        (
            "http://[::ffff:10.0.0.1]",
            ConsoleEgressClass.PRIVATE_NETWORK,
            "http://[::ffff:10.0.0.1]",
        ),
        (
            "https://[::ffff:8.8.8.8]",
            ConsoleEgressClass.PUBLIC_NETWORK,
            "https://[::ffff:8.8.8.8]",
        ),
        (
            "https://8.8.8.8:443/private?q=secret#fragment",
            ConsoleEgressClass.PUBLIC_NETWORK,
            "https://8.8.8.8",
        ),
        (
            "https://[2606:4700:4700::1111]:443/v1",
            ConsoleEgressClass.PUBLIC_NETWORK,
            "https://[2606:4700:4700::1111]",
        ),
        (
            "https://api.openai.com/v1",
            ConsoleEgressClass.PUBLIC_NETWORK,
            "https://api.openai.com",
        ),
        (
            "https://openrouter.ai/api/v1",
            ConsoleEgressClass.PUBLIC_NETWORK,
            "https://openrouter.ai",
        ),
        (
            "https://models.example.test/v1",
            ConsoleEgressClass.UNKNOWN,
            "https://models.example.test",
        ),
        ("", ConsoleEgressClass.UNKNOWN, "external/unknown"),
        ("http://[::1", ConsoleEgressClass.UNKNOWN, "external/unknown"),
        (
            "https://example.test:99999/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        ("https://user:secret@", ConsoleEgressClass.UNKNOWN, "external/unknown"),
        (
            "https://user@@api.openai.com/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "https://%ZZ:secret@api.openai.com/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "https://api.open\u200bai.com/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "ht\u202etps://api.openai.com/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "https://user\u2066:secret@api.openai.com/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "https://api.openai.com\x7f/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "https://api.open\u009fai.com/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "https://example.test/v1\nsecret",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "https://" + ("a" * 3000) + ".test/v1",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
        (
            "file:///private/secret/model.gguf",
            ConsoleEgressClass.UNKNOWN,
            "external/unknown",
        ),
    ],
)
def test_resolved_destination_classifies_only_provable_endpoint_evidence(
    endpoint: str,
    expected_class: ConsoleEgressClass,
    expected_identity: str,
) -> None:
    destination = resolve_console_destination(_resolution(endpoint))

    assert destination.egress_class is expected_class
    assert destination.endpoint_identity == expected_identity
    assert len(destination.endpoint_identity) <= 253


def test_destination_identity_strips_credentials_paths_queries_and_fragments() -> None:
    first = resolve_console_destination(
        _resolution(
            "https://alice:API-SECRET@Models.Example.Test:8443/private/v1"
            "?api_key=API-SECRET#fragment"
        )
    )
    second = resolve_console_destination(
        _resolution(
            "https://bob:OTHER-SECRET@models.example.test:8443/another/path"
            "?token=OTHER-SECRET#other"
        )
    )

    assert first.endpoint_identity == "https://models.example.test:8443"
    assert first.identity_key == second.identity_key
    rendered = repr(first) + repr(first.identity_key)
    for secret in (
        "alice",
        "bob",
        "API-SECRET",
        "OTHER-SECRET",
        "private",
        "another",
        "api_key",
        "fragment",
    ):
        assert secret not in rendered


@pytest.mark.parametrize("control", ["\u200b", "\u202e", "\u2066", "\x7f", "\u009f"])
def test_destination_identity_rejects_unicode_and_c1_controls_from_all_surfaces(
    control: str,
) -> None:
    destination = resolve_console_destination(
        _resolution(f"https://user{control}:secret@api.openai.com/v1")
    )

    assert destination.endpoint_identity == "external/unknown"
    assert destination.egress_class is ConsoleEgressClass.UNKNOWN
    assert control not in repr(destination)
    assert control not in repr(destination.identity_key)


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1:8080/v1",
        "http://10.0.0.8:8080/v1",
        "https://8.8.4.4/v1",
        "https://api.openai.com/v1",
        "https://unresolved.example.test/v1",
    ],
)
def test_provider_name_and_api_key_presence_do_not_affect_egress_class(
    endpoint: str,
) -> None:
    without_key = resolve_console_destination(
        _resolution(endpoint, provider="llama_cpp", api_key=None)
    )
    with_key = resolve_console_destination(
        _resolution(endpoint, provider="openai", api_key="API-KEY-CANARY")
    )

    assert without_key.egress_class is with_key.egress_class
    assert without_key.endpoint_identity == with_key.endpoint_identity
    assert "API-KEY-CANARY" not in repr(with_key)
    assert "API-KEY-CANARY" not in repr(with_key.identity_key)


def _destination(
    endpoint_identity: str,
    egress_class: ConsoleEgressClass,
    *,
    provider: str = "provider-a",
) -> ConsoleResolvedDestination:
    return ConsoleResolvedDestination(
        provider=provider,
        model="model-a",
        endpoint_identity=endpoint_identity,
        egress_class=egress_class,
    )


def test_first_external_destination_does_not_invent_an_on_device_transition() -> None:
    external = _destination(
        "https://api.openai.com", ConsoleEgressClass.PUBLIC_NETWORK
    )

    state = update_console_library_destination_runtime(
        ConsoleLibraryDestinationRuntimeState(),
        external,
        library_data_possible=True,
    )

    assert state.resolved_destination == external
    assert state.last_resolved_identity == external.identity_key
    assert state.disclosure is None


def test_on_device_to_external_disclosure_replaces_on_later_identity_change() -> None:
    local = _destination("http://127.0.0.1:9099", ConsoleEgressClass.ON_DEVICE)
    public = _destination(
        "https://api.openai.com", ConsoleEgressClass.PUBLIC_NETWORK
    )
    private = _destination(
        "http://10.0.0.4:8080", ConsoleEgressClass.PRIVATE_NETWORK
    )
    state = update_console_library_destination_runtime(
        ConsoleLibraryDestinationRuntimeState(),
        local,
        library_data_possible=True,
    )

    disclosed = update_console_library_destination_runtime(
        state,
        public,
        library_data_possible=True,
    )
    repeated = update_console_library_destination_runtime(
        disclosed,
        public,
        library_data_possible=True,
    )
    replaced = update_console_library_destination_runtime(
        repeated,
        private,
        library_data_possible=True,
    )

    assert disclosed.disclosure is not None
    assert disclosed.disclosure.resolved_destination == public
    assert repeated.disclosure == disclosed.disclosure
    assert replaced.disclosure is not None
    assert replaced.disclosure.resolved_destination == private
    assert replaced.disclosure != disclosed.disclosure


def test_disclosure_clears_on_settlement_without_forgetting_destination() -> None:
    local = _destination("http://127.0.0.1:9099", ConsoleEgressClass.ON_DEVICE)
    external = _destination("https://8.8.8.8", ConsoleEgressClass.PUBLIC_NETWORK)
    state = update_console_library_destination_runtime(
        ConsoleLibraryDestinationRuntimeState(),
        local,
        library_data_possible=True,
    )
    state = update_console_library_destination_runtime(
        state,
        external,
        library_data_possible=True,
    )

    settled = settle_console_library_destination_runtime(state)
    same_destination = update_console_library_destination_runtime(
        settled,
        external,
        library_data_possible=True,
    )

    assert settled.disclosure is None
    assert settled.resolved_destination == external
    assert settled.last_resolved_identity == external.identity_key
    assert same_destination.disclosure is None


def test_policy_ineligible_transition_updates_runtime_without_disclosure() -> None:
    local = _destination("http://127.0.0.1:9099", ConsoleEgressClass.ON_DEVICE)
    external = _destination("external/unknown", ConsoleEgressClass.UNKNOWN)
    state = update_console_library_destination_runtime(
        ConsoleLibraryDestinationRuntimeState(),
        local,
        library_data_possible=False,
    )

    changed = update_console_library_destination_runtime(
        state,
        external,
        library_data_possible=False,
    )

    assert changed.last_resolved_identity == external.identity_key
    assert changed.resolved_destination == external
    assert changed.disclosure is None


def test_changing_back_to_on_device_clears_an_active_disclosure() -> None:
    local = _destination("http://127.0.0.1:9099", ConsoleEgressClass.ON_DEVICE)
    external = _destination("https://8.8.8.8", ConsoleEgressClass.PUBLIC_NETWORK)
    state = update_console_library_destination_runtime(
        ConsoleLibraryDestinationRuntimeState(),
        local,
        library_data_possible=True,
    )
    state = update_console_library_destination_runtime(
        state,
        external,
        library_data_possible=True,
    )

    returned = update_console_library_destination_runtime(
        state,
        replace(local, model="model-b"),
        library_data_possible=True,
    )

    assert returned.resolved_destination.egress_class is ConsoleEgressClass.ON_DEVICE
    assert returned.disclosure is None
