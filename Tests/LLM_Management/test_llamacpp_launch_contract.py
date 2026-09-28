from pathlib import Path

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events as events,
)


def test_builder_owns_exact_alias():
    command = events._build_gguf_server_command(
        "llamacpp", "server", Path("/model.gguf"), "127.0.0.1", "8080", ()
    )
    assert command.count("--alias") == 1
    assert command[command.index("--alias") + 1] == "chatbook-llamacpp"


@pytest.mark.parametrize(
    "args",
    [
        ("-a", "other"),
        ("--alias=other",),
        ("--host=localhost",),
        ("--port", "4"),
        ("--hf-file=other",),
        ("--models-dir=/private/models",),
    ],
)
def test_builder_refuses_competing_authority(args):
    with pytest.raises(ValueError):
        events._build_gguf_server_command(
            "llamacpp", "server", Path("/model.gguf"), "127.0.0.1", "8080", args
        )


@pytest.mark.loopback_network
def test_occupied_listener_is_rejected_before_spawn():
    import socket

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(OSError):
            events._preflight_llamacpp_listener("127.0.0.1", str(port))
    events._preflight_llamacpp_listener("127.0.0.1", str(port))


@pytest.mark.parametrize(
    "host,port,expected",
    [
        ("0.0.0.0", "8080", "http://127.0.0.1:8080"),
        ("::", "9000", "http://[::1]:9000"),
        ("localhost", "8080", "http://127.0.0.1:8080"),
    ],
)
def test_local_url_maps_bind_address_to_connectable_loopback(host, port, expected):
    from tldw_chatbook.LLM_Management.llamacpp_connection import local_launch_url

    assert local_launch_url(host, port) == expected
