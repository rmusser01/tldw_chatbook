"""Display copies must not expose credentials from admitted captured calls."""

import json
from copy import deepcopy
from urllib.parse import quote

import pytest

from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.virtual_cli_provider import VirtualCliProvider
from tldw_chatbook.Chat.approval_presentation import (
    capture_approval_view,
    profile_authority,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.virtual_cli_impls import parse_request
from tldw_chatbook.UI.Console_Modules.approval_details import iter_redacted_details

pytestmark = pytest.mark.bootstrap_profile


def captured_row(tmp_path, argv):
    arguments = {"command": "grep", "argv": argv}
    originals = deepcopy(arguments)
    parse_request(arguments["command"], arguments["argv"])
    provider = VirtualCliProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _tool: EffectiveToolState(
            state="ask", origin="global_default"
        ),
        presentation_authority=profile_authority(
            "virtual_cli", "default", "Scratch", "call"
        ),
    )
    pending = provider.pending_gate_for(
        ToolCall("virtual_cli", arguments, "captured-call")
    )
    assert pending is not None
    row = capture_approval_view(
        [pending], round_id="round", session_id="chat", run_id="run", revision=1
    ).rows[0]
    assert dict(row.argument_sets[0]) == originals
    assert arguments == originals
    return row, arguments, originals


@pytest.mark.parametrize(
    "pattern",
    [
        "API_KEY=synthetic-ordinary-credential",
        'API_KEY="synthetic-ordinary-credential"',
        "--api-key=synthetic-ordinary-credential",
        "--api-key='synthetic-ordinary-credential'",
    ],
)
def test_admitted_virtual_cli_details_keep_argv_credentials_masked(tmp_path, pattern):
    row, arguments, originals = captured_row(tmp_path, ["--", pattern])
    assert "synthetic-ordinary-credential" not in row.targets[0]
    pages = list(
        iter_redacted_details(row.argument_sets, targets=row.targets, page_chars=41)
    )
    assert all(len(page.text) <= 41 for page in pages)
    assert "synthetic-ordinary-credential" not in "".join(page.text for page in pages)
    assert arguments == originals and dict(row.argument_sets[0]) == originals


def test_details_argv_key_flag_pairs_mask_only_display_copy():
    arguments = {
        "argv": ["--api-key", "synthetic-ordinary-credential", "--mode", "plain"]
    }
    originals = deepcopy(arguments)
    safe = json.loads("".join(page.text for page in iter_redacted_details([arguments])))
    assert safe == [{"argv": ["--api-key", "***", "--mode", "plain"]}]
    assert arguments == originals


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.test/callback#access_token=synthetic-oauth-credential&state=nonsecret",
        "https://example.test/callback?mode=plain#access_token=synthetic-oauth-credential&state=nonsecret",
        "https://example.test/callback#next="
        + quote(
            "https://example.test/?access_token=synthetic-oauth-credential", safe=""
        )
        + "&state=nonsecret",
    ],
)
def test_admitted_virtual_cli_details_mask_oauth_fragments(tmp_path, uri):
    row, arguments, originals = captured_row(tmp_path, [uri])
    rendered = "".join(
        page.text
        for page in iter_redacted_details(row.argument_sets, targets=row.targets)
    )
    assert "synthetic-oauth-credential" not in rendered
    assert "nonsecret" in rendered
    assert arguments == originals and dict(row.argument_sets[0]) == originals


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.test/docs#chapter-two",
        "https://example.test/callback?mode=a%20b#state=nonsecret&tab=one",
    ],
)
def test_details_preserve_nonsecret_anchors_and_parameter_spelling(tmp_path, uri):
    row, arguments, originals = captured_row(tmp_path, [uri])
    safe = json.loads(
        "".join(page.text for page in iter_redacted_details(row.argument_sets))
    )
    assert safe[0]["argv"] == [uri]
    assert arguments == originals
