from tldw_chatbook.UI.Screens.settings_network_defaults import (
    SettingsNetworkTLS,
    build_network_save_sections,
    load_network_tls,
    network_ssl_toml_value,
    validate_network_tls,
)


def test_load_defaults_to_verify():
    assert load_network_tls({}).mode == "verify"


def test_load_bool_off():
    assert load_network_tls({"network": {"ssl_verify": False}}).mode == "off"


def test_load_lenient_strings():
    assert load_network_tls({"network": {"ssl_verify": "off"}}).mode == "off"
    assert load_network_tls({"network": {"ssl_verify": "1"}}).mode == "verify"


def test_load_custom_ca(tmp_path):
    ca = tmp_path / "corp.pem"
    ca.write_text("# ca")
    loaded = load_network_tls({"network": {"ssl_verify": str(ca)}})
    assert loaded.mode == "custom-ca"
    assert loaded.ca_bundle_path == str(ca)


def test_load_missing_path_is_invalid():
    loaded = load_network_tls({"network": {"ssl_verify": "/nope.pem"}})
    assert loaded.mode == "invalid"
    assert loaded.raw == "/nope.pem"


def test_load_unsupported_type_is_invalid():
    assert load_network_tls({"network": {"ssl_verify": 7}}).mode == "invalid"


def test_validate_custom_ca_requires_existing_readable_file(tmp_path):
    missing = validate_network_tls(SettingsNetworkTLS("custom-ca", "/nope.pem"))
    assert not missing.valid
    empty = validate_network_tls(SettingsNetworkTLS("custom-ca", "  "))
    assert not empty.valid
    ca = tmp_path / "corp.pem"
    ca.write_text("# ca")
    ok = validate_network_tls(SettingsNetworkTLS("custom-ca", str(ca)))
    assert ok.valid
    assert validate_network_tls(SettingsNetworkTLS("verify")).valid
    assert validate_network_tls(SettingsNetworkTLS("off")).valid
    assert not validate_network_tls(SettingsNetworkTLS("invalid", raw="x")).valid


def test_build_sections_round_trip(tmp_path):
    ca = tmp_path / "corp.pem"
    ca.write_text("# ca")
    assert build_network_save_sections(SettingsNetworkTLS("off")) == {
        "network": {"ssl_verify": False}
    }
    assert build_network_save_sections(SettingsNetworkTLS("verify")) == {
        "network": {"ssl_verify": True}
    }
    assert build_network_save_sections(
        SettingsNetworkTLS("custom-ca", str(ca))
    ) == {"network": {"ssl_verify": str(ca)}}
