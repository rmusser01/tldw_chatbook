"""Complete release gates are separate from individual native qualification."""


def test_complete_capability_requires_every_release_gate():
    from tldw_chatbook.Backup_Recovery.qualification import release_capability

    gates = {
        "helper": True,
        "owner_coverage": True,
        "admission": True,
        "archive": True,
        "native_publish": True,
        "restore": True,
        "product_flow": True,
    }
    assert release_capability(**gates) is True
    for name in gates:
        assert release_capability(**{**gates, name: False}) is False, name
