"""Scope resolution must mask logical IDs, not just old revision numbers."""

from Tests.Chat.response_rules_fixtures import revision
from tldw_chatbook.Chat.response_rules.models import RuleBinding, RuleScope
from tldw_chatbook.Chat.response_rules.resolution import resolve_effective_rules

CHAT = RuleScope("chat", "chat")
WORKSPACE = RuleScope("workspace", "workspace")
GLOBAL = RuleScope("global", "profile")


def resolve(bindings, revisions):
    return resolve_effective_rules(
        tuple(bindings),
        tuple(revisions),
        chat=CHAT,
        workspace=WORKSPACE,
        global_scope=GLOBAL,
    )


def test_scope_exclusion_survives_inherited_revision_change():
    exclusion = RuleBinding(CHAT, "rule", None, "excluded", 1)
    for number in (1, 2):
        inherited = RuleBinding(WORKSPACE, "rule", number, "enabled", number)
        excluded_effective = resolve([exclusion, inherited], [revision(number=number)])
        assert excluded_effective == ()
    assert resolve([inherited], [revision(number=2)])[0].revision == 2


def test_precedence_pins_revision_and_deduplicates():
    bindings = [
        RuleBinding(GLOBAL, "rule", 2, "enabled", 1),
        RuleBinding(WORKSPACE, "rule", 2, "enabled", 1),
        RuleBinding(CHAT, "rule", 1, "enabled", 1),
    ]
    selected = resolve(bindings + bindings, [revision(), revision(number=2)])
    assert [(r.rule_id, r.revision) for r in selected] == [("rule", 1)]


def test_disabled_override_and_missing_revision_never_fall_back():
    inherited = RuleBinding(GLOBAL, "rule", 1, "enabled", 1)
    assert (
        resolve([inherited, RuleBinding(CHAT, "rule", 1, "disabled", 1)], [revision()])
        == ()
    )
    assert (
        resolve([inherited, RuleBinding(CHAT, "rule", 2, "enabled", 1)], [revision()])
        == ()
    )
    assert (
        resolve(
            [RuleBinding(RuleScope("chat", "other"), "rule", 1, "enabled", 1)],
            [revision()],
        )
        == ()
    )
