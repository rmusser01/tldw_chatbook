"""Immutable installed Guardian catalogs; no runtime or recovery imports."""

# Native catalogs measured from the exact author v1 DDL fixture and fresh/migrated
# v2 constructors. Historical layouts retain their original version labels.
GUARDIAN_SCHEMAS = (
    (
        2,
        (
            "CREATE INDEX idx_guardian_alerts_rule_ts\n            ON guardian_alerts(rule_id, ts DESC)\n        ",
            "CREATE INDEX idx_guardian_alerts_session_ts\n            ON guardian_alerts(session_id, ts)\n        ",
            "CREATE INDEX idx_guardian_alerts_topic_ts\n            ON guardian_alerts(topic, ts DESC)\n        ",
            "CREATE TABLE guardian_alerts (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            rule_id INTEGER,\n            session_id TEXT NOT NULL,\n            visit_id TEXT NOT NULL,\n            topic TEXT NOT NULL,\n            message_digest TEXT NOT NULL,\n            ts TEXT NOT NULL\n        , crisis_provenance INTEGER NOT NULL DEFAULT -1 CHECK(crisis_provenance IN (-1,0,1)))",
            "CREATE TABLE guardian_escalation_state (\n            rule_id INTEGER PRIMARY KEY,\n            session_count INTEGER NOT NULL DEFAULT 0,\n            window_count INTEGER NOT NULL DEFAULT 0,\n            window_start TEXT,\n            current_action TEXT,\n            cooldown_until TEXT\n        )",
            "CREATE TABLE guardian_meta (\n            key TEXT PRIMARY KEY,\n            value TEXT NOT NULL\n        )",
            "CREATE TABLE guardian_rules (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            name TEXT NOT NULL,\n            topic TEXT NOT NULL,\n            pattern TEXT NOT NULL,\n            except_patterns TEXT NOT NULL DEFAULT '[]',\n            action TEXT NOT NULL DEFAULT 'notify'\n                CHECK(action IN ('notify', 'redact', 'block')),\n            severity TEXT NOT NULL DEFAULT 'info'\n                CHECK(severity IN ('info', 'warning', 'critical')),\n            is_crisis INTEGER NOT NULL DEFAULT 0,\n            notification_frequency TEXT NOT NULL DEFAULT 'every_message'\n                CHECK(notification_frequency IN (\n                    'every_message', 'once_per_conversation',\n                    'once_per_session', 'once_per_day')),\n            display_mode TEXT NOT NULL DEFAULT 'inline_banner'\n                CHECK(display_mode IN (\n                    'inline_banner', 'post_visit_summary', 'silent_log')),\n            escalate_session_threshold INTEGER,\n            escalate_window_threshold INTEGER,\n            escalate_window_days INTEGER,\n            cooldown_minutes INTEGER,\n            feeds_discovery INTEGER NOT NULL DEFAULT 0,\n            enabled INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL\n        )",
            "CREATE TABLE guardian_visit_summaries (\n            visit_id TEXT PRIMARY KEY,\n            session_id TEXT NOT NULL,\n            created_at TEXT NOT NULL,\n            payload TEXT NOT NULL\n        )",
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL)",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    ),
    (
        1,
        (
            "CREATE INDEX idx_guardian_alerts_rule_ts\n            ON guardian_alerts(rule_id, ts DESC)\n        ",
            "CREATE INDEX idx_guardian_alerts_session_ts\n            ON guardian_alerts(session_id, ts)\n        ",
            "CREATE INDEX idx_guardian_alerts_topic_ts\n            ON guardian_alerts(topic, ts DESC)\n        ",
            "CREATE TABLE guardian_alerts (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            rule_id INTEGER,\n            session_id TEXT NOT NULL,\n            visit_id TEXT NOT NULL,\n            topic TEXT NOT NULL,\n            message_digest TEXT NOT NULL,\n            ts TEXT NOT NULL\n        )",
            "CREATE TABLE guardian_escalation_state (\n            rule_id INTEGER PRIMARY KEY,\n            session_count INTEGER NOT NULL DEFAULT 0,\n            window_count INTEGER NOT NULL DEFAULT 0,\n            window_start TEXT,\n            current_action TEXT,\n            cooldown_until TEXT\n        )",
            "CREATE TABLE guardian_meta (\n            key TEXT PRIMARY KEY,\n            value TEXT NOT NULL\n        )",
            "CREATE TABLE guardian_rules (\n            id INTEGER PRIMARY KEY AUTOINCREMENT,\n            name TEXT NOT NULL,\n            topic TEXT NOT NULL,\n            pattern TEXT NOT NULL,\n            except_patterns TEXT NOT NULL DEFAULT '[]',\n            action TEXT NOT NULL DEFAULT 'notify'\n                CHECK(action IN ('notify', 'redact', 'block')),\n            severity TEXT NOT NULL DEFAULT 'info'\n                CHECK(severity IN ('info', 'warning', 'critical')),\n            is_crisis INTEGER NOT NULL DEFAULT 0,\n            notification_frequency TEXT NOT NULL DEFAULT 'every_message'\n                CHECK(notification_frequency IN (\n                    'every_message', 'once_per_conversation',\n                    'once_per_session', 'once_per_day')),\n            display_mode TEXT NOT NULL DEFAULT 'inline_banner'\n                CHECK(display_mode IN (\n                    'inline_banner', 'post_visit_summary', 'silent_log')),\n            escalate_session_threshold INTEGER,\n            escalate_window_threshold INTEGER,\n            escalate_window_days INTEGER,\n            cooldown_minutes INTEGER,\n            feeds_discovery INTEGER NOT NULL DEFAULT 0,\n            enabled INTEGER NOT NULL DEFAULT 1,\n            created_at TEXT NOT NULL,\n            updated_at TEXT NOT NULL\n        )",
            "CREATE TABLE guardian_visit_summaries (\n            visit_id TEXT PRIMARY KEY,\n            session_id TEXT NOT NULL,\n            created_at TEXT NOT NULL,\n            payload TEXT NOT NULL\n        )",
            "CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL)",
            "CREATE TABLE sqlite_sequence(name,seq)",
        ),
    ),
)
