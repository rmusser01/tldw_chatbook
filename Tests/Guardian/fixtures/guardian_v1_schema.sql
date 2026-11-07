-- Exact author 0c941ba2 Guardian v1 DDL; preserved predecessor fixture.
CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);

        CREATE TABLE IF NOT EXISTS guardian_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            topic TEXT NOT NULL,
            pattern TEXT NOT NULL,
            except_patterns TEXT NOT NULL DEFAULT '[]',
            action TEXT NOT NULL DEFAULT 'notify'
                CHECK(action IN ('notify', 'redact', 'block')),
            severity TEXT NOT NULL DEFAULT 'info'
                CHECK(severity IN ('info', 'warning', 'critical')),
            is_crisis INTEGER NOT NULL DEFAULT 0,
            notification_frequency TEXT NOT NULL DEFAULT 'every_message'
                CHECK(notification_frequency IN (
                    'every_message', 'once_per_conversation',
                    'once_per_session', 'once_per_day')),
            display_mode TEXT NOT NULL DEFAULT 'inline_banner'
                CHECK(display_mode IN (
                    'inline_banner', 'post_visit_summary', 'silent_log')),
            escalate_session_threshold INTEGER,
            escalate_window_threshold INTEGER,
            escalate_window_days INTEGER,
            cooldown_minutes INTEGER,
            feeds_discovery INTEGER NOT NULL DEFAULT 0,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ;

        CREATE TABLE IF NOT EXISTS guardian_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id INTEGER,
            session_id TEXT NOT NULL,
            visit_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            message_digest TEXT NOT NULL,
            ts TEXT NOT NULL
        )
        ;

        CREATE INDEX IF NOT EXISTS idx_guardian_alerts_rule_ts
            ON guardian_alerts(rule_id, ts DESC)
        ;

        CREATE INDEX IF NOT EXISTS idx_guardian_alerts_topic_ts
            ON guardian_alerts(topic, ts DESC)
        ;

        CREATE INDEX IF NOT EXISTS idx_guardian_alerts_session_ts
            ON guardian_alerts(session_id, ts)
        ;

        CREATE TABLE IF NOT EXISTS guardian_escalation_state (
            rule_id INTEGER PRIMARY KEY,
            session_count INTEGER NOT NULL DEFAULT 0,
            window_count INTEGER NOT NULL DEFAULT 0,
            window_start TEXT,
            current_action TEXT,
            cooldown_until TEXT
        )
        ;

        CREATE TABLE IF NOT EXISTS guardian_visit_summaries (
            visit_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        )
        ;

        CREATE TABLE IF NOT EXISTS guardian_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        ;
INSERT INTO schema_version VALUES (1);
