
        CREATE TABLE IF NOT EXISTS dream_tracked_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin_story_id INTEGER,
            mechanism TEXT NOT NULL CHECK(mechanism IN ('page','question')),
            intent TEXT NOT NULL CHECK(intent IN ('event','deal','topic')),
            subscription_id INTEGER,
            query_template TEXT,
            event_date TEXT,
            cadence_seconds INTEGER NOT NULL,
            quiet_retire_count INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','paused','retired')),
            retired_reason TEXT,
            created_by_dreams INTEGER NOT NULL DEFAULT 0,
            last_checked TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        ;

        CREATE INDEX IF NOT EXISTS idx_dream_tracked_status
            ON dream_tracked_items(status)
        ;

        CREATE INDEX IF NOT EXISTS idx_dream_tracked_origin_story
            ON dream_tracked_items(origin_story_id)
        ;

        CREATE TABLE IF NOT EXISTS dream_track_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracked_item_id INTEGER NOT NULL REFERENCES dream_tracked_items(id) ON DELETE CASCADE,
            status TEXT NOT NULL CHECK(status IN ('changed','unchanged','baseline','rebaselined','withheld','error','skipped')),
            digest_hash TEXT,
            verdict_note TEXT NOT NULL DEFAULT '',
            notified INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        ;

        CREATE INDEX IF NOT EXISTS idx_dream_track_runs_item
            ON dream_track_runs(tracked_item_id, created_at DESC)
        ;
INSERT OR IGNORE INTO schema_version(version) VALUES (2);
