CREATE TABLE Prompts_sequence_migration (seq INTEGER);
INSERT INTO Prompts_sequence_migration SELECT seq FROM sqlite_sequence WHERE name = 'Prompts';
CREATE TABLE Prompts_mixin_migration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    author TEXT,
    details TEXT,
    system_prompt TEXT,
    user_prompt TEXT,
    uuid TEXT UNIQUE NOT NULL,
    last_modified DATETIME NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    client_id TEXT NOT NULL,
    deleted BOOLEAN NOT NULL DEFAULT 0,
    prev_version INTEGER,
    merge_parent_uuid TEXT,
    prompt_format TEXT NOT NULL DEFAULT 'legacy',
    prompt_schema_version INTEGER,
    prompt_definition TEXT,
    artifact_type TEXT NOT NULL DEFAULT 'prompt'
        CHECK(artifact_type IN ('prompt', 'recipe', 'mixin'))
);
INSERT INTO Prompts_mixin_migration SELECT * FROM Prompts;
DROP TABLE Prompts;
ALTER TABLE Prompts_mixin_migration RENAME TO Prompts;
INSERT INTO sqlite_sequence(name, seq)
SELECT 'Prompts', seq FROM Prompts_sequence_migration
WHERE NOT EXISTS (SELECT 1 FROM sqlite_sequence WHERE name = 'Prompts');
UPDATE sqlite_sequence SET seq = MAX(seq, COALESCE((SELECT seq FROM Prompts_sequence_migration), seq))
WHERE name = 'Prompts';
DROP TABLE Prompts_sequence_migration;
UPDATE schema_version SET version = 6 WHERE version = 5;
