CREATE TABLE workflow_revisions (
    workflow_id TEXT NOT NULL,
    revision_id TEXT NOT NULL PRIMARY KEY,
    parents_json TEXT NOT NULL CHECK (json_valid(parents_json) AND json_type(parents_json) = 'array'),
    definition_json TEXT NOT NULL CHECK (json_valid(definition_json) AND json_type(definition_json) = 'object'),
    created_at TEXT NOT NULL,
    UNIQUE (workflow_id, revision_id)
);

CREATE INDEX workflow_revisions_history ON workflow_revisions(workflow_id, created_at);

CREATE TABLE workflow_heads (
    workflow_id TEXT PRIMARY KEY,
    revision_id TEXT NOT NULL REFERENCES workflow_revisions(revision_id),
    FOREIGN KEY (workflow_id, revision_id) REFERENCES workflow_revisions(workflow_id, revision_id)
);

CREATE TABLE workflow_drafts (
    workflow_id TEXT NOT NULL,
    base_revision_id TEXT NOT NULL REFERENCES workflow_revisions(revision_id),
    generation INTEGER NOT NULL CHECK (typeof(generation) = 'integer' AND generation >= 0),
    raw_text TEXT NOT NULL,
    last_valid_json TEXT NOT NULL CHECK (json_valid(last_valid_json) AND json_type(last_valid_json) = 'object'),
    error TEXT,
    PRIMARY KEY (workflow_id, base_revision_id),
    FOREIGN KEY (workflow_id, base_revision_id) REFERENCES workflow_revisions(workflow_id, revision_id)
);

PRAGMA user_version = 1;
