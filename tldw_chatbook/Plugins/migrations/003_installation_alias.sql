ALTER TABLE installations ADD COLUMN alias TEXT;
CREATE UNIQUE INDEX installation_alias_unique ON installations(alias) WHERE alias IS NOT NULL;
