-- Migration: Add chunking configuration support
-- This migration adds per-document chunking configuration and chunking templates

-- Add chunking_config column to Media table
ALTER TABLE Media ADD COLUMN chunking_config TEXT;

-- Create ChunkingTemplates table for reusable chunking configurations
CREATE TABLE IF NOT EXISTS ChunkingTemplates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    template_json TEXT NOT NULL,
    is_system BOOLEAN DEFAULT 0,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_chunking_templates_name ON ChunkingTemplates(name);
CREATE INDEX IF NOT EXISTS idx_chunking_templates_is_system ON ChunkingTemplates(is_system);

-- Insert default system templates
INSERT INTO ChunkingTemplates (name, description, template_json, is_system) VALUES 
(
    'general',
    'Default balanced chunking approach',
    '{"name": "general", "description": "Default balanced chunking approach", "base_method": "words", "pipeline": [{"stage": "chunk", "method": "words", "options": {"max_size": 400, "overlap": 100}}], "metadata": {"version": "1.0"}}',
    1
),
(
    'academic_paper',
    'Structural chunking for academic papers with section preservation',
    '{"name": "academic_paper", "description": "Structural chunking for academic papers", "base_method": "structural", "pipeline": [{"stage": "preprocess", "operations": [{"type": "section_detection", "params": {"headers": ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "References"]}}]}, {"stage": "chunk", "method": "structural", "options": {"max_size": 500, "overlap": 50, "preserve_sections": true}}], "metadata": {"version": "1.0", "preserve_structure": true}}',
    1
),
(
    'code_documentation',
    'Code-aware chunking for technical documentation',
    '{"name": "code_documentation", "description": "Code-aware chunking", "base_method": "hierarchical", "pipeline": [{"stage": "preprocess", "operations": [{"type": "code_block_detection", "params": {}}]}, {"stage": "chunk", "method": "hierarchical", "options": {"max_size": 600, "overlap": 150, "preserve_code_blocks": true}}], "metadata": {"version": "1.0"}}',
    1
),
(
    'conversational',
    'Semantic chunking for chat logs and conversations',
    '{"name": "conversational", "description": "Semantic chunking for conversations", "base_method": "sentences", "pipeline": [{"stage": "chunk", "method": "sentences", "options": {"max_size": 300, "overlap": 50, "sentence_grouping": true}}], "metadata": {"version": "1.0", "optimize_for": "dialogue"}}',
    1
),
(
    'contextual',
    'Enhanced chunking with surrounding context preservation',
    '{"name": "contextual", "description": "Enhanced chunking with context", "base_method": "contextual", "pipeline": [{"stage": "chunk", "method": "contextual", "options": {"max_size": 400, "overlap": 100}}, {"stage": "postprocess", "operations": [{"type": "add_context", "params": {"context_size": 2}}]}], "metadata": {"version": "1.0"}}',
    1
);

-- Add trigger to update updated_at timestamp
CREATE TRIGGER update_chunking_templates_timestamp 
AFTER UPDATE ON ChunkingTemplates
BEGIN
    UPDATE ChunkingTemplates SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Add column to track which template was used for existing chunks
ALTER TABLE MediaChunks ADD COLUMN chunking_template TEXT;
ALTER TABLE UnvectorizedMediaChunks ADD COLUMN chunking_template TEXT;

-- Add column to track chunking method parameters for reproducibility
ALTER TABLE MediaChunks ADD COLUMN chunking_params TEXT;
ALTER TABLE UnvectorizedMediaChunks ADD COLUMN chunking_params TEXT;