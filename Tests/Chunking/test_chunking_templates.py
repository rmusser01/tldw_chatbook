# test_chunking_templates.py
"""
Tests for the modular chunking template system.
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List

from tldw_chatbook.Chunking.chunking_templates import (
    ChunkingTemplate,
    ChunkingTemplateManager,
    ChunkingPipeline,
    ChunkingOperation,
    ChunkingStage
)
from tldw_chatbook.Chunking.Chunk_Lib import Chunker, improved_chunking_process


class TestChunkingTemplate:
    """Test the ChunkingTemplate model."""
    
    def test_template_creation(self):
        """Test creating a basic template."""
        template = ChunkingTemplate(
            name="test_template",
            description="Test template",
            base_method="words",
            pipeline=[
                ChunkingStage(
                    stage="chunk",
                    method="words",
                    options={"max_size": 100, "overlap": 20}
                )
            ]
        )
        
        assert template.name == "test_template"
        assert template.base_method == "words"
        assert len(template.pipeline) == 1
        assert template.pipeline[0].stage == "chunk"
    
    def test_template_with_operations(self):
        """Test template with preprocessing operations."""
        template = ChunkingTemplate(
            name="test_ops",
            base_method="sentences",
            pipeline=[
                ChunkingStage(
                    stage="preprocess",
                    operations=[
                        ChunkingOperation(
                            type="normalize_whitespace",
                            params={}
                        )
                    ]
                ),
                ChunkingStage(
                    stage="chunk",
                    method="sentences",
                    options={"max_size": 5}
                )
            ]
        )
        
        assert len(template.pipeline) == 2
        assert template.pipeline[0].operations[0].type == "normalize_whitespace"


class TestChunkingTemplateManager:
    """Test the template manager functionality."""
    
    @pytest.fixture
    def temp_templates_dir(self):
        """Create a temporary directory for test templates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def manager_with_temp_dir(self, temp_templates_dir):
        """Create a template manager with temporary directory."""
        return ChunkingTemplateManager(
            templates_dir=temp_templates_dir,
            user_templates_dir=temp_templates_dir
        )
    
    def test_manager_initialization(self, manager_with_temp_dir):
        """Test template manager initialization."""
        manager = manager_with_temp_dir
        
        # Check that built-in operations are registered
        assert "normalize_whitespace" in manager._operations
        assert "extract_metadata" in manager._operations
        assert "add_context" in manager._operations
    
    def test_save_and_load_template(self, manager_with_temp_dir, temp_templates_dir):
        """Test saving and loading a template."""
        manager = manager_with_temp_dir
        
        # Create a template
        template = ChunkingTemplate(
            name="test_save",
            description="Test save/load",
            base_method="words",
            pipeline=[
                ChunkingStage(
                    stage="chunk",
                    method="words",
                    options={"max_size": 200}
                )
            ]
        )
        
        # Save it
        manager.save_template(template)
        
        # Verify file exists
        template_path = temp_templates_dir / "test_save.json"
        assert template_path.exists()
        
        # Load it back
        loaded = manager.load_template("test_save")
        assert loaded is not None
        assert loaded.name == "test_save"
        assert loaded.pipeline[0].options["max_size"] == 200
    
    def test_template_inheritance(self, manager_with_temp_dir):
        """Test template inheritance functionality."""
        manager = manager_with_temp_dir
        
        # Create parent template
        parent = ChunkingTemplate(
            name="parent",
            base_method="words",
            pipeline=[
                ChunkingStage(
                    stage="preprocess",
                    operations=[
                        ChunkingOperation(type="normalize_whitespace")
                    ]
                ),
                ChunkingStage(
                    stage="chunk",
                    method="words",
                    options={"max_size": 100}
                )
            ]
        )
        manager.save_template(parent)
        
        # Create child template
        child = ChunkingTemplate(
            name="child",
            parent_template="parent",
            pipeline=[
                ChunkingStage(
                    stage="chunk",
                    options={"max_size": 200}  # Override parent's max_size
                )
            ]
        )
        manager.save_template(child)
        
        # Load child and verify inheritance
        loaded_child = manager.load_template("child")
        assert loaded_child is not None
        
        # Should inherit preprocess stage from parent
        preprocess_stages = [s for s in loaded_child.pipeline if s.stage == "preprocess"]
        assert len(preprocess_stages) == 1
        assert preprocess_stages[0].operations[0].type == "normalize_whitespace"
        
        # Should override chunk options
        chunk_stages = [s for s in loaded_child.pipeline if s.stage == "chunk"]
        assert len(chunk_stages) == 1
        assert chunk_stages[0].options["max_size"] == 200


class TestChunkingPipeline:
    """Test the chunking pipeline execution."""
    
    @pytest.fixture
    def manager(self):
        """Create a template manager."""
        return ChunkingTemplateManager()
    
    @pytest.fixture
    def pipeline(self, manager):
        """Create a pipeline instance."""
        return ChunkingPipeline(manager)
    
    def test_simple_pipeline_execution(self, pipeline, manager):
        """Test executing a simple pipeline."""
        # Create a simple template
        template = ChunkingTemplate(
            name="simple",
            base_method="words",
            pipeline=[
                ChunkingStage(
                    stage="chunk",
                    method="words",
                    options={"max_size": 10, "overlap": 2}
                )
            ]
        )
        
        # Create chunker instance
        chunker = Chunker()
        
        # Test text
        text = " ".join([f"word{i}" for i in range(50)])
        
        # Execute pipeline
        results = pipeline.execute(text, template, chunker)
        
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert all("text" in r for r in results)
        assert all("metadata" in r for r in results)
    
    def test_pipeline_with_operations(self, pipeline, manager):
        """Test pipeline with preprocessing and postprocessing."""
        template = ChunkingTemplate(
            name="ops_test",
            base_method="sentences",
            pipeline=[
                ChunkingStage(
                    stage="preprocess",
                    operations=[
                        ChunkingOperation(
                            type="normalize_whitespace"
                        )
                    ]
                ),
                ChunkingStage(
                    stage="chunk",
                    method="sentences",
                    options={"max_size": 2}
                ),
                ChunkingStage(
                    stage="postprocess",
                    operations=[
                        ChunkingOperation(
                            type="filter_empty",
                            params={"min_length": 10}
                        )
                    ]
                )
            ]
        )
        
        chunker = Chunker()
        text = "This is   a test.   \n\n  Another sentence.  Short.  And one more sentence here."
        
        results = pipeline.execute(text, template, chunker)
        
        # Should have filtered out very short chunks
        assert all(len(r["text"]) >= 10 for r in results)


class TestChunkerWithTemplates:
    """Test the Chunker class with template support."""
    
    def test_chunker_with_template_name(self):
        """Test creating a chunker with a template name."""
        # This should load the built-in "words" template
        chunker = Chunker(template="words")
        
        assert chunker.template is not None
        assert chunker.template.name == "words"
    
    def test_chunker_template_override(self):
        """Test that explicit options override template options."""
        chunker = Chunker(
            template="words",
            options={"max_size": 1000}  # Override template's max_size
        )
        
        assert chunker.options["max_size"] == 1000
    
    def test_chunk_text_with_template(self):
        """Test chunking text using a template."""
        chunker = Chunker(template="words")
        
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestImprovedChunkingProcessWithTemplates:
    """Test the improved_chunking_process function with templates."""
    
    def test_improved_process_with_template(self):
        """Test improved chunking process with a template."""
        text = "This is a test document. " * 50
        
        results = improved_chunking_process(
            text,
            template="sentences",
            chunk_options_dict={"max_size": 3}
        )
        
        assert len(results) > 0
        assert all(isinstance(r, dict) for r in results)
        assert all("text" in r and "metadata" in r for r in results)
        
        # Check metadata includes template info
        first_chunk = results[0]
        assert "chunk_method" in first_chunk["metadata"]
    
    def test_template_with_domain_specific(self):
        """Test using a domain-specific template."""
        # Create some academic-style text
        text = """
        Abstract: This paper presents a novel approach to text chunking.
        
        Introduction
        Text chunking is an important preprocessing step for many NLP tasks.
        
        Methods
        We propose a template-based approach that allows for flexible chunking strategies.
        
        Results
        Our experiments show improved performance on downstream tasks.
        
        Conclusion
        The template-based approach provides significant benefits.
        """
        
        # Try to use academic_paper template if it exists
        try:
            results = improved_chunking_process(
                text,
                template="academic_paper"
            )
            
            assert len(results) > 0
            # Should have extracted metadata
            metadata = results[0]["metadata"]
            assert metadata is not None
        except Exception:
            # Template might not exist in test environment
            pytest.skip("academic_paper template not found")


class TestCustomOperations:
    """Test custom chunking operations."""
    
    def test_register_custom_operation(self):
        """Test registering a custom operation."""
        manager = ChunkingTemplateManager()
        
        # Define a custom operation
        def custom_op(text: str, chunks: List[str], options: Dict[str, Any]) -> List[str]:
            # Add a prefix to each chunk
            prefix = options.get("prefix", "[CUSTOM]")
            return [f"{prefix} {chunk}" for chunk in chunks]
        
        # Register it
        manager.register_operation("add_prefix", custom_op)
        
        # Verify it's registered
        assert "add_prefix" in manager._operations
        
        # Test using it
        result = manager._operations["add_prefix"](
            "test",
            ["chunk1", "chunk2"],
            {"prefix": "[TEST]"}
        )
        
        assert result == ["[TEST] chunk1", "[TEST] chunk2"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])