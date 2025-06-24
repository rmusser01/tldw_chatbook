#!/usr/bin/env python3
"""
RAG Setup and Validation Script

This script helps set up and validate the RAG system:
1. Checks dependencies
2. Creates necessary directories
3. Adds RAG configuration to config.toml
4. Tests basic functionality
"""

import sys
import os
from pathlib import Path
import subprocess
import toml
from loguru import logger

logger.add(sys.stderr, level="INFO")

class RAGSetup:
    def __init__(self):
        self.home = Path.home()
        self.config_dir = self.home / ".config/tldw_cli"
        self.data_dir = self.home / ".local/share/tldw_cli"
        self.config_file = self.config_dir / "config.toml"
        
    def check_dependencies(self):
        """Check and install RAG dependencies."""
        logger.info("=== Checking Dependencies ===")
        
        required_packages = {
            "chromadb": "chromadb>=0.4.0",
            "sentence_transformers": "sentence-transformers>=2.2.0",
            "flashrank": "flashrank",
            "tiktoken": "tiktoken>=0.5.0"
        }
        
        missing = []
        for module, package in required_packages.items():
            try:
                __import__(module)
                logger.success(f"✅ {module}")
            except ImportError:
                logger.warning(f"❌ {module} - Missing")
                missing.append(package)
                
        if missing:
            logger.info("\nInstalling missing dependencies...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing,
                    check=True
                )
                logger.success("Dependencies installed successfully!")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install dependencies: {e}")
                logger.info("\nPlease install manually:")
                logger.info("pip install -e \".[embeddings_rag]\"")
                return False
                
        return True
        
    def setup_directories(self):
        """Create necessary directories."""
        logger.info("\n=== Setting Up Directories ===")
        
        directories = [
            self.config_dir,
            self.data_dir,
            self.data_dir / "chromadb",
            self.data_dir / "rag_cache"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ {directory}")
            
    def add_rag_config(self):
        """Add RAG configuration to config.toml."""
        logger.info("\n=== Updating Configuration ===")
        
        # RAG configuration to add
        rag_config = {
            "rag": {
                "use_modular_service": False,
                "batch_size": 32,
                "num_workers": 4,
                "log_level": "INFO",
                "log_performance_metrics": False
            },
            "rag.retriever": {
                "fts_top_k": 10,
                "vector_top_k": 10,
                "hybrid_alpha": 0.5,
                "media_collection": "media_embeddings",
                "chat_collection": "chat_embeddings",
                "notes_collection": "notes_embeddings",
                "min_similarity_score": 0.3,
                "max_distance": 1.5
            },
            "rag.processor": {
                "enable_deduplication": True,
                "similarity_threshold": 0.85,
                "max_context_length": 4096,
                "max_context_chars": 16000,
                "enable_reranking": True,
                "reranker_provider": "flashrank",
                "rerank_top_k": 20,
                "use_tiktoken": True,
                "fallback_chars_per_token": 4
            },
            "rag.generator": {
                "enable_streaming": False,
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "system_prompt": "You are a helpful assistant with access to a knowledge base. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so clearly.",
                "include_citations": True,
                "citation_style": "inline"
            },
            "rag.cache": {
                "enable_cache": True,
                "cache_ttl": 3600,
                "max_cache_size": 1000,
                "cache_search_results": True,
                "cache_embeddings": True
            },
            "rag.embeddings": {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32,
                "normalize_embeddings": True
            },
            "rag.chunking": {
                "chunk_size": 400,
                "chunk_overlap": 100,
                "min_chunk_size": 50,
                "separators": ["\n\n", "\n", ". ", "! ", "? ", " "]
            },
            "rag.chroma": {
                "persist_directory": str(self.data_dir / "chromadb"),
                "anonymized_telemetry": False,
                "allow_reset": False
            }
        }
        
        # Load existing config or create new
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = toml.load(f)
                logger.info("Loaded existing config.toml")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
                config = {}
        else:
            config = {}
            logger.info("Creating new config.toml")
            
        # Check if RAG config already exists
        if "rag" in config:
            logger.info("RAG configuration already exists in config.toml")
            response = input("Update existing RAG configuration? (y/n): ")
            if response.lower() != 'y':
                return
                
        # Update config
        for section, values in rag_config.items():
            if "." in section:
                # Handle nested sections like rag.retriever
                main, sub = section.split(".", 1)
                if main not in config:
                    config[main] = {}
                config[main][sub] = values
            else:
                config[section] = values
                
        # Save updated config
        try:
            with open(self.config_file, 'w') as f:
                toml.dump(config, f)
            logger.success("✅ RAG configuration added to config.toml")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            
    def test_rag(self):
        """Test basic RAG functionality."""
        logger.info("\n=== Testing RAG Setup ===")
        
        try:
            # Test imports
            from tldw_chatbook.RAG_Search.Services import RAG_SERVICE_AVAILABLE
            
            if RAG_SERVICE_AVAILABLE:
                logger.success("✅ Modular RAG service is available")
            else:
                logger.warning("⚠️  Modular RAG service not available (check dependencies)")
                
            # Test database connections
            from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
            from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
            
            media_db_path = self.data_dir / "tldw_cli_media_v2.db"
            notes_db_path = self.data_dir / "tldw_chatbook_ChaChaNotes.db"
            
            if media_db_path.exists():
                logger.success("✅ Media database found")
            else:
                logger.warning("⚠️  Media database not found - ingest some media first")
                
            if notes_db_path.exists():
                logger.success("✅ Notes database found")
            else:
                logger.warning("⚠️  Notes database not found - create some notes first")
                
            # Test ChromaDB
            try:
                import chromadb
                client = chromadb.Client()
                logger.success("✅ ChromaDB client initialized")
            except Exception as e:
                logger.error(f"❌ ChromaDB error: {e}")
                
            logger.info("\n✅ RAG setup complete!")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            
    def print_usage(self):
        """Print usage instructions."""
        logger.info("\n=== Next Steps ===")
        logger.info("\n1. To use the legacy RAG (default):")
        logger.info("   python3 -m tldw_chatbook.app")
        logger.info("\n2. To use the new modular RAG:")
        logger.info("   USE_MODULAR_RAG=true python3 -m tldw_chatbook.app")
        logger.info("\n3. To test RAG functionality:")
        logger.info("   python test_rag_comprehensive.py")
        logger.info("\n4. In the chat window:")
        logger.info("   - Click the RAG toggle button")
        logger.info("   - Select data sources (Media, Conversations, Notes)")
        logger.info("   - Ask questions about your indexed content")
        
def main():
    """Run the setup process."""
    logger.info("=== RAG Setup Script ===\n")
    
    setup = RAGSetup()
    
    # Run setup steps
    if not setup.check_dependencies():
        logger.error("Please install dependencies and run again")
        return
        
    setup.setup_directories()
    setup.add_rag_config()
    setup.test_rag()
    setup.print_usage()
    
if __name__ == "__main__":
    main()