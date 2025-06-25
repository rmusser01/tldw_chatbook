#!/usr/bin/env python3
"""
Test script to check RAG dependencies and configuration.

This script verifies:
1. Optional dependencies for RAG features
2. Database connectivity
3. Configuration settings
4. Service availability
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
from loguru import logger
logger.add(sys.stderr, level="INFO")

def check_rag_dependencies() -> Dict[str, bool]:
    """Check all RAG-related dependencies."""
    results = {}
    
    logger.info("=== Checking RAG Dependencies ===")
    
    # Check core dependencies
    try:
        import torch
        results['torch'] = True
        logger.success("✅ torch available")
    except ImportError:
        results['torch'] = False
        logger.warning("❌ torch not available")
    
    try:
        import transformers
        results['transformers'] = True
        logger.success("✅ transformers available")
    except ImportError:
        results['transformers'] = False
        logger.warning("❌ transformers not available")
    
    try:
        import numpy
        results['numpy'] = True
        logger.success("✅ numpy available")
    except ImportError:
        results['numpy'] = False
        logger.warning("❌ numpy not available")
    
    try:
        import chromadb
        results['chromadb'] = True
        logger.success("✅ chromadb available")
    except ImportError:
        results['chromadb'] = False
        logger.warning("❌ chromadb not available")
    
    try:
        import sentence_transformers
        results['sentence_transformers'] = True
        logger.success("✅ sentence_transformers available")
    except ImportError:
        results['sentence_transformers'] = False
        logger.warning("❌ sentence_transformers not available")
    
    # Check reranking dependencies
    try:
        import flashrank
        results['flashrank'] = True
        logger.success("✅ flashrank available (for reranking)")
    except ImportError:
        results['flashrank'] = False
        logger.warning("❌ flashrank not available (reranking will be limited)")
    
    try:
        import cohere
        results['cohere'] = True
        logger.success("✅ cohere available (for advanced reranking)")
    except ImportError:
        results['cohere'] = False
        logger.warning("❌ cohere not available (Cohere reranking disabled)")
    
    # Check if all embeddings_rag dependencies are available
    embeddings_deps = ['torch', 'transformers', 'numpy', 'chromadb', 'sentence_transformers']
    results['embeddings_rag'] = all(results.get(dep, False) for dep in embeddings_deps)
    
    if results['embeddings_rag']:
        logger.success("✅ All embeddings/RAG dependencies available - Full RAG enabled")
    else:
        logger.warning("❌ Some embeddings/RAG dependencies missing - Only plain RAG available")
    
    return results

def check_databases() -> Dict[str, bool]:
    """Check database availability."""
    results = {}
    
    logger.info("\n=== Checking Databases ===")
    
    # Check Media DB
    media_db_path = Path.home() / ".local/share/tldw_cli/tldw_cli_media_v2.db"
    if media_db_path.exists():
        results['media_db'] = True
        logger.success(f"✅ Media DB found at: {media_db_path}")
        
        # Try to connect
        try:
            from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
            db = MediaDatabase(str(media_db_path), client_id="test")
            # Try a simple query
            count = db.get_media_count()
            logger.info(f"   Media items in DB: {count}")
        except Exception as e:
            logger.warning(f"   Warning: Could not query media DB: {e}")
    else:
        results['media_db'] = False
        logger.warning(f"❌ Media DB not found at: {media_db_path}")
    
    # Check ChaChaNotes DB
    chachanotes_db_path = Path.home() / ".local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
    if chachanotes_db_path.exists():
        results['chachanotes_db'] = True
        logger.success(f"✅ ChaChaNotes DB found at: {chachanotes_db_path}")
        
        # Try to connect
        try:
            from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
            db = CharactersRAGDB(str(chachanotes_db_path), client_id="test")
            # Try simple queries
            conv_count = len(db.get_all_conversations())
            note_count = len(db.get_all_notes())
            logger.info(f"   Conversations in DB: {conv_count}")
            logger.info(f"   Notes in DB: {note_count}")
        except Exception as e:
            logger.warning(f"   Warning: Could not query ChaChaNotes DB: {e}")
    else:
        results['chachanotes_db'] = False
        logger.warning(f"❌ ChaChaNotes DB not found at: {chachanotes_db_path}")
    
    return results

def check_rag_services() -> Dict[str, bool]:
    """Check RAG service availability."""
    results = {}
    
    logger.info("\n=== Checking RAG Services ===")
    
    # Check if RAG event handlers are available
    try:
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
            perform_plain_rag_search,
            perform_full_rag_pipeline,
            get_rag_context_for_chat
        )
        results['rag_events'] = True
        logger.success("✅ RAG event handlers available")
    except ImportError as e:
        results['rag_events'] = False
        logger.warning(f"❌ RAG event handlers not available: {e}")
    
    # Check if modular RAG is available
    try:
        from tldw_chatbook.RAG_Search.Services import RAGService, RAG_SERVICE_AVAILABLE
        results['modular_rag'] = RAG_SERVICE_AVAILABLE
        if RAG_SERVICE_AVAILABLE:
            logger.success("✅ Modular RAG service available")
        else:
            logger.warning("❌ Modular RAG service not available")
    except ImportError:
        results['modular_rag'] = False
        logger.warning("❌ Modular RAG service not found")
    
    # Check embeddings service
    try:
        from tldw_chatbook.RAG_Search.Services import EmbeddingsService
        results['embeddings_service'] = True
        logger.success("✅ Embeddings service available")
    except ImportError:
        results['embeddings_service'] = False
        logger.warning("❌ Embeddings service not available")
    
    # Check chunking service
    try:
        from tldw_chatbook.RAG_Search.Services import ChunkingService
        results['chunking_service'] = True
        logger.success("✅ Chunking service available")
    except ImportError:
        results['chunking_service'] = False
        logger.warning("❌ Chunking service not available")
    
    return results

def check_configuration() -> Dict[str, Any]:
    """Check configuration settings."""
    results = {}
    
    logger.info("\n=== Checking Configuration ===")
    
    # Check environment variables
    use_modular_rag = os.environ.get('USE_MODULAR_RAG', 'false').lower() in ('true', '1', 'yes')
    results['USE_MODULAR_RAG'] = use_modular_rag
    logger.info(f"USE_MODULAR_RAG environment variable: {use_modular_rag}")
    
    # Check config file
    config_path = Path.home() / ".config/tldw_cli/config.toml"
    if config_path.exists():
        results['config_exists'] = True
        logger.success(f"✅ Config file found at: {config_path}")
        
        # Try to load and check RAG settings
        try:
            import tomli
            with open(config_path, 'rb') as f:
                config = tomli.load(f)
            
            rag_config = config.get('rag', {})
            if rag_config:
                logger.info("   RAG configuration found:")
                logger.info(f"   - use_modular_service: {rag_config.get('use_modular_service', False)}")
                logger.info(f"   - batch_size: {rag_config.get('batch_size', 32)}")
                results['rag_config'] = True
            else:
                logger.info("   No RAG configuration section in config.toml")
                results['rag_config'] = False
        except Exception as e:
            logger.warning(f"   Could not parse config file: {e}")
            results['rag_config'] = False
    else:
        results['config_exists'] = False
        logger.warning(f"❌ Config file not found at: {config_path}")
    
    return results

def print_summary(deps: Dict[str, bool], dbs: Dict[str, bool], 
                  services: Dict[str, bool], config: Dict[str, Any]) -> None:
    """Print a summary of the checks."""
    logger.info("\n=== SUMMARY ===")
    
    # Determine RAG capabilities
    plain_rag_available = (
        dbs.get('media_db', False) or 
        dbs.get('chachanotes_db', False)
    ) and services.get('rag_events', False)
    
    full_rag_available = plain_rag_available and deps.get('embeddings_rag', False)
    
    modular_rag_available = services.get('modular_rag', False)
    
    logger.info("\nRAG Capabilities:")
    if plain_rag_available:
        logger.success("✅ Plain RAG (BM25/FTS5): AVAILABLE")
    else:
        logger.error("❌ Plain RAG (BM25/FTS5): NOT AVAILABLE")
    
    if full_rag_available:
        logger.success("✅ Full RAG (with embeddings): AVAILABLE")
    else:
        logger.warning("⚠️  Full RAG (with embeddings): NOT AVAILABLE")
        if not deps.get('embeddings_rag', False):
            logger.info("   To enable: pip install tldw_chatbook[embeddings_rag]")
    
    if modular_rag_available:
        logger.success("✅ Modular RAG (new architecture): AVAILABLE")
        if config.get('USE_MODULAR_RAG', False):
            logger.info("   Status: ENABLED via environment variable")
        else:
            logger.info("   Status: DISABLED (set USE_MODULAR_RAG=true to enable)")
    else:
        logger.info("ℹ️  Modular RAG: NOT AVAILABLE (optional)")
    
    # Reranking capabilities
    logger.info("\nReranking Capabilities:")
    if deps.get('flashrank', False):
        logger.success("✅ FlashRank reranking: AVAILABLE")
    else:
        logger.info("ℹ️  FlashRank reranking: NOT AVAILABLE (optional)")
        logger.info("   To enable: pip install flashrank")
    
    if deps.get('cohere', False):
        logger.success("✅ Cohere reranking: AVAILABLE")
    else:
        logger.info("ℹ️  Cohere reranking: NOT AVAILABLE (optional)")
        logger.info("   To enable: pip install cohere")

def main():
    """Run all checks."""
    logger.info("RAG Dependency and Configuration Check\n")
    
    # Run checks
    deps = check_rag_dependencies()
    dbs = check_databases()
    services = check_rag_services()
    config = check_configuration()
    
    # Print summary
    print_summary(deps, dbs, services, config)
    
    logger.info("\nCheck complete!")

if __name__ == "__main__":
    main()