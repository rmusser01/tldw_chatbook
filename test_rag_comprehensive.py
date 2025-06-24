#!/usr/bin/env python3
"""
Comprehensive RAG Testing Suite

This script tests various aspects of the RAG implementation:
- Basic search functionality
- Different data sources
- Chunking overlap with Chunk_Lib
- Performance comparison
- Dependency checks
"""

import asyncio
import sys
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
from loguru import logger
logger.add(sys.stderr, level="INFO")

class RAGTester:
    """Comprehensive RAG testing suite."""
    
    def __init__(self):
        self.results = {
            "dependencies": {},
            "database_checks": {},
            "search_tests": {},
            "chunking_tests": {},
            "performance": {},
            "errors": []
        }
        
    async def check_dependencies(self):
        """Check if all RAG dependencies are available."""
        logger.info("=== Checking Dependencies ===")
        
        dependencies = {
            "chromadb": "Vector database",
            "sentence_transformers": "Embeddings",
            "flashrank": "Reranking",
            "tiktoken": "Token counting",
            "langdetect": "Language detection",
            "nltk": "Text processing"
        }
        
        for module, description in dependencies.items():
            try:
                __import__(module)
                self.results["dependencies"][module] = {"status": "✅", "description": description}
                logger.success(f"{module}: ✅ {description}")
            except ImportError:
                self.results["dependencies"][module] = {"status": "❌", "description": description}
                logger.warning(f"{module}: ❌ {description} - Not installed")
                
    async def check_databases(self):
        """Check if required databases exist and have content."""
        logger.info("\n=== Checking Databases ===")
        
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
        
        # Check Media DB
        media_db_path = Path.home() / ".local/share/tldw_cli/tldw_cli_media_v2.db"
        if media_db_path.exists():
            try:
                media_db = MediaDatabase(media_db_path)
                # Check for content
                with media_db.get_connection() as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM media_files")
                    count = cursor.fetchone()[0]
                self.results["database_checks"]["media_db"] = {
                    "exists": True,
                    "record_count": count
                }
                logger.info(f"Media DB: ✅ Found {count} media files")
            except Exception as e:
                self.results["database_checks"]["media_db"] = {"exists": True, "error": str(e)}
                logger.error(f"Media DB: ❌ Error: {e}")
        else:
            self.results["database_checks"]["media_db"] = {"exists": False}
            logger.warning("Media DB: ❌ Not found")
            
        # Check ChaChaNotes DB
        notes_db_path = Path.home() / ".local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
        if notes_db_path.exists():
            try:
                notes_db = CharactersRAGDB(notes_db_path)
                # Check for content
                note_count = len(notes_db.get_all_notes())
                chat_count = len(notes_db.get_all_conversations())
                self.results["database_checks"]["chachanotes_db"] = {
                    "exists": True,
                    "note_count": note_count,
                    "chat_count": chat_count
                }
                logger.info(f"ChaChaNotes DB: ✅ Found {note_count} notes, {chat_count} conversations")
            except Exception as e:
                self.results["database_checks"]["chachanotes_db"] = {"exists": True, "error": str(e)}
                logger.error(f"ChaChaNotes DB: ❌ Error: {e}")
        else:
            self.results["database_checks"]["chachanotes_db"] = {"exists": False}
            logger.warning("ChaChaNotes DB: ❌ Not found")
            
        # Check ChromaDB
        chroma_path = Path.home() / ".local/share/tldw_cli/chromadb"
        self.results["database_checks"]["chromadb"] = {"path_exists": chroma_path.exists()}
        logger.info(f"ChromaDB path: {'✅' if chroma_path.exists() else '❌'} {chroma_path}")
        
    async def test_basic_search(self):
        """Test basic RAG search functionality."""
        logger.info("\n=== Testing Basic Search ===")
        
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import perform_plain_rag_search
        
        # Create mock app
        class MockApp:
            def __init__(self):
                from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
                from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
                
                self.media_db = MediaDatabase(Path.home() / ".local/share/tldw_cli/tldw_cli_media_v2.db")
                self.rag_db = CharactersRAGDB(Path.home() / ".local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db")
                self.config = {}
                
        app = MockApp()
        
        # Test queries
        test_queries = [
            ("test", "Simple keyword"),
            ("What is machine learning?", "Question format"),
            ("explain neural networks", "Technical topic")
        ]
        
        for query, description in test_queries:
            logger.info(f"\nTesting: {description} - '{query}'")
            
            try:
                start_time = time.time()
                
                results, context = await perform_plain_rag_search(
                    app=app,
                    query=query,
                    sources={'media': True, 'conversations': True, 'notes': True},
                    top_k=3,
                    max_context_length=2000,
                    enable_rerank=False
                )
                
                elapsed = time.time() - start_time
                
                self.results["search_tests"][query] = {
                    "success": True,
                    "result_count": len(results),
                    "context_length": len(context),
                    "elapsed_time": elapsed,
                    "sources": [r['source'] for r in results[:3]]
                }
                
                logger.success(f"✅ Found {len(results)} results in {elapsed:.2f}s")
                
            except Exception as e:
                self.results["search_tests"][query] = {"success": False, "error": str(e)}
                logger.error(f"❌ Error: {e}")
                
    async def test_chunking_comparison(self):
        """Compare RAG chunking with Chunk_Lib."""
        logger.info("\n=== Testing Chunking Functionality ===")
        
        try:
            # Test RAG chunking
            from tldw_chatbook.RAG_Search.Services.utils import chunk_text as rag_chunk
            
            # Test Chunk_Lib chunking
            from tldw_chatbook.Chunking.Chunk_Lib import Chunker
            
            test_text = """This is a test document for chunking comparison.
            
            It contains multiple paragraphs to test how different chunking methods work.
            The RAG system has basic chunking functionality, while the Chunk_Lib provides
            more advanced options including semantic chunking and language-specific handling.
            
            This third paragraph helps us test overlap handling and chunk boundaries.
            We want to see how each implementation handles text splitting."""
            
            # RAG chunking
            rag_chunks = rag_chunk(test_text, chunk_size=100, chunk_overlap=20)
            
            # Chunk_Lib chunking
            chunker = Chunker({
                'method': 'words',
                'max_size': 100,
                'overlap': 20
            })
            chunk_lib_result = chunker.chunk(test_text)
            chunk_lib_chunks = chunk_lib_result['chunks']
            
            self.results["chunking_tests"] = {
                "rag_chunks": len(rag_chunks),
                "chunk_lib_chunks": len(chunk_lib_chunks),
                "rag_implementation": "Basic character-based with overlap",
                "chunk_lib_methods": ["words", "sentences", "paragraphs", "tokens", "semantic"],
                "recommendation": "Integrate Chunk_Lib for advanced chunking in RAG"
            }
            
            logger.info(f"RAG chunking: {len(rag_chunks)} chunks")
            logger.info(f"Chunk_Lib: {len(chunk_lib_chunks)} chunks")
            logger.info("Chunk_Lib provides more sophisticated chunking methods")
            
        except Exception as e:
            self.results["chunking_tests"]["error"] = str(e)
            logger.error(f"Chunking test error: {e}")
            
    async def test_modular_rag(self):
        """Test the new modular RAG implementation."""
        logger.info("\n=== Testing Modular RAG Service ===")
        
        try:
            from tldw_chatbook.RAG_Search.Services import RAGService, RAG_SERVICE_AVAILABLE
            
            if not RAG_SERVICE_AVAILABLE:
                self.results["performance"]["modular_available"] = False
                logger.warning("Modular RAG service not available")
                return
                
            # Initialize service
            service = RAGService(
                media_db_path=Path.home() / ".local/share/tldw_cli/tldw_cli_media_v2.db",
                chachanotes_db_path=Path.home() / ".local/share/tldw_cli/tldw_chatbook_ChaChaNotes.db"
            )
            
            await service.initialize()
            
            # Test search
            start_time = time.time()
            results = await service.search(
                query="test query",
                sources=["MEDIA_DB", "NOTES"]
            )
            elapsed = time.time() - start_time
            
            # Get stats
            stats = service.get_stats()
            
            self.results["performance"]["modular_rag"] = {
                "available": True,
                "initialization": "Success",
                "search_time": elapsed,
                "result_count": len(results),
                "stats": stats
            }
            
            logger.success(f"✅ Modular RAG working - {len(results)} results in {elapsed:.2f}s")
            
        except Exception as e:
            self.results["performance"]["modular_rag"] = {"available": False, "error": str(e)}
            logger.error(f"Modular RAG error: {e}")
            
    async def check_rag_indexing(self):
        """Check RAG indexing database."""
        logger.info("\n=== Checking RAG Indexing ===")
        
        try:
            from tldw_chatbook.DB.RAG_Indexing_DB import RAGIndexingDB
            
            indexing_db_path = Path.home() / ".local/share/tldw_cli/rag_indexing.db"
            
            if indexing_db_path.exists():
                indexing_db = RAGIndexingDB(indexing_db_path)
                
                # Check indexed items
                items = indexing_db.get_all_indexed_items()
                
                # Group by type
                by_type = {}
                for item in items:
                    item_type = item['item_type']
                    if item_type not in by_type:
                        by_type[item_type] = 0
                    by_type[item_type] += 1
                    
                self.results["database_checks"]["rag_indexing"] = {
                    "exists": True,
                    "total_indexed": len(items),
                    "by_type": by_type
                }
                
                logger.info(f"RAG Indexing DB: ✅ {len(items)} indexed items")
                for item_type, count in by_type.items():
                    logger.info(f"  - {item_type}: {count} items")
            else:
                self.results["database_checks"]["rag_indexing"] = {"exists": False}
                logger.info("RAG Indexing DB: Not found (will be created on first index)")
                
        except Exception as e:
            self.results["database_checks"]["rag_indexing"] = {"error": str(e)}
            logger.error(f"RAG Indexing check error: {e}")
            
    def generate_report(self):
        """Generate a comprehensive test report."""
        logger.info("\n=== Test Report ===")
        
        report = []
        report.append("# RAG Testing Report\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"USE_MODULAR_RAG: {os.environ.get('USE_MODULAR_RAG', 'false')}\n")
        
        # Dependencies
        report.append("\n## Dependencies\n")
        all_deps_ok = True
        for dep, info in self.results["dependencies"].items():
            status = info["status"]
            if status == "❌":
                all_deps_ok = False
            report.append(f"- {dep}: {status} {info['description']}\n")
            
        # Databases
        report.append("\n## Databases\n")
        for db, info in self.results["database_checks"].items():
            report.append(f"\n### {db}\n")
            if isinstance(info, dict):
                for key, value in info.items():
                    report.append(f"- {key}: {value}\n")
            else:
                report.append(f"- Status: {info}\n")
                
        # Search Tests
        report.append("\n## Search Tests\n")
        for query, result in self.results["search_tests"].items():
            report.append(f"\n### Query: '{query}'\n")
            if result.get("success"):
                report.append(f"- Results: {result['result_count']}\n")
                report.append(f"- Time: {result['elapsed_time']:.2f}s\n")
                report.append(f"- Sources: {result.get('sources', [])}\n")
            else:
                report.append(f"- Error: {result.get('error')}\n")
                
        # Chunking
        report.append("\n## Chunking Comparison\n")
        if self.results["chunking_tests"]:
            for key, value in self.results["chunking_tests"].items():
                report.append(f"- {key}: {value}\n")
                
        # Performance
        report.append("\n## Performance\n")
        if self.results["performance"]:
            for key, value in self.results["performance"].items():
                report.append(f"\n### {key}\n")
                if isinstance(value, dict):
                    for k, v in value.items():
                        report.append(f"- {k}: {v}\n")
                else:
                    report.append(f"- {value}\n")
                    
        # Recommendations
        report.append("\n## Recommendations\n")
        
        if not all_deps_ok:
            report.append("1. Install missing dependencies:\n")
            report.append("   ```bash\n")
            report.append("   pip install -e \".[embeddings_rag]\"\n")
            report.append("   ```\n")
            
        if not self.results["database_checks"].get("media_db", {}).get("record_count"):
            report.append("2. Ingest some media files to test with real data\n")
            
        if self.results.get("chunking_tests", {}).get("recommendation"):
            report.append(f"3. {self.results['chunking_tests']['recommendation']}\n")
            
        if os.environ.get('USE_MODULAR_RAG', '').lower() != 'true':
            report.append("4. Test with modular RAG: `USE_MODULAR_RAG=true python test_rag_comprehensive.py`\n")
            
        # Save report
        report_path = Path("RAG_TEST_REPORT.md")
        report_path.write_text("".join(report))
        logger.success(f"\nReport saved to: {report_path}")
        
        # Save JSON results
        json_path = Path("rag_test_results.json")
        json_path.write_text(json.dumps(self.results, indent=2))
        logger.info(f"JSON results saved to: {json_path}")
        
async def main():
    """Run all tests."""
    tester = RAGTester()
    
    try:
        # Run all tests
        await tester.check_dependencies()
        await tester.check_databases()
        await tester.check_rag_indexing()
        await tester.test_basic_search()
        await tester.test_chunking_comparison()
        await tester.test_modular_rag()
        
    except Exception as e:
        logger.error(f"Test suite error: {e}")
        tester.results["errors"].append(str(e))
        
    finally:
        # Generate report
        tester.generate_report()

if __name__ == "__main__":
    asyncio.run(main())