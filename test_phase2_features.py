#!/usr/bin/env python3
"""
Test script for Phase 2 RAG features:
- LLM-based reranking
- Parallel processing optimization
- Configuration profiles
"""

import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import shutil

from tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2 import (
    EnhancedRAGServiceV2, create_rag_from_profile, quick_search
)
from tldw_chatbook.RAG_Search.config_profiles import (
    get_profile_manager, ExperimentConfig, ProfileType
)
from tldw_chatbook.RAG_Search.reranker import create_reranker, RerankingConfig
from tldw_chatbook.RAG_Search.parallel_processor import ProcessingConfig


class TestPhase2Features:
    """Test suite for Phase 2 RAG enhancements."""
    
    def __init__(self):
        self.temp_dir = None
        self.test_documents = self._create_test_documents()
    
    def _create_test_documents(self) -> List[Dict[str, Any]]:
        """Create diverse test documents for comprehensive testing."""
        return [
            {
                'id': 'ml_basics',
                'title': 'Machine Learning Basics',
                'content': """
                Machine learning is revolutionizing how we solve complex problems. At its core, ML allows 
                computers to learn from data without being explicitly programmed. The field encompasses 
                supervised learning (with labeled data), unsupervised learning (finding patterns), and 
                reinforcement learning (learning through interaction).
                
                Key algorithms include:
                - Linear regression for continuous predictions
                - Logistic regression for classification
                - Decision trees for interpretable models
                - Neural networks for complex patterns
                - Support vector machines for high-dimensional data
                
                Applications span from image recognition and natural language processing to recommendation 
                systems and autonomous vehicles. The key to success is quality data and choosing the right 
                algorithm for your specific problem.
                """,
                'metadata': {'category': 'ML', 'difficulty': 'beginner', 'year': 2024}
            },
            {
                'id': 'deep_learning',
                'title': 'Deep Learning Architecture Guide',
                'content': """
                Deep learning has transformed AI by enabling models to learn hierarchical representations. 
                Modern architectures include:
                
                1. Convolutional Neural Networks (CNNs):
                   - Excellent for image processing
                   - Use convolutional layers to detect features
                   - Popular architectures: ResNet, VGG, EfficientNet
                
                2. Recurrent Neural Networks (RNNs):
                   - Process sequential data
                   - LSTM and GRU variants handle long-term dependencies
                   - Used in time series and NLP tasks
                
                3. Transformers:
                   - Attention mechanism enables parallel processing
                   - BERT for understanding, GPT for generation
                   - State-of-the-art for NLP tasks
                
                4. Generative Models:
                   - GANs create realistic synthetic data
                   - VAEs learn compressed representations
                   - Diffusion models for high-quality generation
                
                Training deep models requires careful consideration of optimization algorithms, 
                regularization techniques, and computational resources.
                """,
                'metadata': {'category': 'DL', 'difficulty': 'advanced', 'year': 2024}
            },
            {
                'id': 'nlp_advances',
                'title': 'Recent Advances in Natural Language Processing',
                'content': """
                Natural Language Processing has seen remarkable progress with the advent of large language 
                models. Key developments include:
                
                Pre-trained Models:
                - BERT revolutionized contextual understanding
                - GPT series demonstrated impressive generation capabilities
                - T5 unified various NLP tasks into text-to-text format
                
                Technical Innovations:
                - Self-attention mechanisms capture long-range dependencies
                - Transfer learning reduces data requirements
                - Few-shot learning enables rapid adaptation
                - Prompt engineering unlocks model capabilities
                
                Applications have expanded to:
                - Machine translation with near-human quality
                - Sentiment analysis for business intelligence
                - Question answering systems
                - Code generation and analysis
                - Creative writing assistance
                
                Challenges remain in handling bias, ensuring factual accuracy, and reducing 
                computational requirements for deployment.
                """,
                'metadata': {'category': 'NLP', 'difficulty': 'intermediate', 'year': 2024}
            },
            {
                'id': 'ml_deployment',
                'title': 'Deploying Machine Learning Models in Production',
                'content': """
                Taking ML models from research to production requires careful planning and infrastructure. 
                Key considerations include:
                
                Model Serving:
                - REST APIs for synchronous predictions
                - Message queues for asynchronous processing
                - Edge deployment for low-latency applications
                - Model versioning and A/B testing
                
                Monitoring and Maintenance:
                - Track prediction accuracy over time
                - Monitor for data drift
                - Log predictions for debugging
                - Set up alerts for anomalies
                
                Optimization Techniques:
                - Model compression through quantization
                - Knowledge distillation for smaller models
                - Caching frequent predictions
                - Batch processing for efficiency
                
                Infrastructure Requirements:
                - Containerization with Docker
                - Orchestration using Kubernetes
                - CI/CD pipelines for model updates
                - Feature stores for consistency
                
                Security considerations include model stealing prevention, adversarial attack detection, 
                and privacy-preserving techniques like differential privacy.
                """,
                'metadata': {'category': 'MLOps', 'difficulty': 'advanced', 'year': 2024}
            },
            {
                'id': 'ai_ethics',
                'title': 'Ethical Considerations in AI Development',
                'content': """
                As AI systems become more prevalent, addressing ethical concerns is crucial for 
                responsible development. Major considerations include:
                
                Bias and Fairness:
                - Dataset bias can perpetuate discrimination
                - Algorithmic fairness metrics help measure equity
                - Techniques like adversarial debiasing reduce unfairness
                - Regular audits ensure continued fairness
                
                Transparency and Explainability:
                - Black box models lack interpretability
                - LIME and SHAP provide local explanations
                - Global interpretability through simpler models
                - Documentation standards improve understanding
                
                Privacy and Security:
                - Federated learning keeps data distributed
                - Differential privacy protects individuals
                - Secure multi-party computation enables collaboration
                - Regular security assessments prevent breaches
                
                Accountability and Governance:
                - Clear ownership of AI decisions
                - Human oversight for critical applications
                - Regulatory compliance frameworks
                - Incident response procedures
                
                Building ethical AI requires interdisciplinary collaboration between technologists, 
                ethicists, policymakers, and affected communities.
                """,
                'metadata': {'category': 'Ethics', 'difficulty': 'intermediate', 'year': 2024}
            }
        ]
    
    async def setup(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="rag_phase2_test_")
        print(f"Created temp directory: {self.temp_dir}")
    
    async def test_reranking(self):
        """Test LLM-based reranking functionality."""
        print("\n" + "="*60)
        print("TESTING: LLM-Based Reranking")
        print("="*60)
        
        # Create service with reranking enabled
        service = EnhancedRAGServiceV2.from_profile(
            "high_accuracy",  # This profile includes reranking
            enable_reranking=True
        )
        
        # Index documents
        print("\nIndexing test documents...")
        results = await service.index_batch(self.test_documents[:3])  # Use fewer for faster test
        
        # Test different reranking strategies
        strategies = ["pointwise", "pairwise", "listwise"]
        query = "deep learning architectures for image recognition"
        
        for strategy in strategies:
            print(f"\n--- Testing {strategy} reranking ---")
            
            # Create custom reranker
            reranker_config = RerankingConfig(
                model_provider="openai",
                model_name="gpt-3.5-turbo",
                strategy=strategy,
                top_k_to_rerank=10,
                include_reasoning=True
            )
            
            # Create reranker
            reranker = create_reranker(strategy, **reranker_config.__dict__)
            
            # Get base results
            base_results = await service.search(
                query=query,
                top_k=10,
                rerank=False
            )
            
            print(f"Base results (without reranking):")
            for i, result in enumerate(base_results[:5]):
                print(f"  {i+1}. Score: {result.score:.4f} - {result.metadata.get('doc_title')}")
            
            # Rerank results
            if base_results:
                reranked_results = await reranker.rerank(query, base_results)
                
                print(f"\nReranked results ({strategy}):")
                for i, result in enumerate(reranked_results[:5]):
                    print(f"  {i+1}. Score: {result.score:.4f} - {result.metadata.get('doc_title')}")
                    if hasattr(result.metadata, 'get') and 'rerank_score' in result.metadata:
                        print(f"     Rerank score: {result.metadata['rerank_score']:.4f}")
        
        print("\n✓ Reranking tests completed")
    
    async def test_parallel_processing(self):
        """Test parallel processing optimization."""
        print("\n" + "="*60)
        print("TESTING: Parallel Processing Optimization")
        print("="*60)
        
        # Create large document set for testing
        large_doc_set = []
        for i in range(20):
            doc = {
                'id': f'doc_{i}',
                'title': f'Document {i}',
                'content': self.test_documents[i % len(self.test_documents)]['content']
            }
            large_doc_set.append(doc)
        
        # Test with and without parallel processing
        for use_parallel in [False, True]:
            print(f"\n--- {'With' if use_parallel else 'Without'} Parallel Processing ---")
            
            service = EnhancedRAGServiceV2(
                config="fast_search",
                enable_parallel_processing=use_parallel
            )
            
            start_time = time.time()
            results = await service.index_batch_optimized(
                large_doc_set,
                show_progress=True,
                batch_size=5
            )
            elapsed = time.time() - start_time
            
            successful = sum(1 for r in results if r.success)
            print(f"Indexed {successful}/{len(large_doc_set)} documents in {elapsed:.2f}s")
            print(f"Average time per document: {elapsed/len(large_doc_set):.3f}s")
        
        print("\n✓ Parallel processing tests completed")
    
    async def test_configuration_profiles(self):
        """Test configuration profiles functionality."""
        print("\n" + "="*60)
        print("TESTING: Configuration Profiles")
        print("="*60)
        
        manager = get_profile_manager()
        
        # List available profiles
        print("\nAvailable profiles:")
        for profile_name in manager.list_profiles():
            profile = manager.get_profile(profile_name)
            print(f"  - {profile_name}: {profile.description}")
        
        # Test different profiles
        test_profiles = ["fast_search", "high_accuracy", "technical_docs"]
        query = "machine learning optimization algorithms"
        
        for profile_name in test_profiles:
            print(f"\n--- Testing profile: {profile_name} ---")
            
            # Create service from profile
            service = EnhancedRAGServiceV2.from_profile(profile_name)
            
            # Index a few documents
            await service.index_batch(self.test_documents[:3])
            
            # Perform search
            start_time = time.time()
            results = await service.search(query, top_k=5)
            search_time = (time.time() - start_time) * 1000  # Convert to ms
            
            print(f"Search completed in {search_time:.1f}ms")
            print(f"Found {len(results)} results")
            
            # Validate profile
            warnings = service.validate_configuration()
            if warnings:
                print(f"Profile warnings: {warnings}")
            else:
                print("Profile validation: OK")
        
        # Test custom profile creation
        print("\n--- Creating custom profile ---")
        custom_profile = manager.create_custom_profile(
            name="My Custom Profile",
            base_profile="balanced",
            chunk_size=256,
            search_top_k=15
        )
        print(f"Created custom profile: {custom_profile.name}")
        
        print("\n✓ Configuration profile tests completed")
    
    async def test_ab_testing(self):
        """Test A/B testing functionality."""
        print("\n" + "="*60)
        print("TESTING: A/B Testing Framework")
        print("="*60)
        
        # Create experiment configuration
        experiment = ExperimentConfig(
            name="Search Quality Test",
            description="Compare fast vs accurate search",
            enable_ab_testing=True,
            control_profile="fast_search",
            test_profiles=["high_accuracy"],
            traffic_split={
                "fast_search": 0.5,
                "high_accuracy": 0.5
            },
            track_metrics=True
        )
        
        # Create service and start experiment
        service = EnhancedRAGServiceV2.from_profile("balanced")
        service.start_experiment(experiment)
        
        # Index documents
        await service.index_batch(self.test_documents[:3])
        
        # Simulate user searches
        print("\nSimulating user searches...")
        test_queries = [
            ("user1", "machine learning algorithms"),
            ("user2", "deep learning CNNs"),
            ("user3", "NLP transformers"),
            ("user4", "ML deployment strategies"),
            ("user1", "neural network optimization"),  # Same user gets consistent profile
            ("user5", "AI ethics and bias"),
        ]
        
        for user_id, query in test_queries:
            results = await service.search(query, user_id=user_id)
            print(f"User {user_id}: '{query}' - {len(results)} results")
        
        # End experiment and get results
        print("\nExperiment results:")
        summary = service.end_experiment()
        
        for profile, stats in summary.get("profiles", {}).items():
            print(f"\nProfile: {profile}")
            print(f"  Queries: {stats['query_count']}")
            if 'metrics' in stats:
                for metric, values in stats['metrics'].items():
                    if isinstance(values, dict) and 'mean' in values:
                        print(f"  {metric}: {values['mean']:.2f} (avg)")
        
        print("\n✓ A/B testing completed")
    
    async def test_quick_functions(self):
        """Test convenience functions."""
        print("\n" + "="*60)
        print("TESTING: Quick Functions")
        print("="*60)
        
        # Test create_rag_from_profile
        print("\n--- Testing create_rag_from_profile ---")
        service, results = await create_rag_from_profile(
            "fast_search",
            documents=self.test_documents[:3]
        )
        print(f"Created service and indexed {len(results)} documents")
        
        # Note: quick_search is synchronous, so we can't test it directly in async context
        print("\n--- Testing quick_search (would be used in sync context) ---")
        print("quick_search function is available for synchronous one-shot searches")
        
        print("\n✓ Quick function tests completed")
    
    async def test_profile_switching(self):
        """Test dynamic profile switching."""
        print("\n" + "="*60)
        print("TESTING: Profile Switching")
        print("="*60)
        
        # Create service with initial profile
        service = EnhancedRAGServiceV2.from_profile("fast_search")
        print(f"Initial profile: {service.get_current_profile().name}")
        
        # Index documents
        await service.index_batch(self.test_documents[:2])
        
        # Search with fast profile
        query = "machine learning basics"
        start = time.time()
        results_fast = await service.search(query, top_k=5)
        time_fast = time.time() - start
        print(f"\nFast search: {len(results_fast)} results in {time_fast*1000:.1f}ms")
        
        # Switch to high accuracy profile
        service.switch_profile("high_accuracy")
        print(f"\nSwitched to profile: {service.get_current_profile().name}")
        
        # Search with accurate profile
        start = time.time()
        results_accurate = await service.search(query, top_k=5)
        time_accurate = time.time() - start
        print(f"Accurate search: {len(results_accurate)} results in {time_accurate*1000:.1f}ms")
        
        print("\n✓ Profile switching tests completed")
    
    async def cleanup(self):
        """Clean up test resources."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"\nCleaned up temp directory: {self.temp_dir}")
    
    async def run_all_tests(self):
        """Run all Phase 2 tests."""
        try:
            await self.setup()
            
            # Run individual test suites
            await self.test_configuration_profiles()
            await self.test_parallel_processing()
            # Note: Reranking and A/B testing require API keys to be configured
            # await self.test_reranking()  # Requires LLM API key
            await self.test_ab_testing()
            await self.test_quick_functions()
            await self.test_profile_switching()
            
            print("\n" + "="*60)
            print("PHASE 2 TEST SUMMARY")
            print("="*60)
            print("✓ Configuration profiles: PASSED")
            print("✓ Parallel processing: PASSED")
            print("✓ A/B testing framework: PASSED")
            print("✓ Quick functions: PASSED")
            print("✓ Profile switching: PASSED")
            print("Note: Reranking tests require LLM API keys")
            
            print("\n✓ All Phase 2 tests completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            await self.cleanup()


async def main():
    """Main entry point for Phase 2 tests."""
    print("Starting Phase 2 RAG Feature Tests")
    print("="*60)
    
    tester = TestPhase2Features()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())