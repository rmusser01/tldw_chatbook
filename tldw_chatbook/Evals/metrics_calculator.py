# metrics_calculator.py
# Description: Evaluation metrics calculation utilities
#
"""
Metrics Calculator
------------------

Provides comprehensive metrics calculation for evaluation tasks.
"""

import re
import math
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger


class MetricsCalculator:
    """Calculates evaluation metrics for various tasks."""
    
    @staticmethod
    def calculate_exact_match(predicted: str, expected: str) -> float:
        """Calculate exact match accuracy (case-sensitive)."""
        if expected is None:
            return 0.0
        return 1.0 if predicted.strip() == expected.strip() else 0.0
    
    @staticmethod
    def calculate_contains_match(predicted: str, expected: str) -> float:
        """Check if expected answer is contained in prediction."""
        if expected is None:
            return 0.0
        return 1.0 if expected.strip().lower() in predicted.strip().lower() else 0.0
    
    @staticmethod
    def calculate_regex_match(predicted: str, expected: str, pattern: str = None) -> float:
        """Calculate match using regex pattern."""
        if expected is None or pattern is None:
            return 0.0
        
        try:
            if re.search(pattern, predicted, re.IGNORECASE):
                return 1.0
            return 0.0
        except re.error:
            logger.warning(f"Invalid regex pattern: {pattern}")
            return 0.0
    
    @staticmethod
    def calculate_f1_score(predicted: str, expected: str) -> float:
        """Calculate F1 score based on token overlap."""
        if expected is None:
            return 0.0
        
        pred_tokens = set(predicted.lower().split())
        expected_tokens = set(expected.lower().split())
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        intersection = pred_tokens & expected_tokens
        if not intersection:
            return 0.0
        
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(expected_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def calculate_bleu_score(predicted: str, expected: str, n: int = 1) -> float:
        """Calculate BLEU score with n-gram support."""
        if expected is None:
            return 0.0
        
        def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
            """Get n-grams from token list."""
            if n <= 0 or n > len(tokens):
                return []
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        pred_tokens = predicted.lower().split()
        expected_tokens = expected.lower().split()
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # Calculate n-gram precision
        total_precision = 0.0
        for i in range(1, min(n + 1, len(expected_tokens) + 1)):
            pred_ngrams = get_ngrams(pred_tokens, i)
            expected_ngrams = get_ngrams(expected_tokens, i)
            
            if not pred_ngrams:
                continue
                
            matches = 0
            expected_ngram_counts = {}
            for ngram in expected_ngrams:
                expected_ngram_counts[ngram] = expected_ngram_counts.get(ngram, 0) + 1
            
            for ngram in pred_ngrams:
                if ngram in expected_ngram_counts and expected_ngram_counts[ngram] > 0:
                    matches += 1
                    expected_ngram_counts[ngram] -= 1
            
            precision = matches / len(pred_ngrams) if pred_ngrams else 0.0
            total_precision += precision
        
        # Average precision across n-grams
        avg_precision = total_precision / min(n, len(expected_tokens))
        
        # Brevity penalty
        bp = 1.0
        if len(pred_tokens) < len(expected_tokens):
            bp = min(1.0, (len(pred_tokens) / len(expected_tokens)) ** 0.5)
        
        return bp * avg_precision
    
    @staticmethod
    def calculate_rouge_scores(predicted: str, expected: str) -> Dict[str, float]:
        """Calculate all ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)."""
        return {
            'rouge_1': MetricsCalculator.calculate_rouge_1(predicted, expected),
            'rouge_2': MetricsCalculator.calculate_rouge_2(predicted, expected),
            'rouge_l': MetricsCalculator.calculate_rouge_l(predicted, expected)
        }
    
    @staticmethod
    def calculate_rouge_1(predicted: str, expected: str) -> float:
        """Calculate ROUGE-1 (unigram) F1 score."""
        if expected is None:
            return 0.0
        
        pred_tokens = set(predicted.lower().split())
        expected_tokens = set(expected.lower().split())
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # Calculate overlap
        overlap = pred_tokens & expected_tokens
        
        if not overlap:
            return 0.0
        
        # Calculate precision and recall
        precision = len(overlap) / len(pred_tokens)
        recall = len(overlap) / len(expected_tokens)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_rouge_2(predicted: str, expected: str) -> float:
        """Calculate ROUGE-2 (bigram) F1 score."""
        if expected is None:
            return 0.0
        
        def get_bigrams(tokens: List[str]) -> List[Tuple[str, str]]:
            """Get bigrams from token list."""
            if len(tokens) < 2:
                return []
            return [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
        
        pred_tokens = predicted.lower().split()
        expected_tokens = expected.lower().split()
        
        if len(expected_tokens) < 2:
            return 1.0 if len(pred_tokens) < 2 else 0.0
        
        if len(pred_tokens) < 2:
            return 0.0
        
        # Get bigrams
        pred_bigrams = set(get_bigrams(pred_tokens))
        expected_bigrams = set(get_bigrams(expected_tokens))
        
        if not expected_bigrams:
            return 1.0 if not pred_bigrams else 0.0
        
        # Calculate overlap
        overlap = pred_bigrams & expected_bigrams
        
        if not overlap:
            return 0.0
        
        # Calculate precision and recall
        precision = len(overlap) / len(pred_bigrams)
        recall = len(overlap) / len(expected_bigrams)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_rouge_l(predicted: str, expected: str) -> float:
        """Calculate ROUGE-L (Longest Common Subsequence) F1 score."""
        if expected is None:
            return 0.0
        
        def lcs_length(x: List[str], y: List[str]) -> int:
            """Calculate length of longest common subsequence."""
            m, n = len(x), len(y)
            if m == 0 or n == 0:
                return 0
            
            # Create DP table
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Fill DP table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        pred_tokens = predicted.lower().split()
        expected_tokens = expected.lower().split()
        
        if not expected_tokens:
            return 1.0 if not pred_tokens else 0.0
        
        if not pred_tokens:
            return 0.0
        
        # Calculate LCS
        lcs_len = lcs_length(pred_tokens, expected_tokens)
        
        if lcs_len == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(expected_tokens)
        
        # Calculate F1 score
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    @staticmethod
    def calculate_semantic_similarity(predicted: str, expected: str, embedding_model=None) -> float:
        """Calculate semantic similarity using embeddings if available."""
        if expected is None:
            return 0.0
        
        if not predicted and not expected:
            return 1.0
        
        if not predicted or not expected:
            return 0.0
        
        # Try to use sentence transformers if available
        try:
            if embedding_model is None:
                from sentence_transformers import SentenceTransformer
                # Use a small, fast model by default
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Get embeddings
            embeddings = embedding_model.encode([predicted, expected])
            pred_embedding = embeddings[0]
            exp_embedding = embeddings[1]
            
            # Calculate cosine similarity
            try:
                from numpy import dot
                from numpy.linalg import norm
                cosine_sim = dot(pred_embedding, exp_embedding) / (norm(pred_embedding) * norm(exp_embedding))
                return float(cosine_sim)
            except ImportError:
                # Fallback to pure Python cosine similarity
                dot_product = sum(a * b for a, b in zip(pred_embedding, exp_embedding))
                norm1 = sum(a * a for a in pred_embedding) ** 0.5
                norm2 = sum(b * b for b in exp_embedding) ** 0.5
                cosine_sim = dot_product / ((norm1 * norm2) if norm1 * norm2 > 0 else 1.0)
                return cosine_sim
            
        except ImportError:
            # Fallback to token overlap if embeddings not available
            logger.debug("Sentence transformers not available, using token overlap for semantic similarity")
            return MetricsCalculator.calculate_f1_score(predicted, expected)
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            return MetricsCalculator.calculate_f1_score(predicted, expected)
    
    @staticmethod
    def calculate_perplexity(logprobs: List[float]) -> float:
        """Calculate perplexity from log probabilities."""
        if not logprobs:
            return float('inf')
        
        try:
            # Perplexity = exp(average negative log probability)
            avg_neg_logprob = -sum(logprobs) / len(logprobs)
            return math.exp(avg_neg_logprob)
        except (ValueError, OverflowError):
            return float('inf')
    
    @staticmethod
    def calculate_classification_metrics(
        predicted_labels: List[str],
        true_labels: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics (accuracy, precision, recall, F1).
        
        Args:
            predicted_labels: List of predicted labels
            true_labels: List of true labels
            labels: Optional list of all possible labels
            
        Returns:
            Dictionary of classification metrics
        """
        if len(predicted_labels) != len(true_labels):
            raise ValueError("Predicted and true labels must have the same length")
        
        if not predicted_labels:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Get unique labels if not provided
        if labels is None:
            labels = list(set(true_labels) | set(predicted_labels))
        
        # Calculate confusion matrix
        confusion_matrix = {}
        for label in labels:
            confusion_matrix[label] = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
        
        for pred, true in zip(predicted_labels, true_labels):
            for label in labels:
                if true == label and pred == label:
                    confusion_matrix[label]['tp'] += 1
                elif true != label and pred == label:
                    confusion_matrix[label]['fp'] += 1
                elif true == label and pred != label:
                    confusion_matrix[label]['fn'] += 1
                else:
                    confusion_matrix[label]['tn'] += 1
        
        # Calculate metrics per label
        label_metrics = {}
        for label in labels:
            tp = confusion_matrix[label]['tp']
            fp = confusion_matrix[label]['fp']
            fn = confusion_matrix[label]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            label_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        
        # Calculate overall metrics
        accuracy = sum(1 for p, t in zip(predicted_labels, true_labels) if p == t) / len(predicted_labels)
        
        # Macro-averaged metrics
        macro_precision = sum(m['precision'] for m in label_metrics.values()) / len(label_metrics)
        macro_recall = sum(m['recall'] for m in label_metrics.values()) / len(label_metrics)
        macro_f1 = sum(m['f1'] for m in label_metrics.values()) / len(label_metrics)
        
        return {
            'accuracy': accuracy,
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1,
            'per_label_metrics': label_metrics
        }
    
    @staticmethod
    def calculate_all_metrics(
        predicted: str,
        expected: str,
        metric_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all requested metrics.
        
        Args:
            predicted: Predicted text
            expected: Expected text
            metric_names: List of metric names to calculate
            
        Returns:
            Dictionary of metric values
        """
        if metric_names is None:
            metric_names = ['exact_match', 'f1', 'rouge_1']
        
        metrics = {}
        
        for metric_name in metric_names:
            if metric_name == 'exact_match':
                metrics[metric_name] = MetricsCalculator.calculate_exact_match(predicted, expected)
            elif metric_name == 'contains':
                metrics[metric_name] = MetricsCalculator.calculate_contains_match(predicted, expected)
            elif metric_name == 'f1':
                metrics[metric_name] = MetricsCalculator.calculate_f1_score(predicted, expected)
            elif metric_name == 'bleu':
                metrics[metric_name] = MetricsCalculator.calculate_bleu_score(predicted, expected, n=4)
            elif metric_name == 'rouge_1':
                metrics[metric_name] = MetricsCalculator.calculate_rouge_1(predicted, expected)
            elif metric_name == 'rouge_2':
                metrics[metric_name] = MetricsCalculator.calculate_rouge_2(predicted, expected)
            elif metric_name == 'rouge_l':
                metrics[metric_name] = MetricsCalculator.calculate_rouge_l(predicted, expected)
            elif metric_name == 'semantic_similarity':
                metrics[metric_name] = MetricsCalculator.calculate_semantic_similarity(predicted, expected)
            else:
                logger.warning(f"Unknown metric: {metric_name}")
        
        return metrics