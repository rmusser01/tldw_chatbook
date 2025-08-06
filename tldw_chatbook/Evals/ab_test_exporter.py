# ab_test_exporter.py
# Description: Export functionality for A/B test results
#
"""
A/B Test Exporter
-----------------

Provides export functionality for A/B test results:
- CSV export with statistical analysis
- JSON export with full details
- Markdown report generation
- LaTeX table generation
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from loguru import logger

from .ab_testing import ABTestResult

class ABTestExporter:
    """Exports A/B test results in various formats."""
    
    def export_to_csv(self, result: ABTestResult, output_path: Union[str, Path]) -> None:
        """
        Export A/B test results to CSV format.
        
        Args:
            result: A/B test result to export
            output_path: Path to save the CSV file
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['A/B Test Results Export'])
            writer.writerow(['Generated:', datetime.now().isoformat()])
            writer.writerow([])
            
            # Test information
            writer.writerow(['Test Information'])
            writer.writerow(['Test ID:', result.test_id])
            writer.writerow(['Test Name:', result.test_name])
            writer.writerow(['Model A:', result.model_a_name])
            writer.writerow(['Model B:', result.model_b_name])
            writer.writerow(['Sample Size:', result.sample_size])
            writer.writerow(['Winner:', result.winner or 'No significant difference'])
            writer.writerow([])
            
            # Metrics comparison
            writer.writerow(['Metrics Comparison'])
            writer.writerow(['Metric', 'Model A', 'Model B', 'Difference', 'P-value', 'Significant'])
            
            for metric in result.model_a_metrics:
                if metric in result.model_b_metrics and metric in result.statistical_tests:
                    value_a = result.model_a_metrics[metric]
                    value_b = result.model_b_metrics[metric]
                    test = result.statistical_tests[metric]
                    
                    writer.writerow([
                        metric,
                        f"{value_a:.4f}" if isinstance(value_a, float) else value_a,
                        f"{value_b:.4f}" if isinstance(value_b, float) else value_b,
                        f"{test.get('difference', 0):.4f}",
                        f"{test.get('p_value', 1):.4f}",
                        'Yes' if test.get('is_significant', False) else 'No'
                    ])
            
            writer.writerow([])
            
            # Performance metrics
            writer.writerow(['Performance Metrics'])
            writer.writerow(['Metric', 'Model A', 'Model B'])
            writer.writerow(['Latency (ms)', f"{result.model_a_latency:.2f}", f"{result.model_b_latency:.2f}"])
            writer.writerow(['Cost ($)', f"{result.model_a_cost:.4f}", f"{result.model_b_cost:.4f}"])
            
            writer.writerow([])
            
            # Sample-level results (first 100)
            writer.writerow(['Sample Results (First 100)'])
            writer.writerow(['Sample', 'Input', 'Expected', 'Model A Output', 'Model B Output', 'Model A Correct', 'Model B Correct'])
            
            for i, sample in enumerate(result.sample_results[:100]):
                writer.writerow([
                    i + 1,
                    sample.get('input', '')[:100],  # Truncate long inputs
                    sample.get('expected', '')[:50],
                    sample.get('model_a_output', '')[:50],
                    sample.get('model_b_output', '')[:50],
                    'Yes' if sample.get('model_a_correct') else 'No',
                    'Yes' if sample.get('model_b_correct') else 'No'
                ])
        
        logger.info(f"Exported A/B test results to CSV: {output_path}")
    
    def export_to_json(self, result: ABTestResult, output_path: Union[str, Path]) -> None:
        """
        Export A/B test results to JSON format.
        
        Args:
            result: A/B test result to export
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        
        export_data = {
            'test_id': result.test_id,
            'test_name': result.test_name,
            'model_a_name': result.model_a_name,
            'model_b_name': result.model_b_name,
            'sample_size': result.sample_size,
            'winner': result.winner,
            'model_a_metrics': result.model_a_metrics,
            'model_b_metrics': result.model_b_metrics,
            'statistical_tests': result.statistical_tests,
            'confidence_intervals': result.confidence_intervals,
            'model_a_latency': result.model_a_latency,
            'model_b_latency': result.model_b_latency,
            'model_a_cost': result.model_a_cost,
            'model_b_cost': result.model_b_cost,
            'sample_results': result.sample_results,
            'started_at': result.started_at.isoformat() if result.started_at else None,
            'completed_at': result.completed_at.isoformat() if result.completed_at else None,
            'metadata': result.metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported A/B test results to JSON: {output_path}")
    
    def export_to_markdown(self, result: ABTestResult, output_path: Union[str, Path]) -> None:
        """
        Export A/B test results to Markdown report format.
        
        Args:
            result: A/B test result to export
            output_path: Path to save the Markdown file
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write(f"# A/B Test Results: {result.test_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Model A:** {result.model_a_name}\n")
            f.write(f"- **Model B:** {result.model_b_name}\n")
            f.write(f"- **Sample Size:** {result.sample_size}\n")
            f.write(f"- **Winner:** {result.winner or 'No significant difference'}\n\n")
            
            # Metrics Comparison
            f.write("## Metrics Comparison\n\n")
            f.write("| Metric | Model A | Model B | Difference | P-value | Significant |\n")
            f.write("|--------|---------|---------|------------|---------|-------------|\n")
            
            for metric in result.model_a_metrics:
                if metric in result.model_b_metrics and metric in result.statistical_tests:
                    value_a = result.model_a_metrics[metric]
                    value_b = result.model_b_metrics[metric]
                    test = result.statistical_tests[metric]
                    
                    f.write(f"| {metric} | ")
                    f.write(f"{value_a:.4f} | " if isinstance(value_a, float) else f"{value_a} | ")
                    f.write(f"{value_b:.4f} | " if isinstance(value_b, float) else f"{value_b} | ")
                    f.write(f"{test.get('difference', 0):.4f} | ")
                    f.write(f"{test.get('p_value', 1):.4f} | ")
                    f.write("✓ |" if test.get('is_significant', False) else "✗ |\n")
            
            f.write("\n")
            
            # Statistical Analysis
            f.write("## Statistical Analysis\n\n")
            for metric, test in result.statistical_tests.items():
                f.write(f"### {metric}\n\n")
                f.write(f"- **Mean A:** {test.get('mean_a', 0):.4f}\n")
                f.write(f"- **Mean B:** {test.get('mean_b', 0):.4f}\n")
                f.write(f"- **Std Dev A:** {test.get('std_a', 0):.4f}\n")
                f.write(f"- **Std Dev B:** {test.get('std_b', 0):.4f}\n")
                f.write(f"- **T-statistic:** {test.get('t_statistic', 0):.4f}\n")
                f.write(f"- **P-value:** {test.get('p_value', 1):.4f}\n")
                f.write(f"- **Effect Size (Cohen's d):** {test.get('effect_size', 0):.3f}\n")
                f.write(f"- **Relative Difference:** {test.get('relative_difference', 0):.1f}%\n\n")
            
            # Performance Comparison
            f.write("## Performance Comparison\n\n")
            f.write(f"- **Model A Latency:** {result.model_a_latency:.2f} ms\n")
            f.write(f"- **Model B Latency:** {result.model_b_latency:.2f} ms\n")
            f.write(f"- **Model A Cost:** ${result.model_a_cost:.4f}\n")
            f.write(f"- **Model B Cost:** ${result.model_b_cost:.4f}\n\n")
            
            # Confidence Intervals
            if result.confidence_intervals:
                f.write("## Confidence Intervals\n\n")
                for metric, (lower, upper) in result.confidence_intervals.items():
                    f.write(f"- **{metric}:** [{lower:.4f}, {upper:.4f}]\n")
                f.write("\n")
            
            # Sample Analysis
            f.write("## Sample Analysis\n\n")
            
            # Count correct predictions
            model_a_correct = sum(1 for s in result.sample_results if s.get('model_a_correct'))
            model_b_correct = sum(1 for s in result.sample_results if s.get('model_b_correct'))
            both_correct = sum(1 for s in result.sample_results if s.get('model_a_correct') and s.get('model_b_correct'))
            neither_correct = sum(1 for s in result.sample_results if not s.get('model_a_correct') and not s.get('model_b_correct'))
            
            f.write(f"- **Model A Correct:** {model_a_correct} ({model_a_correct/len(result.sample_results)*100:.1f}%)\n")
            f.write(f"- **Model B Correct:** {model_b_correct} ({model_b_correct/len(result.sample_results)*100:.1f}%)\n")
            f.write(f"- **Both Correct:** {both_correct} ({both_correct/len(result.sample_results)*100:.1f}%)\n")
            f.write(f"- **Neither Correct:** {neither_correct} ({neither_correct/len(result.sample_results)*100:.1f}%)\n\n")
            
            # Sample Examples
            f.write("## Sample Examples\n\n")
            f.write("### Cases where models disagreed:\n\n")
            
            disagreements = [s for s in result.sample_results[:20] 
                           if s.get('model_a_correct') != s.get('model_b_correct')]
            
            for i, sample in enumerate(disagreements[:5]):
                f.write(f"**Sample {i+1}:**\n")
                f.write(f"- Input: {sample.get('input', '')[:200]}...\n")
                f.write(f"- Expected: {sample.get('expected', '')}\n")
                f.write(f"- Model A: {sample.get('model_a_output', '')} ")
                f.write("(✓)\n" if sample.get('model_a_correct') else "(✗)\n")
                f.write(f"- Model B: {sample.get('model_b_output', '')} ")
                f.write("(✓)\n\n" if sample.get('model_b_correct') else "(✗)\n\n")
        
        logger.info(f"Exported A/B test results to Markdown: {output_path}")
    
    def export_to_latex(self, result: ABTestResult, output_path: Union[str, Path]) -> None:
        """
        Export A/B test results to LaTeX table format.
        
        Args:
            result: A/B test result to export
            output_path: Path to save the LaTeX file
        """
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Document header
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{amsmath}\n")
            f.write("\\begin{document}\n\n")
            
            # Title
            f.write(f"\\section{{A/B Test Results: {self._escape_latex(result.test_name)}}}\n\n")
            
            # Summary table
            f.write("\\subsection{Summary}\n")
            f.write("\\begin{tabular}{ll}\n")
            f.write("\\toprule\n")
            f.write("\\textbf{Property} & \\textbf{Value} \\\\\n")
            f.write("\\midrule\n")
            f.write(f"Model A & {self._escape_latex(result.model_a_name)} \\\\\n")
            f.write(f"Model B & {self._escape_latex(result.model_b_name)} \\\\\n")
            f.write(f"Sample Size & {result.sample_size} \\\\\n")
            f.write(f"Winner & {self._escape_latex(result.winner or 'No significant difference')} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n\n")
            
            # Metrics comparison table
            f.write("\\subsection{Metrics Comparison}\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lccccc}\n")
            f.write("\\toprule\n")
            f.write("Metric & Model A & Model B & Difference & P-value & Significant \\\\\n")
            f.write("\\midrule\n")
            
            for metric in result.model_a_metrics:
                if metric in result.model_b_metrics and metric in result.statistical_tests:
                    value_a = result.model_a_metrics[metric]
                    value_b = result.model_b_metrics[metric]
                    test = result.statistical_tests[metric]
                    
                    f.write(f"{self._escape_latex(metric)} & ")
                    f.write(f"{value_a:.4f} & " if isinstance(value_a, float) else f"{value_a} & ")
                    f.write(f"{value_b:.4f} & " if isinstance(value_b, float) else f"{value_b} & ")
                    f.write(f"{test.get('difference', 0):.4f} & ")
                    f.write(f"{test.get('p_value', 1):.4f} & ")
                    f.write("$\\checkmark$" if test.get('is_significant', False) else "$\\times$")
                    f.write(" \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            f.write("\\end{document}\n")
        
        logger.info(f"Exported A/B test results to LaTeX: {output_path}")
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        if not isinstance(text, str):
            text = str(text)
        
        replacements = {
            '\\': '\\textbackslash{}',
            '#': '\\#',
            '$': '\\$',
            '%': '\\%',
            '^': '\\^{}',
            '&': '\\&',
            '_': '\\_',
            '{': '\\{',
            '}': '\\}',
            '~': '\\textasciitilde{}',
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text