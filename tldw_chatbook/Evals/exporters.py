# exporters.py
# Description: Unified export functionality for evaluation results
#
"""
Evaluation Exporters
--------------------

Provides unified export functionality for all evaluation results:
- CSV export for tabular data
- JSON export with complete details
- Markdown report generation
- LaTeX table generation
- Support for both standard runs and A/B test results
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from loguru import logger


class EvaluationExporter:
    """Unified exporter for all evaluation result types."""
    
    def export(
        self,
        result: Any,
        output_path: Union[str, Path],
        format: str = 'csv',
        options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Export evaluation results in specified format.
        
        Args:
            result: Result object (ABTestResult or standard run data)
            output_path: Path to save the exported file
            format: Export format ('csv', 'json', 'markdown', 'latex')
            options: Export options specific to format
        """
        output_path = Path(output_path)
        options = options or {}
        
        # Determine result type and dispatch accordingly
        if hasattr(result, 'test_id') and hasattr(result, 'model_a_name'):
            # ABTestResult
            self._export_ab_test(result, output_path, format, options)
        else:
            # Standard evaluation run
            self._export_standard_run(result, output_path, format, options)
    
    def _export_ab_test(
        self,
        result: Any,  # ABTestResult
        output_path: Path,
        format: str,
        options: Dict[str, Any]
    ) -> None:
        """Export A/B test results."""
        if format == 'csv':
            self._export_ab_test_csv(result, output_path)
        elif format == 'json':
            self._export_ab_test_json(result, output_path, options)
        elif format == 'markdown':
            self._export_ab_test_markdown(result, output_path)
        elif format == 'latex':
            self._export_ab_test_latex(result, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_standard_run(
        self,
        run_data: Dict[str, Any],
        output_path: Path,
        format: str,
        options: Dict[str, Any]
    ) -> None:
        """Export standard evaluation run results."""
        if format == 'csv':
            self._export_run_csv(run_data, output_path, options)
        elif format == 'json':
            self._export_run_json(run_data, output_path, options)
        elif format == 'markdown':
            self._export_run_markdown(run_data, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    # A/B Test Export Methods
    
    def _export_ab_test_csv(self, result: Any, output_path: Path) -> None:
        """Export A/B test results to CSV."""
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
            
            # Sample results if available
            if hasattr(result, 'sample_results') and result.sample_results:
                writer.writerow([])
                writer.writerow(['Sample Results (First 10)'])
                writer.writerow(['Sample ID', 'Input', 'Model A Output', 'Model B Output', 'Model A Score', 'Model B Score'])
                
                for i, sample in enumerate(result.sample_results[:10]):
                    writer.writerow([
                        sample.get('id', i),
                        sample.get('input', '')[:100],  # Truncate long inputs
                        sample.get('model_a_output', '')[:100],
                        sample.get('model_b_output', '')[:100],
                        sample.get('model_a_score', ''),
                        sample.get('model_b_score', '')
                    ])
        
        logger.info(f"Exported A/B test results to CSV: {output_path}")
    
    def _export_ab_test_json(self, result: Any, output_path: Path, options: Dict[str, Any]) -> None:
        """Export A/B test results to JSON."""
        include_raw = options.get('include_raw_outputs', False)
        
        data = {
            'test_id': result.test_id,
            'test_name': result.test_name,
            'timestamp': result.timestamp,
            'configuration': {
                'model_a': result.model_a_name,
                'model_b': result.model_b_name,
                'sample_size': result.sample_size,
                'significance_level': getattr(result, 'significance_level', 0.05)
            },
            'metrics': {
                'model_a': result.model_a_metrics,
                'model_b': result.model_b_metrics
            },
            'statistical_tests': result.statistical_tests,
            'performance': {
                'model_a_latency_ms': result.model_a_latency,
                'model_b_latency_ms': result.model_b_latency,
                'model_a_cost_usd': result.model_a_cost,
                'model_b_cost_usd': result.model_b_cost
            },
            'conclusion': {
                'winner': result.winner,
                'confidence': result.confidence if hasattr(result, 'confidence') else None,
                'recommendations': result.recommendations if hasattr(result, 'recommendations') else []
            }
        }
        
        if include_raw and hasattr(result, 'sample_results'):
            data['sample_results'] = result.sample_results
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported A/B test results to JSON: {output_path}")
    
    def _export_ab_test_markdown(self, result: Any, output_path: Path) -> None:
        """Export A/B test results to Markdown report."""
        lines = []
        
        # Header
        lines.append(f"# A/B Test Report: {result.test_name}")
        lines.append(f"\n**Test ID:** {result.test_id}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("## Executive Summary")
        lines.append("")
        winner_text = f"**{result.winner}**" if result.winner else "No significant difference"
        lines.append(f"- **Winner:** {winner_text}")
        lines.append(f"- **Sample Size:** {result.sample_size}")
        lines.append(f"- **Models Tested:** {result.model_a_name} vs {result.model_b_name}")
        lines.append("")
        
        # Metrics Comparison
        lines.append("## Metrics Comparison")
        lines.append("")
        lines.append("| Metric | Model A | Model B | Difference | P-value | Significant |")
        lines.append("|--------|---------|---------|------------|---------|-------------|")
        
        for metric in result.model_a_metrics:
            if metric in result.model_b_metrics and metric in result.statistical_tests:
                value_a = result.model_a_metrics[metric]
                value_b = result.model_b_metrics[metric]
                test = result.statistical_tests[metric]
                
                value_a_str = f"{value_a:.4f}" if isinstance(value_a, float) else str(value_a)
                value_b_str = f"{value_b:.4f}" if isinstance(value_b, float) else str(value_b)
                diff_str = f"{test.get('difference', 0):.4f}"
                p_value_str = f"{test.get('p_value', 1):.4f}"
                sig_str = "✓" if test.get('is_significant', False) else "✗"
                
                lines.append(f"| {metric} | {value_a_str} | {value_b_str} | {diff_str} | {p_value_str} | {sig_str} |")
        
        lines.append("")
        
        # Performance Comparison
        lines.append("## Performance Comparison")
        lines.append("")
        lines.append("| Metric | Model A | Model B | Better |")
        lines.append("|--------|---------|---------|---------|")
        
        latency_better = "Model A" if result.model_a_latency < result.model_b_latency else "Model B"
        cost_better = "Model A" if result.model_a_cost < result.model_b_cost else "Model B"
        
        lines.append(f"| Latency (ms) | {result.model_a_latency:.2f} | {result.model_b_latency:.2f} | {latency_better} |")
        lines.append(f"| Cost ($) | {result.model_a_cost:.4f} | {result.model_b_cost:.4f} | {cost_better} |")
        lines.append("")
        
        # Statistical Analysis
        lines.append("## Statistical Analysis")
        lines.append("")
        
        if hasattr(result, 'statistical_summary'):
            for key, value in result.statistical_summary.items():
                lines.append(f"- **{key}:** {value}")
        else:
            lines.append("- Statistical tests performed using appropriate methods for each metric")
            lines.append("- Significance level: α = 0.05")
        
        lines.append("")
        
        # Recommendations
        if hasattr(result, 'recommendations'):
            lines.append("## Recommendations")
            lines.append("")
            for rec in result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported A/B test report to Markdown: {output_path}")
    
    def _export_ab_test_latex(self, result: Any, output_path: Path) -> None:
        """Export A/B test results to LaTeX table."""
        lines = []
        
        # LaTeX document header
        lines.append("\\documentclass{article}")
        lines.append("\\usepackage{booktabs}")
        lines.append("\\usepackage{array}")
        lines.append("\\begin{document}")
        lines.append("")
        
        # Title
        lines.append(f"\\section*{{A/B Test Results: {result.test_name}}}")
        lines.append("")
        
        # Metrics table
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\caption{Metrics Comparison}")
        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append("Metric & Model A & Model B & Difference & P-value & Significant \\\\")
        lines.append("\\midrule")
        
        for metric in result.model_a_metrics:
            if metric in result.model_b_metrics and metric in result.statistical_tests:
                value_a = result.model_a_metrics[metric]
                value_b = result.model_b_metrics[metric]
                test = result.statistical_tests[metric]
                
                value_a_str = f"{value_a:.4f}" if isinstance(value_a, float) else str(value_a)
                value_b_str = f"{value_b:.4f}" if isinstance(value_b, float) else str(value_b)
                diff_str = f"{test.get('difference', 0):.4f}"
                p_value_str = f"{test.get('p_value', 1):.4f}"
                sig_str = "Yes" if test.get('is_significant', False) else "No"
                
                # Escape underscores in metric names for LaTeX
                metric_escaped = metric.replace('_', '\\_')
                
                lines.append(f"{metric_escaped} & {value_a_str} & {value_b_str} & {diff_str} & {p_value_str} & {sig_str} \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")
        
        lines.append("\\end{document}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported A/B test results to LaTeX: {output_path}")
    
    # Standard Run Export Methods
    
    def _export_run_csv(
        self,
        run_data: Dict[str, Any],
        output_path: Path,
        options: Dict[str, Any]
    ) -> None:
        """Export standard evaluation run to CSV."""
        try:
            results = run_data.get('results', [])
            
            if not results:
                # Create a summary CSV
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Run ID', 'Model', 'Task', 'Status', 'Total Cost', 'Samples'])
                    writer.writerow([
                        run_data.get('run_id', ''),
                        run_data.get('model', ''),
                        run_data.get('task', ''),
                        run_data.get('status', ''),
                        run_data.get('total_cost', 0),
                        run_data.get('completed_samples', 0)
                    ])
            else:
                # Create detailed results CSV
                with open(output_path, 'w', newline='') as f:
                    # Determine columns based on first result
                    first_result = results[0] if results else {}
                    columns = list(first_result.keys())
                    
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(results)
            
            logger.info(f"Exported evaluation run to CSV: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            raise
    
    def _export_run_json(
        self,
        run_data: Dict[str, Any],
        output_path: Path,
        options: Dict[str, Any]
    ) -> None:
        """Export standard evaluation run to JSON."""
        try:
            # Filter data based on options
            output_data = run_data.copy()
            
            if not options.get('include_raw_outputs', False):
                # Remove raw outputs if not requested
                if 'results' in output_data:
                    for result in output_data['results']:
                        result.pop('raw_output', None)
            
            if not options.get('include_metadata', True):
                # Remove metadata if not requested
                output_data.pop('metadata', None)
            
            # Write JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Exported evaluation run to JSON: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            raise
    
    def _export_run_markdown(
        self,
        run_data: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Export standard evaluation run to Markdown report."""
        lines = []
        
        # Header
        lines.append(f"# Evaluation Report")
        lines.append(f"\n**Run ID:** {run_data.get('run_id', 'N/A')}")
        lines.append(f"**Model:** {run_data.get('model', 'N/A')}")
        lines.append(f"**Task:** {run_data.get('task', 'N/A')}")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Status:** {run_data.get('status', 'Unknown')}")
        lines.append(f"- **Samples Evaluated:** {run_data.get('completed_samples', 0)}")
        lines.append(f"- **Total Cost:** ${run_data.get('total_cost', 0):.4f}")
        lines.append(f"- **Duration:** {run_data.get('duration_seconds', 0):.2f} seconds")
        lines.append("")
        
        # Metrics
        if 'metrics' in run_data:
            lines.append("## Metrics")
            lines.append("")
            
            metrics = run_data['metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    lines.append(f"- **{metric_name}:** {metric_value:.4f}")
                else:
                    lines.append(f"- **{metric_name}:** {metric_value}")
            lines.append("")
        
        # Sample Results
        if 'results' in run_data and run_data['results']:
            lines.append("## Sample Results (First 5)")
            lines.append("")
            
            for i, result in enumerate(run_data['results'][:5], 1):
                lines.append(f"### Sample {i}")
                lines.append(f"**Input:** {result.get('input', 'N/A')[:200]}...")
                lines.append(f"**Output:** {result.get('output', 'N/A')[:200]}...")
                if 'score' in result:
                    lines.append(f"**Score:** {result['score']}")
                lines.append("")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Exported evaluation report to Markdown: {output_path}")


# Convenience functions for backward compatibility

def export_to_csv(run_data: Dict[str, Any], output_path: Path, options: Dict[str, Any]) -> None:
    """Legacy function for CSV export."""
    exporter = EvaluationExporter()
    exporter.export(run_data, output_path, 'csv', options)


def export_to_json(run_data: Dict[str, Any], output_path: Path, options: Dict[str, Any]) -> None:
    """Legacy function for JSON export."""
    exporter = EvaluationExporter()
    exporter.export(run_data, output_path, 'json', options)