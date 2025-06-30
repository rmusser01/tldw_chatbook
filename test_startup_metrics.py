#!/usr/bin/env python3
"""
Quick test script to verify startup metrics are being logged.
Run this after the main app to see if metrics were captured.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tldw_chatbook.Metrics.metrics import _metrics_registry

def display_startup_metrics():
    """Display all startup-related metrics that were captured."""
    print("=== STARTUP METRICS CAPTURED ===")
    print(f"Total metrics registered: {len(_metrics_registry)}")
    print("")
    
    # Look for startup-related metrics
    startup_metrics = []
    for key, metric in _metrics_registry.items():
        metric_type, name, labels = key
        if 'startup' in name or 'app_' in name or 'compose' in name or 'mount' in name:
            startup_metrics.append((metric_type, name, labels))
    
    if startup_metrics:
        print(f"Found {len(startup_metrics)} startup-related metrics:")
        for metric_type, name, labels in sorted(startup_metrics):
            print(f"  - {metric_type}: {name}")
            if labels:
                print(f"    Labels: {labels}")
    else:
        print("No startup metrics found. Make sure to run the app first.")
    
    print("\n=== METRIC VALUES ===")
    # Try to display some values (this is limited without a prometheus server)
    for key, metric in _metrics_registry.items():
        metric_type, name, labels = key
        if 'startup' in name or 'app_' in name:
            print(f"\n{name} ({metric_type}):")
            try:
                # For counters and gauges, we can try to get the value
                if hasattr(metric, '_value'):
                    print(f"  Value: {metric._value}")
                elif hasattr(metric, '_metrics'):
                    # For metrics with labels
                    for label_values, labeled_metric in metric._metrics.items():
                        if hasattr(labeled_metric, '_value'):
                            print(f"  {dict(zip(labels, label_values))}: {labeled_metric._value}")
                        elif hasattr(labeled_metric, '_sum') and hasattr(labeled_metric, '_count'):
                            # Histogram
                            if labeled_metric._count > 0:
                                avg = labeled_metric._sum / labeled_metric._count
                                print(f"  {dict(zip(labels, label_values))}: sum={labeled_metric._sum:.3f}, count={labeled_metric._count}, avg={avg:.3f}")
            except Exception as e:
                print(f"  Could not read value: {e}")

if __name__ == "__main__":
    print("This script displays metrics captured during app startup.")
    print("Note: Metrics are only available in the same Python process.")
    print("For production use, metrics should be exported to Prometheus.")
    print("")
    display_startup_metrics()