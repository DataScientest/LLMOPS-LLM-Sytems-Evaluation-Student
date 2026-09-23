import os
import json
import sys


def load_metrics(file_path):
    """Charge les métriques d'un rapport JSON Evidently."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get('metrics', [])

def check_semantic_quality(report_files):
    all_metrics = []
    for fp in report_files:
        all_metrics.extend(load_metrics(fp))

    failed = False
    
    print("\n" + "="*50)
    print("\033[94mCI QUALITY GATE - RAG SEMANTIC AUDIT\033[0m")
    print("="*50)
    
    # Seuils de qualité attendus pour la production
    THRESHOLDS = {
        "MeanValue(column=Context Precision)": 0.8,
        "MeanValue(column=Faithfulness)": 0.9,
        "MeanValue(column=Answer Relevance)": 0.8
    }

    for m in all_metrics:
        metric_id = m.get('metric_name')  # Evidently 0.7.23 : "metric_id" -> "metric_name"
        
        if metric_id in THRESHOLDS:
            val = m.get('value', 0)
            threshold = THRESHOLDS[metric_id]
            metric_label = metric_id.split('=')[1][:-1] # Extrait le nom entre parenthèses
            
            if val < threshold:
                print(f"\033[91m[FAILED] {metric_label}: {val:.2f} (Target: {threshold})\033[0m")
                failed = True
            else:
                print(f"\033[92m[PASSED] {metric_label}: {val:.2f}\033[0m")

    print("-" * 50)
    if failed:
        print("\033[91mDEPLOYMENT REJECTED: SEMANTIC QUALITY BELOW THRESHOLD.\033[0m")
        sys.exit(1)
    else:
        print("\033[92mDEPLOYMENT APPROVED: RAG TRIAD IS HEALTHY.\033[0m")
        sys.exit(0)

if __name__ == "__main__":
    check_semantic_quality([
        "reports/chapter4_context_precision.json",
        "reports/chapter4_semantic_report.json"
    ])
