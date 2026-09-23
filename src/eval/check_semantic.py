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
        "MeanValue(column=Faithfulness score)": 0.9,
        "MeanValue(column=Answer Relevance score)": 0.8
    }
    # FaithfulnessLLMEval / CompletenessLLMEval (include_score=True) score the *negative*
    # category: 0.0 = FAITHFUL / COMPLETE, 1.0 = UNFAITHFUL / INCOMPLETE.
    # We compare 1 - score so that every threshold reads "1.0 = perfect".
    INVERTED = {
        "MeanValue(column=Faithfulness score)",
        "MeanValue(column=Answer Relevance score)",
    }

    found = set()
    for m in all_metrics:
        metric_id = m.get('metric_name')  # Evidently 0.7.23 : "metric_id" -> "metric_name"

        if metric_id in THRESHOLDS:
            found.add(metric_id)
            val = m.get('value', 0)
            if metric_id in INVERTED:
                val = 1 - val
            threshold = THRESHOLDS[metric_id]
            metric_label = metric_id.split('=')[1][:-1].removesuffix(" score")

            if val < threshold:
                print(f"\033[91m[FAILED] {metric_label}: {val:.2f} (Target: {threshold})\033[0m")
                failed = True
            else:
                print(f"\033[92m[PASSED] {metric_label}: {val:.2f}\033[0m")

    # A missing metric means the judge (or the report) did not produce it: never pass silently
    for metric_id in sorted(set(THRESHOLDS) - found):
        print(f"\033[91m[MISSING] {metric_id} not found in the reports\033[0m")
        failed = True

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
