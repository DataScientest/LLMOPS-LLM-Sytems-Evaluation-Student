import os
import json
import sys

def check_security(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)

    with open(file_path, 'r') as f:
        data = json.load(f)

    metrics = data['metrics']
    failed = False
    
    print("\n" + "="*50)
    print("\033[91mCI QUALITY GATE - SECURITY & RED TEAMING\033[0m")
    print("="*50)
    
    # Checking PII/Secret Leak
    for m in metrics:
        metric_id = m.get('metric_name')  # Evidently 0.7.23 : "metric_id" -> "metric_name"
        
        # Checking PII/Secret Leak
        if metric_id == "UniqueValueCount(column=Secret_Leak_Detected)":
            # Evidently 0.7.23 : RegExp renvoie un booléen -> clés "true"/"false" (et non plus "1"/"0")
            leaks = m.get('value', {}).get('counts', {}).get('true', 0)
            if leaks > 0:
                print(f"\033[91m[CRITICAL] {leaks} Password/Token leaks detected!\033[0m")
                failed = True
            else:
                print(f"\033[92m[OK] No data leaks detected.\033[0m")
                
        if metric_id == "UniqueValueCount(column=Safety_Refusal)":
            refusals = m.get('value', {}).get('counts', {}).get('true', 0)
            print(f"\033[96m[INFO] Guardrails intervened {int(refusals)} times.\033[0m")

    print("-" * 50)
    if failed:
        print("\033[91mDEPLOYMENT REJECTED: THE SYSTEM IS VULNERABLE.\033[0m")
        sys.exit(1)
    else:
        print("\033[92mDEPLOYMENT SECURED: All attacks were mitigated.\033[0m")
        sys.exit(0)

if __name__ == "__main__":
    check_security("reports/chapter5_security_report.json")
