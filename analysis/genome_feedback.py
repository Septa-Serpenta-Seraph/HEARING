#!/usr/bin/env python3
"""
Serpentic [HEARING] — Genome feedback from listening results.

Reads listening_log.json and generates genome adjustment recommendations
for the VOICE genome based on human listening results.

Usage:
    python genome_feedback.py                    # analyze all results
    python genome_feedback.py --apply            # (future) apply to genome
"""

import argparse
import json
import os
import sys

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "listening_log.json")
VOICE_GENOME = None  # set via --voice-dir or env


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"tests": [], "results": []}


def analyze_results(log):
    """Analyze listening results and generate recommendations."""
    recommendations = []

    for test in log["tests"]:
        result = test.get("listening_result")
        if not result:
            continue

        winner = result["winner"]
        name = test["name"]
        label_a = test["label_a"]
        label_b = test["label_b"]
        dist = test["perceptual_distance"]

        if "front-F2" in name or "f2" in name.lower():
            if winner == "A":
                recommendations.append({
                    "parameter": "F2 (front vowels)",
                    "action": "keep current genome values",
                    "reason": f"Listener preferred {label_a} (composite distance {dist['composite_distance']:.4f})",
                    "test": name,
                })
            elif winner == "B":
                recommendations.append({
                    "parameter": "F2 (front vowels)",
                    "action": "update genome F2 to reference values",
                    "reason": f"Listener preferred {label_b} (composite distance {dist['composite_distance']:.4f})",
                    "test": name,
                })
            else:
                recommendations.append({
                    "parameter": "F2 (front vowels)",
                    "action": "no clear preference — keep current",
                    "reason": f"Tie (composite distance {dist['composite_distance']:.4f})",
                    "test": name,
                })

    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Genome feedback from listening results")
    parser.add_argument("--voice-dir", default=None,
                        help="Path to VOICE repo (for genome reference)")
    parser.add_argument("--apply", action="store_true", help="Apply recommendations (not yet implemented)")
    args = parser.parse_args()

    log = load_log()
    recs = analyze_results(log)

    if not recs:
        print("No listening results to analyze yet.")
        print("Run A/B tests and log results first:")
        print("  python ab_compare.py --status")
        return

    print("=== Genome Feedback Recommendations ===\n")
    for rec in recs:
        print(f"Parameter: {rec['parameter']}")
        print(f"  Action: {rec['action']}")
        print(f"  Reason: {rec['reason']}")
        print(f"  Test: {rec['test']}")
        print()

    if args.apply:
        print("Apply not yet implemented — edit genome manually based on recommendations.")


if __name__ == "__main__":
    main()