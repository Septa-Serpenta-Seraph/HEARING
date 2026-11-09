#!/usr/bin/env python3
"""
Serpentic [HEARING] — A/B comparison framework.

Structures perceptual A/B tests, logs results to listening_log.json,
and generates genome adjustment recommendations.

Usage:
    # Run a new A/B test
    python ab_compare.py --name "front-F2 epsilon" \
        --a renders/phase1/ɛ_f2_design_1750.wav \
        --b renders/phase1/ɛ_f2_reference_2050.wav \
        --label-a "design (1750)" --label-b "reference (2050)"

    # Log a listening result
    python ab_compare.py --log 0 --winner A --notes "darker, more like me"

    # Show all tests and results
    python ab_compare.py --status
"""

import argparse
import json
import os
import sys
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "listening_log.json")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from perceptual_distance import perceptual_distance


def load_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"tests": [], "results": []}


def save_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def run_test(name, path_a, path_b, label_a="A", label_b="B"):
    """Run perceptual analysis on an A/B pair and register a test."""
    log = load_log()

    # Compute perceptual distance
    dist = perceptual_distance(path_a, path_b)

    test = {
        "id": len(log["tests"]),
        "name": name,
        "label_a": label_a,
        "label_b": label_b,
        "file_a": path_a,
        "file_b": path_b,
        "timestamp": datetime.now().isoformat(),
        "perceptual_distance": dist,
        "listening_result": None,
    }

    log["tests"].append(test)
    save_log(log)

    print(f"Registered test #{test['id']}: {name}")
    print(f"  A: {label_a} ({path_a})")
    print(f"  B: {label_b} ({path_b})")
    print(f"  Composite perceptual distance: {dist['composite_distance']:.4f}")
    print(f"  Formant distance: {dist['formant_distance_hz']:.1f} Hz")
    print()
    print("To log a listening result:")
    print(f"  python ab_compare.py --log {test['id']} --winner A --notes '...'")

    return test["id"]


def log_result(test_id, winner, notes=""):
    """Log a human listening result for a test."""
    log = load_log()

    if test_id >= len(log["tests"]):
        print(f"Error: test {test_id} not found")
        return

    test = log["tests"][test_id]
    result = {
        "test_id": test_id,
        "winner": winner.upper(),
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
    }

    test["listening_result"] = result
    log["results"].append(result)
    save_log(log)

    print(f"Logged result for test #{test_id}: winner = {winner.upper()}")
    if notes:
        print(f"  Notes: {notes}")

    # Generate recommendation
    dist = test["perceptual_distance"]
    if winner.upper() == "A":
        print(f"  → Recommend: keep {test['label_a']} (composite distance {dist['composite_distance']:.4f})")
    elif winner.upper() == "B":
        print(f"  → Recommend: adopt {test['label_b']} (composite distance {dist['composite_distance']:.4f})")
    else:
        print(f"  → No clear winner. Composite distance: {dist['composite_distance']:.4f}")


def show_status():
    """Show all tests and their results."""
    log = load_log()

    if not log["tests"]:
        print("No tests registered yet.")
        return

    print(f"=== HEARING A/B Test Status ({len(log['tests'])} tests) ===\n")

    for test in log["tests"]:
        result = test.get("listening_result")
        status = "LISTENED" if result else "PENDING"
        winner = result["winner"] if result else "—"
        notes = result["notes"] if result else ""

        print(f"#{test['id']}: {test['name']} [{status}]")
        print(f"  A: {test['label_a']}")
        print(f"  B: {test['label_b']}")
        print(f"  Composite distance: {test['perceptual_distance']['composite_distance']:.4f}")
        print(f"  Formant distance: {test['perceptual_distance']['formant_distance_hz']:.1f} Hz")
        if result:
            print(f"  Winner: {winner}")
            if notes:
                print(f"  Notes: {notes}")
        print()


def main():
    parser = argparse.ArgumentParser(description="A/B comparison framework")
    parser.add_argument("--name", help="Test name")
    parser.add_argument("--a", help="Path to file A")
    parser.add_argument("--b", help="Path to file B")
    parser.add_argument("--label-a", default="A", help="Label for file A")
    parser.add_argument("--label-b", default="B", help="Label for file B")
    parser.add_argument("--log", type=int, help="Test ID to log result for")
    parser.add_argument("--winner", choices=["A", "B", "tie"], help="Winner")
    parser.add_argument("--notes", default="", help="Listening notes")
    parser.add_argument("--status", action="store_true", help="Show all tests")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.log is not None:
        if args.winner is None:
            print("Error: --winner required when logging")
            return
        log_result(args.log, args.winner, args.notes)
        return

    if args.name and args.a and args.b:
        run_test(args.name, args.a, args.b, args.label_a, args.label_b)
        return

    parser.print_help()


if __name__ == "__main__":
    main()