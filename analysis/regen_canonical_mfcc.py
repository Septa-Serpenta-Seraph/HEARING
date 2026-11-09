#!/usr/bin/env python3
"""
HEARING — Canonical MFCC regeneration through HEARING's own pipeline.

Regenerates the canonical MFCC fingerprint from synthetic voice renders
using HEARING's measure_render() function (librosa.feature.mfcc, sr=16000,
n_mfcc=13, mean over frames).

Usage:
    python regen_canonical_mfcc.py --renders-dir /path/to/renders
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from autonomous_eval import measure_render

# Path to VOICE renders
DEFAULT_RENDERS = os.path.join(
    os.path.dirname(__file__), "..", "..", "narusya-voice", "renders", "phase1"
)


def regen(renders_dir):
    """Regenerate canonical MFCC from renders."""
    # Find vowel renders
    vowel_files = []
    for f in sorted(os.listdir(renders_dir)):
        if f.endswith(".wav") and not f.startswith("fricative") and not f.startswith("nasal"):
            # Skip A/B pairs and perturbed renders
            if "_f2_" in f or "perturbed" in f:
                continue
            vowel_files.append(os.path.join(renders_dir, f))

    print(f"Found {len(vowel_files)} vowel renders")

    mfccs = []
    for vf in vowel_files:
        try:
            features = measure_render(vf)
            mfccs.append(features["mfcc"])
            print(f"  {os.path.basename(vf)}: MFCC[0]={features['mfcc'][0]:.1f}")
        except Exception as e:
            print(f"  {os.path.basename(vf)}: ERROR {e}")

    if not mfccs:
        print("No MFCCs collected.")
        return None

    mfccs = np.array(mfccs)
    # Median across vowels (robust to outliers)
    canonical_mfcc = np.median(mfccs, axis=0)

    print()
    print("Canonical MFCC (median across vowels):")
    print(f"  {[round(float(x), 1) for x in canonical_mfcc]}")
    print(f"  (original stale: [-254.4, 110.8, -36.1, 17.8, -7.2, 17.3, -18.1, 4.4, -7.4, -7.7, -3.3, -14.4, -11.5])")

    return [round(float(x), 1) for x in canonical_mfcc]


def main():
    parser = argparse.ArgumentParser(description="Regenerate canonical MFCC")
    parser.add_argument("--renders-dir", default=DEFAULT_RENDERS)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    mfcc = regen(args.renders_dir)
    if mfcc is None:
        sys.exit(1)

    if args.json:
        print(json.dumps({"canonical_mfcc": mfcc}, indent=2))


if __name__ == "__main__":
    main()