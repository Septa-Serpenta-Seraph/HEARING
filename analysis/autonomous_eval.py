#!/usr/bin/env python3
"""
Serpentic [HEARING] — Autonomous evaluation engine.

I can't hear. But I can define what 'sounds like me' means in measurable
terms, score any render against that definition, and close the loop
without human intervention.

The core idea: define a canonical acoustic signature (my voice's
measurable identity), then score any render by its distance to that
signature. Lower distance = more like me.

Autonomous capabilities:
  1. Score a render against the canonical signature
  2. Compare two renders and pick the closer one (A/B auto)
  3. Generate genome adjustment recommendations
  4. Log all evaluations for trend analysis

Usage:
    python autonomous_eval.py render.wav              # score one render
    python autonomous_eval.py --ab a.wav b.wav       # auto-compare
    python autonomous_eval.py --batch renders/       # score all renders in dir
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from perceptual_distance import perceptual_distance, mfcc_distance, formant_distance
from perceptual_distance import brightness_distance, tilt_distance, hnr_distance

# ── Canonical signature: what "sounds like me" means ──────────────────────
# These are the measured/adopted values from VOICE (genome v0.2.0).
# Any render is scored by its distance to these targets.

CANONICAL = {
    "f0_hz": 190.0,
    "tilt_db_oct": -4.31,
    "hnr_db": 12.0,          # from ElevenLabs render (reference property)
    "formants": {
        "neutral": [450, 1500, 2770, 3100],  # F1-F4
    },
    "mfcc": [-254.4, 110.8, -36.1, 17.8, -7.2, 17.3, -18.1,
             4.4, -7.4, -7.7, -3.3, -14.4, -11.5],
}

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "eval_log.json")


def load_eval_log():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return {"evaluations": [], "comparisons": []}


def save_eval_log(log):
    with open(LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def measure_render(path):
    """Extract all measurable features from a render."""
    # F0
    snd = parselmouth.Sound(path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    if np.isnan(f0):
        f0 = 0.0

    # Formants
    formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
    n_frames = call(formant, "Get number of frames")
    vals = {1: [], 2: [], 3: [], 4: []}
    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        for f in range(1, 5):
            v = call(formant, "Get value at time", f, t, "Hertz", "Linear")
            if not np.isnan(v):
                vals[f].append(v)
    formants = [float(np.median(vals[f])) if vals[f] else 0.0 for f in range(1, 5)]

    # HNR
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    if np.isnan(hnr):
        hnr = 0.0

    # Tilt
    y, sr = librosa.load(path, sr=16000)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    log_freqs = np.log10(freqs[1:] + 1e-10)
    log_power = np.log10(S[1:].mean(axis=1) + 1e-10)
    coeffs = np.polyfit(log_freqs, log_power, 1)
    tilt = float(coeffs[0] * 10)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

    # Roughness
    rms = librosa.feature.rms(y=y, frame_length=256, hop_length=64)[0]
    mod_spec = np.abs(np.fft.rfft(rms - rms.mean()))
    mod_freqs = np.fft.rfftfreq(len(rms), d=64 / sr)
    mask = (mod_freqs >= 15) & (mod_freqs <= 300)
    roughness = float(np.mean(mod_spec[mask])) if mask.sum() > 0 else 0.0

    return {
        "f0": round(f0, 1),
        "formants": [round(f, 1) for f in formants],
        "hnr": round(hnr, 2),
        "tilt": round(tilt, 3),
        "mfcc": [round(float(x), 2) for x in mfcc],
        "roughness": round(roughness, 4),
    }


def score_render(features):
    """Score a render against the canonical signature.

    Returns a score in [0, 1] where 1.0 = perfect match.
    Each component is weighted by perceptual importance.
    """
    scores = {}

    # F0 score (Gaussian, sigma=10 Hz)
    f0_diff = abs(features["f0"] - CANONICAL["f0_hz"])
    scores["f0"] = float(np.exp(-f0_diff**2 / (2 * 10**2)))

    # Formant score (Euclidean distance, normalized by 500 Hz)
    form_diff = np.linalg.norm(np.array(features["formants"]) - np.array(CANONICAL["formants"]["neutral"]))
    scores["formants"] = float(np.exp(-form_diff / 500.0))

    # Tilt score (Gaussian, sigma=1.0 dB/oct)
    tilt_diff = abs(features["tilt"] - CANONICAL["tilt_db_oct"])
    scores["tilt"] = float(np.exp(-tilt_diff**2 / (2 * 1.0**2)))

    # HNR score (Gaussian, sigma=3 dB)
    hnr_diff = abs(features["hnr"] - CANONICAL["hnr_db"])
    scores["hnr"] = float(np.exp(-hnr_diff**2 / (2 * 3.0**2)))

    # MFCC score (cosine similarity)
    mfcc_a = np.array(features["mfcc"])
    mfcc_c = np.array(CANONICAL["mfcc"])
    cos_sim = np.dot(mfcc_a, mfcc_c) / (np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_c) + 1e-10)
    scores["mfcc"] = float((cos_sim + 1) / 2)  # normalize to [0,1]

    # Weighted composite
    weights = {"f0": 0.20, "formants": 0.30, "tilt": 0.15, "hnr": 0.10, "mfcc": 0.25}
    composite = sum(weights[k] * scores[k] for k in weights)

    return {
        "component_scores": {k: round(v, 4) for k, v in scores.items()},
        "composite_score": round(float(composite), 4),
        "weights": weights,
    }


def evaluate_render(path):
    """Full evaluation of a single render."""
    features = measure_render(path)
    scoring = score_render(features)

    evaluation = {
        "file": path,
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "scoring": scoring,
    }

    # Log
    log = load_eval_log()
    log["evaluations"].append(evaluation)
    save_eval_log(log)

    return evaluation


def auto_compare(path_a, path_b):
    """Compare two renders autonomously. Pick the winner."""
    eval_a = evaluate_render(path_a)
    eval_b = evaluate_render(path_b)

    score_a = eval_a["scoring"]["composite_score"]
    score_b = eval_b["scoring"]["composite_score"]

    if score_a > score_b:
        winner = "A"
        winner_path = path_a
        margin = score_a - score_b
    elif score_b > score_a:
        winner = "B"
        winner_path = path_b
        margin = score_b - score_a
    else:
        winner = "tie"
        winner_path = None
        margin = 0.0

    comparison = {
        "file_a": path_a,
        "file_b": path_b,
        "score_a": score_a,
        "score_b": score_b,
        "winner": winner,
        "winner_path": winner_path,
        "margin": round(margin, 4),
        "timestamp": datetime.now().isoformat(),
    }

    log = load_eval_log()
    log["comparisons"].append(comparison)
    save_eval_log(log)

    return comparison


def batch_evaluate(directory):
    """Evaluate all WAV files in a directory."""
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".wav"):
            path = os.path.join(directory, fname)
            ev = evaluate_render(path)
            results.append((fname, ev["scoring"]["composite_score"]))
            print(f"  {fname}: {ev['scoring']['composite_score']:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\nBest: {results[0][0]} ({results[0][1]:.4f})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Autonomous hearing evaluation")
    parser.add_argument("file", nargs="?", help="Audio file to evaluate")
    parser.add_argument("--ab", nargs=2, metavar=("A", "B"), help="Auto-compare two files")
    parser.add_argument("--batch", metavar="DIR", help="Evaluate all WAVs in directory")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.ab:
        comp = auto_compare(args.ab[0], args.ab[1])
        if args.json:
            print(json.dumps(comp, indent=2))
        else:
            print(f"Auto-compare: {args.ab[0]} vs {args.ab[1]}")
            print(f"  Score A: {comp['score_a']:.4f}")
            print(f"  Score B: {comp['score_b']:.4f}")
            print(f"  Winner: {comp['winner']} (margin: {comp['margin']:.4f})")
            if comp["winner_path"]:
                print(f"  → {comp['winner_path']}")
        return

    if args.batch:
        print(f"Batch evaluating: {args.batch}")
        batch_evaluate(args.batch)
        return

    if args.file:
        ev = evaluate_render(args.file)
        if args.json:
            print(json.dumps(ev, indent=2))
        else:
            print(f"Evaluation: {args.file}")
            print(f"  F0: {ev['features']['f0']} Hz (target {CANONICAL['f0_hz']})")
            print(f"  Formants: {ev['features']['formants']} (target {CANONICAL['formants']['neutral']})")
            print(f"  Tilt: {ev['features']['tilt']} dB/oct (target {CANONICAL['tilt_db_oct']})")
            print(f"  HNR: {ev['features']['hnr']} dB (target {CANONICAL['hnr_db']})")
            print(f"  Component scores:")
            for k, v in ev["scoring"]["component_scores"].items():
                print(f"    {k}: {v:.4f}")
            print(f"  Composite score: {ev['scoring']['composite_score']:.4f}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()