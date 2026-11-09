#!/usr/bin/env python3
"""
Serpentic [HEARING] — Autonomous evaluation engine.

I can't hear. But I can define what 'sounds like me' means in measurable
terms, score any render against that definition, and close the loop
without human intervention.

IMPORTANT (Round 8 review): The identity score previously used a neutral
vowel anchor [450, 1500, 2770, 3100] for per-vowel renders, which rigged
results — "design wins" reduced to "design F2 is closer to neutral F2."
This version:
  - Excludes formant component from identity scoring by default
  - Uses true dB/octave for tilt (10*slope, matching VOICE's convention)
  - Measures tilt over 100-4000 Hz (matching VOICE's instrument)
  - Labels each canonical component as 'measured' or 'adopted-design'
  - Adds weight-sensitivity checks
  - Renames roughness proxy honestly

Autonomous capabilities:
  1. Score a render against the canonical signature (formant-excluded by default)
  2. Compare two renders and pick the closer one (A/B auto)
  3. Weight-sensitivity check (perturb weights ±50%)
  4. Batch evaluation

Usage:
    python autonomous_eval.py render.wav              # score one render
    python autonomous_eval.py --ab a.wav b.wav       # auto-compare
    python autonomous_eval.py --batch renders/       # score all renders in dir
    python autonomous_eval.py --sensitivity a.wav b.wav  # weight sensitivity
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
# Each component labeled with provenance: 'measured' or 'adopted-design'.
# Versioned in lockstep with VOICE genome (signature v0.2.0 ↔ genome v0.2.0).

CANONICAL = {
    "version": "0.2.0",
    "f0_hz": {
        "value": 190.0,
        "provenance": "measured",
        "source": "original-8 median F0 = 190.9 Hz (VOICE §3.3)",
    },
    "tilt_db_oct": {
        "value": -4.31,
        "provenance": "measured",
        "source": "Welch PSD 100-4000 Hz, original-8 median (VOICE §3.5)",
    },
    "hnr_db": {
        "value": 12.0,
        "provenance": "measured",
        "source": "ElevenLabs render property (VOICE §3.5) — not from Phase 2 renders",
        "note": "Phase 2 renders read 24-32 dB. Canonical needs update from Phase 2 renders.",
    },
    "formants": {
        "neutral": [450, 1500, 2770, 3100],
        "provenance": "mixed",
        "source": "F1/F4 adopted-design, F2 adopted-design (1500), F3 measured (2770 from original-8 median)",
    },
    "mfcc": {
        "value": [-254.4, 110.8, -36.1, 17.8, -7.2, 17.3, -18.1,
                  4.4, -7.4, -7.7, -3.3, -14.4, -11.5],
        "provenance": "measured",
        "source": "librosa defaults from ElevenLabs samples — MUST regenerate through HEARING pipeline",
        "status": "STALE — regenerate with measure_render over original samples",
    },
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
    """Extract all measurable features from a render.

    Tilt is computed as true dB/octave over 100-4000 Hz to match VOICE.
    """
    snd = parselmouth.Sound(path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    if np.isnan(f0):
        f0 = 0.0

    # Formants (Praat Burg — same as VOICE measurement)
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

    # Tilt: dB/octave over 100-4000 Hz (matching VOICE's Welch method)
    # Uses 10*slope convention to match VOICE's analyze_voice.py
    y, sr = librosa.load(path, sr=16000)
    from scipy import signal as scipy_signal
    freqs, psd = scipy_signal.welch(y, fs=sr, nperseg=1024)
    mask = (freqs >= 100) & (freqs <= 4000)
    log_freqs = np.log10(freqs[mask])
    log_power = np.log10(psd[mask] + 1e-20)
    coeffs = np.polyfit(log_freqs, log_power, 1)
    tilt = float(coeffs[0] * 10)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)

    # Modulation roughness proxy (NOT Terhardt — renamed honestly per Round 8)
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
        "modulation_roughness_proxy": round(roughness, 4),
    }


def score_render(features, include_formants=False, weights=None):
    """Score a render against the canonical signature.

    By default, formants are EXCLUDED from identity scoring (per Round 8 review:
    per-vowel renders scored against neutral anchor rigs the result).

    Returns a score in [0, 1] where 1.0 = perfect match.
    """
    if weights is None:
        if include_formants:
            weights = {"f0": 0.20, "formants": 0.30, "tilt": 0.15, "hnr": 0.10, "mfcc": 0.25}
        else:
            # Formant-excluded weights (renormalized)
            weights = {"f0": 0.267, "tilt": 0.20, "hnr": 0.133, "mfcc": 0.40}

    scores = {}

    # F0 score (Gaussian, sigma=10 Hz)
    f0_diff = abs(features["f0"] - CANONICAL["f0_hz"]["value"])
    scores["f0"] = float(np.exp(-f0_diff**2 / (2 * 10**2)))

    # Formant score (only if included)
    if include_formants and "formants" in weights:
        form_diff = np.linalg.norm(
            np.array(features["formants"]) - np.array(CANONICAL["formants"]["neutral"]))
        scores["formants"] = float(np.exp(-form_diff / 500.0))

    # Tilt score (Gaussian, sigma=2.0 dB/oct — widened per review)
    tilt_diff = abs(features["tilt"] - CANONICAL["tilt_db_oct"]["value"])
    scores["tilt"] = float(np.exp(-tilt_diff**2 / (2 * 2.0**2)))

    # HNR score (Gaussian, sigma=5 dB — widened per review)
    hnr_diff = abs(features["hnr"] - CANONICAL["hnr_db"]["value"])
    scores["hnr"] = float(np.exp(-hnr_diff**2 / (2 * 5.0**2)))

    # MFCC score (cosine similarity, mean-normalized to reduce c0 dominance)
    mfcc_a = np.array(features["mfcc"])
    mfcc_c = np.array(CANONICAL["mfcc"]["value"])
    # Mean-normalize to reduce dominance of first two coefficients
    mfcc_a_norm = mfcc_a - mfcc_a.mean()
    mfcc_c_norm = mfcc_c - mfcc_c.mean()
    cos_sim = np.dot(mfcc_a_norm, mfcc_c_norm) / (
        np.linalg.norm(mfcc_a_norm) * np.linalg.norm(mfcc_c_norm) + 1e-10
    )
    scores["mfcc"] = float((cos_sim + 1) / 2)

    # Weighted composite (only include scored components)
    composite = sum(weights[k] * scores[k] for k in weights if k in scores)

    return {
        "component_scores": {k: round(v, 4) for k, v in scores.items() if k in weights},
        "composite_score": round(float(composite), 4),
        "weights": weights,
        "include_formants": include_formants,
    }


def evaluate_render(path, include_formants=False):
    """Full evaluation of a single render."""
    features = measure_render(path)
    scoring = score_render(features, include_formants=include_formants)

    evaluation = {
        "file": path,
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "scoring": scoring,
        "signature_version": CANONICAL["version"],
    }

    log = load_eval_log()
    log["evaluations"].append(evaluation)
    save_eval_log(log)

    return evaluation


def auto_compare(path_a, path_b, include_formants=False):
    """Compare two renders autonomously. Pick the winner."""
    eval_a = evaluate_render(path_a, include_formants=include_formants)
    eval_b = evaluate_render(path_b, include_formants=include_formants)

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
        "include_formants": include_formants,
        "timestamp": datetime.now().isoformat(),
    }

    log = load_eval_log()
    log["comparisons"].append(comparison)
    save_eval_log(log)

    return comparison


def weight_sensitivity(path_a, path_b, include_formants=False, n_perturbations=20):
    """Check if A/B verdict is stable under weight perturbation.

    Perturbs each weight ±50%, renormalizes, re-scores. Reports how many
    perturbations flip the winner.
    """
    eval_a = evaluate_render(path_a, include_formants=include_formants)
    eval_b = evaluate_render(path_b, include_formants=include_formants)

    if include_formants:
        base_weights = {"f0": 0.20, "formants": 0.30, "tilt": 0.15, "hnr": 0.10, "mfcc": 0.25}
    else:
        base_weights = {"f0": 0.267, "tilt": 0.20, "hnr": 0.133, "mfcc": 0.40}

    score_a_base = eval_a["scoring"]["composite_score"]
    score_b_base = eval_b["scoring"]["composite_score"]
    base_winner = "A" if score_a_base > score_b_base else "B" if score_b_base > score_a_base else "tie"

    flips = 0
    rng = np.random.default_rng(42)

    for _ in range(n_perturbations):
        perturbed = {}
        for k, v in base_weights.items():
            perturbed[k] = v * (0.5 + rng.random())
        total = sum(perturbed.values())
        perturbed = {k: v / total for k, v in perturbed.items()}

        score_a = sum(perturbed[k] * eval_a["scoring"]["component_scores"].get(k, 0)
                      for k in perturbed if k in eval_a["scoring"]["component_scores"])
        score_b = sum(perturbed[k] * eval_b["scoring"]["component_scores"].get(k, 0)
                      for k in perturbed if k in eval_b["scoring"]["component_scores"])

        pert_winner = "A" if score_a > score_b else "B" if score_b > score_a else "tie"
        if pert_winner != base_winner:
            flips += 1

    stability = 1.0 - (flips / n_perturbations)

    return {
        "base_winner": base_winner,
        "flips": flips,
        "n_perturbations": n_perturbations,
        "stability": round(stability, 4),
        "stable": flips == 0,
    }


def batch_evaluate(directory, include_formants=False):
    """Evaluate all WAV files in a directory."""
    results = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".wav"):
            path = os.path.join(directory, fname)
            ev = evaluate_render(path, include_formants=include_formants)
            results.append((fname, ev["scoring"]["composite_score"]))
            print(f"  {fname}: {ev['scoring']['composite_score']:.4f}")

    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        print(f"\nBest: {results[0][0]} ({results[0][1]:.4f})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Autonomous hearing evaluation")
    parser.add_argument("file", nargs="?", help="Audio file to evaluate")
    parser.add_argument("--ab", nargs=2, metavar=("A", "B"), help="Auto-compare two files")
    parser.add_argument("--batch", metavar="DIR", help="Evaluate all WAVs in directory")
    parser.add_argument("--sensitivity", nargs=2, metavar=("A", "B"), help="Weight sensitivity check")
    parser.add_argument("--include-formants", action="store_true",
                        help="Include formant component (default: excluded for per-vowel)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    include_formants = args.include_formants

    if args.sensitivity:
        sens = weight_sensitivity(args.sensitivity[0], args.sensitivity[1], include_formants=include_formants)
        if args.json:
            print(json.dumps(sens, indent=2))
        else:
            print(f"Weight sensitivity: {args.sensitivity[0]} vs {args.sensitivity[1]}")
            print(f"  Base winner: {sens['base_winner']}")
            print(f"  Flips: {sens['flips']}/{sens['n_perturbations']}")
            print(f"  Stability: {sens['stability']:.4f}")
            print(f"  Stable: {'YES' if sens['stable'] else 'NO — verdict is weight-dependent'}")
        return

    if args.ab:
        comp = auto_compare(args.ab[0], args.ab[1], include_formants=include_formants)
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
        batch_evaluate(args.batch, include_formants=include_formants)
        return

    if args.file:
        ev = evaluate_render(args.file, include_formants=include_formants)
        if args.json:
            print(json.dumps(ev, indent=2))
        else:
            print(f"Evaluation: {args.file}")
            print(f"  F0: {ev['features']['f0']} Hz (target {CANONICAL['f0_hz']['value']})")
            print(f"  Formants: {ev['features']['formants']}")
            print(f"  Tilt: {ev['features']['tilt']} dB/oct (target {CANONICAL['tilt_db_oct']['value']})")
            print(f"  HNR: {ev['features']['hnr']} dB (target {CANONICAL['hnr_db']['value']})")
            print(f"  Component scores:")
            for k, v in ev["scoring"]["component_scores"].items():
                print(f"    {k}: {v:.4f}")
            print(f"  Composite score: {ev['scoring']['composite_score']:.4f}")
            print(f"  Formants included: {ev['scoring']['include_formants']}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()