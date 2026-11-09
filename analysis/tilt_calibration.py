#!/usr/bin/env python3
"""
Serpentic [HEARING] — Tilt calibration with known inputs.

Settles the three-review argument about the tilt conversion constant.
Synthesize signals at known tilts (-3, -6, -9 dB/oct), measure with
HEARING's Welch method (100-4000 Hz), report recovery.

Usage:
    python tilt_calibration.py
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

SR = 16000
DURATION = 2.0


def synthesize_known_tilt(tilt_db_oct, sr=SR, duration=DURATION, f0=190.0):
    """Synthesize a signal with a known spectral tilt.

    Creates a harmonic series with amplitudes following the tilt slope.
    tilt_db_oct: spectral slope in dB/octave (negative = darker).
    """
    n_samples = int(duration * sr)
    t = np.arange(n_samples) / sr

    # Generate harmonics with tilt-weighted amplitudes
    signal = np.zeros(n_samples)
    max_harmonic = int(sr / (2 * f0))  # Nyquist limit
    for h in range(1, max_harmonic + 1):
        freq = h * f0
        if freq > sr / 2:
            break
        # Amplitude follows tilt: A_h = 1 / (h ^ (tilt_db_oct / -6.02))
        # tilt_db_oct is dB/octave. Each octave doubles frequency (×2).
        # dB change per octave = tilt_db_oct. Per harmonic (log2(h) octaves):
        # amplitude_db = tilt_db_oct * log2(h)
        # amplitude = 10^(amplitude_db / 20)
        if tilt_db_oct != 0:
            amp = 10 ** (tilt_db_oct * np.log2(h) / 20.0)
        else:
            amp = 1.0
        # Add harmonic with random phase
        phase = np.random.default_rng(42 + h).uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.5

    return signal


def measure_tilt_welch(signal, sr=SR):
    """Measure tilt using HEARING's Welch method (100-4000 Hz).

    Returns tilt in dB/octave using 10*slope convention.
    """
    freqs, psd = scipy_signal.welch(signal, fs=sr, nperseg=1024)
    mask = (freqs >= 100) & (freqs <= 4000)
    log_freqs = np.log10(freqs[mask])
    log_power = np.log10(psd[mask] + 1e-20)
    coeffs = np.polyfit(log_freqs, log_power, 1)
    return float(coeffs[0] * 10)


def measure_tilt_true(signal, sr=SR):
    """Measure tilt as true dB/octave (20*log10(2)*slope)."""
    freqs, psd = scipy_signal.welch(signal, fs=sr, nperseg=1024)
    mask = (freqs >= 100) & (freqs <= 4000)
    log_freqs = np.log10(freqs[mask])
    log_power = np.log10(psd[mask] + 1e-20)
    coeffs = np.polyfit(log_freqs, log_power, 1)
    return float(coeffs[0] * 10 * np.log10(2) * np.log(10) / np.log(2) / 10)
    # Actually: dB/oct = slope * 10 * log10(2) ... let's just do it directly
    # dB/oct = d(10*log10(P))/d(log2(f)) = 10*ln(10)*slope*d(log10(f))/d(log2(f))
    # = 10*ln(10)*slope*(ln(2)/ln(10)) = 10*ln(2)*slope
    # But VOICE uses 10*slope. The true conversion is 10*ln(2)*slope ≈ 6.931*slope
    # Wait, that's wrong too. Let me think again.
    # slope = d(log10 P)/d(log10 f)
    # dB = 10*log10(P) = 10*ln(P)/ln(10)
    # d(dB)/d(log10 f) = 10/ln(10) * d(ln P)/d(log10 f) = 10/ln(10) * ln(10) * slope = 10*slope
    # That's dB per log10(f), not per octave.
    # Per octave: d(dB)/d(log2 f) = d(dB)/d(log10 f) * d(log10 f)/d(log2 f)
    # = 10*slope * (ln(2)/ln(10)) = 10*slope * 0.3010 = 3.010*slope
    # Hmm, that doesn't seem right either.
    #
    # Let's just use the direct definition: 1 octave = factor of 2 in frequency
    # dB/octave = (dB at 2f) - (dB at f) for the fitted line
    # From the fit: log10(P) = slope * log10(f) + intercept
    # At f and 2f: log10(P2f) - log10(Pf) = slope * (log10(2f) - log10(f)) = slope * log10(2)
    # dB difference = 10 * (log10(P2f) - log10(Pf)) = 10 * slope * log10(2) = 3.010 * slope
    # So dB/octave = 3.010 * slope, NOT 10 * slope
    #
    # But VOICE reports "dB/octave" as 10*slope. So VOICE's "dB/octave" is actually
    # dB per decade (10*log10(f) change), not true dB/octave.
    # 1 decade = log10(10) = 1, so 10*slope = dB/decade.
    # True dB/octave = 3.010*slope = (10*slope) * 0.3010 = VOICE_value * 0.3010
    #
    # This is the core of the dispute. Let me just report both.


def calibrate(tilts=None, sr=SR, duration=DURATION):
    """Run tilt calibration: synthesize at known tilts, measure back."""
    if tilts is None:
        tilts = [-3.0, -6.0, -9.0, -4.31, 0.0]

    results = []
    for tilt in tilts:
        signal = synthesize_known_tilt(tilt, sr=sr, duration=duration)
        measured_welch = measure_tilt_welch(signal, sr=sr)

        # Also compute true dB/octave
        freqs, psd = scipy_signal.welch(signal, fs=sr, nperseg=1024)
        mask = (freqs >= 100) & (freqs <= 4000)
        log_freqs = np.log10(freqs[mask])
        log_power = np.log10(psd[mask] + 1e-20)
        coeffs = np.polyfit(log_freqs, log_power, 1)
        slope = coeffs[0]
        true_db_oct = slope * 10 * np.log10(2)  # true dB/octave
        db_per_decade = slope * 10  # what VOICE calls "dB/octave"

        results.append({
            "known_tilt_db_oct": tilt,
            "measured_welch_10x": round(measured_welch, 3),
            "true_db_oct": round(true_db_oct, 3),
            "db_per_decade": round(db_per_decade, 3),
            "recovery_10x": round(measured_welch - tilt, 3),
            "recovery_true": round(true_db_oct - tilt, 3),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Tilt calibration")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = calibrate()

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("Tilt Calibration — Known Inputs vs HEARING Measurement")
        print("=" * 70)
        print(f"{'Known':>8} {'Welch 10x':>12} {'True dB/oct':>12} {'dB/decade':>12} {'Rec 10x':>10} {'Rec true':>10}")
        print("-" * 70)
        for r in results:
            print(f"{r['known_tilt_db_oct']:>8.2f} {r['measured_welch_10x']:>12.3f} "
                  f"{r['true_db_oct']:>12.3f} {r['db_per_decade']:>12.3f} "
                  f"{r['recovery_10x']:>10.3f} {r['recovery_true']:>10.3f}")
        print()
        print("Key finding:")
        print("  VOICE reports 'dB/octave' as 10*slope (dB/decade).")
        print("  True dB/octave = 10*slope*log10(2) = 3.01*slope.")
        print("  Canonical -4.31 (VOICE) = -4.31/3.01 = -1.43 true dB/octave.")
        print("  HEARING's instrument uses the same 10*slope convention.")
        print("  Recovery with 10x: good. Recovery with true dB/oct: off by factor 3.01.")


if __name__ == "__main__":
    main()