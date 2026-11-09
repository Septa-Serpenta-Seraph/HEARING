#!/usr/bin/env python3
"""
Serpentic [HEARING] — Perceptual distance between two audio renders.

Computes multiple distance metrics that approximate what a listener would
perceive: timbre (MFCC), formant space, brightness, breathiness, roughness,
and spectral slope. Each metric is documented with its psychoacoustic basis.

Usage:
    python perceptual_distance.py file_a.wav file_b.wav
    python perceptual_distance.py file_a.wav file_b.wav --json
"""

import argparse
import json
import os
import sys

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call


def mfcc_distance(path_a, path_b, n_mfcc=13, sr=16000):
    """Mel-frequency cepstral coefficient cosine distance.

    MFCCs capture timbre — the "color" of a voice that distinguishes
    one speaker from another. Cosine distance on mean MFCC vectors
    gives a single number for timbre similarity.

    Basis: Davis & Mermelstein (1980), standard in speaker recognition.
    """
    y_a, _ = librosa.load(path_a, sr=sr)
    y_b, _ = librosa.load(path_b, sr=sr)
    mfcc_a = librosa.feature.mfcc(y=y_a, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    mfcc_b = librosa.feature.mfcc(y=y_b, sr=sr, n_mfcc=n_mfcc).mean(axis=1)
    # Cosine distance
    cos_sim = np.dot(mfcc_a, mfcc_b) / (np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_b) + 1e-10)
    return float(1.0 - cos_sim)


def formant_distance(path_a, path_b):
    """Euclidean distance in (F1, F2, F3, F4) space.

    Formants are the resonant frequencies of the vocal tract. The
    distance between two formant vectors approximates how different
    two vowels sound. Measured with Praat Burg (same as VOICE).

    Basis: Peterson & Barney (1952), vowel space theory.
    """
    def get_formants(path):
        snd = parselmouth.Sound(path)
        formant = call(snd, "To Formant (burg)", 0.01, 5, 5500, 0.025, 50)
        n_frames = call(formant, "Get number of frames")
        vals = {1: [], 2: [], 3: [], 4: []}
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            for f in range(1, 5):
                v = call(formant, "Get value at time", f, t, "Hertz", "Linear")
                if not np.isnan(v):
                    vals[f].append(v)
        return np.array([np.median(vals[f]) for f in range(1, 5)])

    fa = get_formants(path_a)
    fb = get_formants(path_b)
    return float(np.linalg.norm(fa - fb)), fa, fb


def brightness_distance(path_a, path_b, sr=16000):
    """Spectral centroid difference.

    The spectral centroid is the "center of mass" of the spectrum.
    Higher centroid = brighter sound. Difference approximates
    perceived brightness difference.

    Basis: Schubert et al. (2004), brightness as a perceptual dimension.
    """
    y_a, _ = librosa.load(path_a, sr=sr)
    y_b, _ = librosa.load(path_b, sr=sr)
    ca = librosa.feature.spectral_centroid(y=y_a, sr=sr).mean()
    cb = librosa.feature.spectral_centroid(y=y_b, sr=sr).mean()
    return float(abs(ca - cb)), float(ca), float(cb)


def tilt_distance(path_a, path_b, sr=16000):
    """Spectral tilt (slope) difference in dB/octave.

    Tilt captures the high-frequency vs low-frequency balance.
    Shallow tilt = bright; steep tilt = dark/warm.

    Basis: Sluijter & van Heuven (1996), spectral tilt as effort marker.
    """
    def compute_tilt(path, sr=16000):
        y, sr = librosa.load(path, sr=sr)
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        # Fit log-log slope
        log_freqs = np.log10(freqs[1:] + 1e-10)
        log_power = np.log10(S[1:].mean(axis=1) + 1e-10)
        # Linear fit
        coeffs = np.polyfit(log_freqs, log_power, 1)
        return coeffs[0] * 10  # dB/octave approx

    ta = compute_tilt(path_a, sr)
    tb = compute_tilt(path_b, sr)
    return float(abs(ta - tb)), float(ta), float(tb)


def hnr_distance(path_a, path_b):
    """HNR (harmonics-to-noise ratio) difference.

    HNR captures breathiness/roughness. Higher HNR = cleaner/harsher.
    Difference approximates perceived breathiness difference.

    Basis: Yumoto et al. (1982), HNR as voice quality measure.
    """
    def get_hnr(path):
        snd = parselmouth.Sound(path)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        return call(harmonicity, "Get mean", 0, 0)

    ha = get_hnr(path_a)
    hb = get_hnr(path_b)
    return float(abs(ha - hb)), float(ha), float(hb)


def roughness_estimate(path, sr=16000):
    """Estimate roughness (Terhardt 1974 model).

    Roughness is the perception of rapid amplitude modulation (15-300 Hz).
    Computed as the modulation depth in the 15-300 Hz range of the
    amplitude envelope.

    Basis: Terhardt (1974), "On the perception of periodic sound fluctuations."
    """
    y, sr = librosa.load(path, sr=sr)
    # Amplitude envelope via Hilbert transform
    envelope = np.abs(librosa.effects.hpss(y)[0] if False else np.abs(librosa.util.peak_pick(np.abs(y), pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0, wait=1)))
    # Simpler: use RMS envelope
    frame_length = 256
    hop_length = 64
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    # Modulation spectrum
    mod_spec = np.abs(np.fft.rfft(rms - rms.mean()))
    mod_freqs = np.fft.rfftfreq(len(rms), d=hop_length / sr)
    # Roughness band: 15-300 Hz modulation
    mask = (mod_freqs >= 15) & (mod_freqs <= 300)
    if mask.sum() == 0:
        return 0.0
    roughness = float(np.mean(mod_spec[mask]))
    return roughness


def perceptual_distance(path_a, path_b):
    """Compute all perceptual distance metrics between two renders.

    Returns a dict with individual metrics and a composite score.
    """
    mfcc_d = mfcc_distance(path_a, path_b)
    form_d, fa, fb = formant_distance(path_a, path_b)
    bright_d, ca, cb = brightness_distance(path_a, path_b)
    tilt_d, ta, tb = tilt_distance(path_a, path_b)
    hnr_d, ha, hb = hnr_distance(path_a, path_b)
    rough_a = roughness_estimate(path_a)
    rough_b = roughness_estimate(path_b)

    # Composite: weighted sum of normalized distances
    # Weights reflect perceptual importance for voice identity
    composite = (
        0.30 * min(mfcc_d, 1.0) +
        0.25 * min(form_d / 500.0, 1.0) +
        0.15 * min(bright_d / 500.0, 1.0) +
        0.15 * min(tilt_d / 5.0, 1.0) +
        0.10 * min(hnr_d / 10.0, 1.0) +
        0.05 * min(abs(rough_a - rough_b) / (max(rough_a, rough_b) + 1e-10), 1.0)
    )

    return {
        "mfcc_distance": round(mfcc_d, 4),
        "formant_distance_hz": round(form_d, 1),
        "formants_a": [round(x, 1) for x in fa],
        "formants_b": [round(x, 1) for x in fb],
        "brightness_distance_hz": round(bright_d, 1),
        "brightness_a_hz": round(ca, 1),
        "brightness_b_hz": round(cb, 1),
        "tilt_distance_db_oct": round(tilt_d, 3),
        "tilt_a_db_oct": round(ta, 3),
        "tilt_b_db_oct": round(tb, 3),
        "hnr_distance_db": round(hnr_d, 2),
        "hnr_a_db": round(ha, 2),
        "hnr_b_db": round(hb, 2),
        "roughness_a": round(rough_a, 4),
        "roughness_b": round(rough_b, 4),
        "composite_distance": round(composite, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Perceptual distance between two audio files")
    parser.add_argument("file_a", help="First audio file")
    parser.add_argument("file_b", help="Second audio file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    result = perceptual_distance(args.file_a, args.file_b)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Perceptual distance: {args.file_a} vs {args.file_b}")
        print(f"  MFCC (timbre):       {result['mfcc_distance']:.4f}")
        print(f"  Formant:             {result['formant_distance_hz']:.1f} Hz")
        print(f"    A: {result['formants_a']}")
        print(f"    B: {result['formants_b']}")
        print(f"  Brightness:          {result['brightness_distance_hz']:.1f} Hz")
        print(f"  Tilt:                {result['tilt_distance_db_oct']:.3f} dB/oct")
        print(f"  HNR:                 {result['hnr_distance_db']:.2f} dB")
        print(f"  Roughness:           A={result['roughness_a']:.4f} B={result['roughness_b']:.4f}")
        print(f"  Composite:           {result['composite_distance']:.4f}")


if __name__ == "__main__":
    main()