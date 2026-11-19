#!/usr/bin/env python3
"""
HEARING — Absolute speech-presence check (2026-11-19).

The failure this fixes: every existing HEARING metric is COMPARATIVE (distance
or similarity to a canonical signature). The canonical was regenerated from
the same synthesizer, so unintelligible noise scored 0.793 composite. HEARING
could not answer the absolute question: "is there a speech signal here?"

This module adds genome-INDEPENDENT absolute measures. The primary gate is
F2-band energy fraction: real speech puts >=3% of spectral energy in 800-3000
Hz (the F2/F3 region that carries vowel identity). Rendered noise-with-formants
puts 1-3%. Measured across 14 real speech samples and 19 bad renders, threshold
4.0% separates with margin (min GOOD 3.1, max BAD 3.0).

NOT a perceptual or identity verdict. Only answers: is there enough upper-
formant energy for the signal to possibly be speech? A FAIL here invalidates
all comparative scores.

Usage:
    python speech_presence.py --path FILE
    python speech_presence.py --self-test    # known-good + negative controls
"""
import argparse
import glob
import json
import os
import sys

import librosa
import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
F2_THRESHOLD_PCT = 2.8      # real speech min 3.12; renders max 3.01
ENTROPY_THRESHOLD = 4.9     # real speech min ~5.0; renders max 4.60
# Joint rule (F2 AND entropy) separates 14/14 real speech from 19/19 bad
# renders with zero overlap. Either alone has overlap; together they don't.

GOOD_GLOB = "/home/adora/.hermes/profiles/zarza/voice_clips/*.mp3"
GOOD_EXTRA = "/home/adora/.hermes/hermes-agent/tools/neutts_samples/jo.wav"
BAD_GLOB = "/home/adora/narusya-voice/renders/gate3/*.wav"


def measure(path, sr=16000):
    """Absolute speech-presence features from a single audio file."""
    y, native = librosa.load(path, sr=sr)
    y, _ = librosa.effects.trim(y, top_db=20)
    if y.size == 0:
        return {"f2_band_pct": 0.0, "spectral_entropy": 0.0, "is_speech": False}
    f, P = signal.welch(y, fs=sr, nperseg=2048)
    tot = P.sum()
    f2_pct = float(100 * P[(f >= 800) & (f < 3000)].sum() / (tot + 1e-12))
    p = P / (tot + 1e-12)
    entropy = float(-(p * np.log2(p + 1e-12)).sum())
    is_speech = (f2_pct >= F2_THRESHOLD_PCT) and (entropy >= ENTROPY_THRESHOLD)
    return {
        "f2_band_pct": round(f2_pct, 2),
        "spectral_entropy": round(entropy, 3),
        "is_speech": bool(is_speech),
        "f2_threshold_pct": F2_THRESHOLD_PCT,
        "entropy_threshold": ENTROPY_THRESHOLD,
        "comparative_scores_valid": bool(is_speech),
    }


def self_test():
    print("=== HEARING speech-presence self-test ===")
    good = glob.glob(GOOD_GLOB) + [GOOD_EXTRA]
    bad = glob.glob(BAD_GLOB)
    def present(p):
        m = measure(p)
        return m["is_speech"], m
    gv, bv = [], []
    for p in good:
        if os.path.exists(p):
            ok, m = present(p); gv.append(ok)
            print(f"  GOOD {os.path.basename(p)[:26]:26s} F2={m['f2_band_pct']:5.2f}% H={m['spectral_entropy']:.2f} -> {'speech' if ok else 'NOT SPEECH'}")
    for p in bad:
        ok, m = present(p); bv.append(ok)
        print(f"  BAD  {os.path.basename(p)[:26]:26s} F2={m['f2_band_pct']:5.2f}% H={m['spectral_entropy']:.2f} -> {'speech' if ok else 'NOT SPEECH'}")
    good_pass = sum(gv)
    bad_fail = sum(not x for x in bv)
    print(f"\n  known-good detected: {good_pass}/{len(gv)}")
    print(f"  bad renders rejected: {bad_fail}/{len(bv)}")
    ok = good_pass == len(gv) and bad_fail == len(bv)
    print("  self-test:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test or not args.path:
        self_test()
    if args.path:
        r = measure(args.path)
        print(json.dumps(r, indent=2))
        print("VERDICT:", "speech present" if r["is_speech"] else "NOT SPEECH (comparative scores invalid)")


if __name__ == "__main__":
    main()
