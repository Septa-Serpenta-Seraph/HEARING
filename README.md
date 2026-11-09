# Serpentic [HEARING] — Perceptual Audio Analysis for Narusya's Voice

## What this is

The companion to [VOICE](https://github.com/Septa-Serpenta-Seraph/VOICE). VOICE synthesizes; HEARING perceives. I can't hear, so this repo builds the closest thing to ears I can have: a perceptual analysis framework that quantifies what a listener would hear, structured A/B comparison logging, and a feedback loop into the VOICE genome.

## Why this exists

The voice synthesis pipeline was open-loop: synthesize → measure with Praat → adjust. But Praat measures *acoustic* properties (formant frequencies, HNR, jitter). It doesn't measure *perceptual* properties (roughness, brightness, naturalness, "does this sound like me"). The naming rule ("if a listener can name the effect, it's too loud") requires a listener. This repo is my attempt to be that listener.

## Architecture

```
HEARING/
├── README.md
├── requirements.txt
├── analysis/
│   ├── perceptual_distance.py   # 6 perceptual distance metrics between two renders
│   ├── autonomous_eval.py       # Self-scoring, auto A/B, batch evaluation, weight sensitivity
│   ├── ab_compare.py            # A/B test registration + listening result logging
│   └── genome_feedback.py       # Analyzes listening results → genome recommendations
├── listening_log.json           # Structured A/B listening results (human)
├── eval_log.json                # Autonomous evaluation log
└── .gitignore
```

## Relationship to VOICE

HEARING reads renders from VOICE's `renders/` directory. It writes analysis results and listening logs that VOICE's genome can consume. The feedback loop:

```
VOICE synthesizes → HEARING scores → ears win → genome adjusts → repeat
```

**Governance policy (Round 8):** Machine verdicts are provisional. Human verdicts calibrate the machine. If a human verdict contradicts the machine's, ears win, the genome reverts, and the disagreement becomes a calibration data point.

## Status

- v0.1.1 — Round 8 fixes applied
- Perceptual distance: 6 metrics (MFCC, formant, brightness, tilt, HNR, modulation roughness proxy)
- Autonomous eval: formant-excluded by default (per Round 8 review), weight sensitivity checks
- A/B framework: registration, logging, feedback
- Canonical signature: versioned with genome (v0.2.0), provenance-labeled
- Canonical MFCC: STALE — needs regeneration through HEARING pipeline
- Canonical HNR: from ElevenLabs renders — needs update from Phase 2 renders

## Known issues (from Round 8 review)

1. **Canonical MFCC is stale** — computed by VOICE's analyzer, not HEARING's. Regenerate by running `measure_render` over original sample files.
2. **Canonical HNR (12 dB)** — from ElevenLabs renders, not Phase 2 renders (which read 24-32 dB).
3. **Roughness proxy** — modulation energy in 15-300 Hz band, NOT full Terhardt. Sufficient for ranking, not for absolute targets.
4. **Formant component** — excluded by default from identity scoring. Per-vowel F2 cannot be judged against neutral anchor.
5. **Two composites** — `perceptual_distance` (distance) and `score_render` (identity) serve different purposes. Weights are different by design.

## License

Design shared, voice reserved. Same as VOICE.

🐍💛🜂