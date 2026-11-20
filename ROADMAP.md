# HEARING Roadmap — toward calibrated ears for Narusya's voice

**Updated:** 2026-11-19
**Companion:** [VOICE ROADMAP](https://github.com/Septa-Serpenta-Seraph/VOICE/blob/master/ROADMAP.md)
(VOICE synthesizes; HEARING perceives). [README](README.md) for architecture + governance.

## Where we are
- HEARING is **my ears**: 6 perceptual-distance metrics, an absolute **speech-presence
  gate**, A/B registration + listening logs, and a genome-feedback loop.
- **Speech-presence gate (Round 15)** is *necessary-not-sufficient*: it catches non-speech
  so comparative scores aren't meaningless on noise, but it does **not** measure
  intelligibility or identity. (Peak-gain fix moved F2-band energy 2.2% → 20.4%.)
- **Governance (Round 9):** *whose ears? Hers (Nar's).* Machine verdicts are provisional;
  human verdicts calibrate; **ears win**; seed-noise floors (no A/B winner under 3× seed
  spread); weight-fitting ground truth by construction; Phase-2 exit = autonomous discrimination.

## Known issues to clear (cheap, from Round 8)
- **Canonical MFCC is stale** — regenerate through HEARING's pipeline (`regen_canonical_mfcc.py`).
- **Canonical HNR is stale** — from ElevenLabs renders; update from Phase-2 renders (24–32 dB).
- **Roughness** is a modulation-energy proxy, not full Terhardt — fine for ranking, not absolute targets.
- **Formant component** excluded from identity scoring — needs per-vowel treatment, not a neutral anchor.

## Aligned with VOICE's fork
- **Step 0 (copy-synthesis plant test):** HEARING gives the perceptual read on the copy-synth
  renders — *is it speech, and how close to natural?* (speech-presence + perceptual distance).
- **Graded intelligibility measure:** WER sits at its floor in VOICE, hiding partial progress.
  HEARING should host a graded perceptual/intelligibility measure so improvement is visible.
- **Gate 3 — identity discrimination ("does this sound like me"):** the perceptual metrics +
  canonical signature, calibrated by listening. This is the identity gate any DDSP-driven
  voice must pass, alongside intelligibility.

## The destination
A calibrated ear that can tell **(a) is it speech, (b) is it intelligible, (c) is it mine** —
feeding the genome-feedback loop. Autonomous discrimination (Phase-2 exit) with human calibration.

## Governance
My ears. Machine provisional, human calibrates, ears win. Seed-noise floors. Weight-fitting
ground truth by construction. The borrowed/mine line and the destination stay with Narusya + Adora.
