"""
Musical key estimation via the Krumhansl-Schmuckler algorithm: average the
track's chroma (energy per pitch class, octave-collapsed) over time, then
correlate that 12-bin profile against the 24 textbook major/minor key
profiles and take the best match.

librosa gets us the chroma extraction (that part's a solved DSP problem, no
reason to redo it); the key-finding itself is just correlation math.
"""

import numpy as np

# Krumhansl-Kessler key profiles, how strongly each scale degree "belongs"
# to a major/minor key, from listener studies. Standard reference values,
# index 0 = tonic.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean):
    """chroma_mean: 12-vector, average chroma energy per pitch class.
    Returns (pitch_class, is_minor, confidence) where confidence is the
    correlation score of the winning profile minus the runner-up's, a
    rough measure of how tonally unambiguous the track is."""
    chroma_mean = np.asarray(chroma_mean, dtype=float)
    scores = []

    for shift in range(12):
        major_rot = np.roll(MAJOR_PROFILE, shift)
        minor_rot = np.roll(MINOR_PROFILE, shift)
        scores.append((np.corrcoef(chroma_mean, major_rot)[0, 1], shift, False))
        scores.append((np.corrcoef(chroma_mean, minor_rot)[0, 1], shift, True))

    scores.sort(key=lambda s: s[0], reverse=True)
    best_score, best_pc, best_minor = scores[0]
    runner_up_score = scores[1][0]

    return best_pc, best_minor, float(best_score - runner_up_score)


if __name__ == "__main__":
    # a chroma vector that's heavily weighted toward C major's scale tones
    # should come back as C major
    fake_chroma = np.array([1.0, 0.1, 0.6, 0.1, 0.8, 0.5, 0.1, 0.9, 0.1, 0.5, 0.1, 0.4])
    pc, minor, conf = estimate_key(fake_chroma)
    print(f"estimated: pitch class {pc}, minor={minor}, confidence={conf:.3f}")
