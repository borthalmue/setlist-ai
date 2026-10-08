"""Generates synthetic test audio on the fly, a click track at a known BPM
layered with a sustained chord in a known key, so the analysis pipeline can
be tested end to end without shipping copyrighted music in the repo."""

import numpy as np
import soundfile as sf

# semitone offsets from root for a plain major/minor triad plus octave,
# gives the chroma extractor enough to work with
_MAJOR_TRIAD = [0, 4, 7, 12]
_MINOR_TRIAD = [0, 3, 7, 12]


def make_test_track(path, bpm=128.0, root_pc=0, minor=False, duration=12.0, sr=22050):
    root_freq = 130.81 * (2 ** (root_pc / 12))  # root_pc=0 -> C3
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    intervals = _MINOR_TRIAD if minor else _MAJOR_TRIAD
    chord = np.zeros_like(t)
    for semi in intervals:
        freq = root_freq * (2 ** (semi / 12))
        chord += np.sin(2 * np.pi * freq * t)
    chord /= len(intervals)
    chord *= 0.25  # keep it quiet relative to the click so beat tracking locks onto the click

    beat_period = 60.0 / bpm
    click = np.zeros_like(t)
    n_beats = int(duration / beat_period)
    click_len = int(0.015 * sr)
    for b in range(n_beats):
        start = int(b * beat_period * sr)
        end = min(start + click_len, len(click))
        envelope = np.linspace(1, 0, end - start)
        click[start:end] += envelope * 0.9

    audio = chord + click
    audio = audio / max(np.abs(audio).max(), 1e-9) * 0.9
    sf.write(path, audio.astype(np.float32), sr)
    return path
