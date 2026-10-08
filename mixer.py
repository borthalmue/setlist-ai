"""
Does the actual audio mixing: renders a continuous set from a track order by
crossfading the tail of each track into the head of the next one. Crossfade
length is tied to BPM (a fixed number of bars, not a fixed number of
seconds), the same way a real DJ thinks about a transition.
"""

import librosa
import numpy as np
import soundfile as sf

SR = 44100


def bars_to_seconds(n_bars, bpm, beats_per_bar=4):
    return n_bars * beats_per_bar * 60.0 / bpm


def equal_power_fade(n_samples):
    t = np.linspace(0, np.pi / 2, n_samples)
    fade_out = np.cos(t)
    fade_in = np.sin(t)
    return fade_in, fade_out


def load_for_mix(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    return y  # shape (2, n_samples)


def crossfade_pair(y_a, y_b, bpm_a, bars=8, sr=SR):
    """Overlaps the tail of track A with the head of track B. Returns the
    combined audio and the sample index where B's independent portion
    starts, so callers can keep chaining transitions."""
    fade_len = int(bars_to_seconds(bars, bpm_a) * sr)
    fade_len = min(fade_len, y_a.shape[1], y_b.shape[1])

    fade_in, fade_out = equal_power_fade(fade_len)

    a_tail = y_a[:, -fade_len:] * fade_out
    b_head = y_b[:, :fade_len] * fade_in
    overlap = a_tail + b_head

    combined = np.concatenate([y_a[:, :-fade_len], overlap, y_b[:, fade_len:]], axis=1)
    return combined


def render_set(tracks_in_order, out_path, crossfade_bars=8, sr=SR):
    """tracks_in_order: list of analyzed-track dicts (needs 'path' and
    'bpm'), already in the order you want them mixed."""
    if not tracks_in_order:
        raise ValueError("need at least one track")

    mix = load_for_mix(tracks_in_order[0]["path"], sr=sr)

    for prev, nxt in zip(tracks_in_order, tracks_in_order[1:]):
        y_b = load_for_mix(nxt["path"], sr=sr)
        mix = crossfade_pair(mix, y_b, prev["bpm"], bars=crossfade_bars, sr=sr)

    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix / peak * 0.95

    sf.write(out_path, mix.T, sr)
    return out_path
