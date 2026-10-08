"""
End-to-end tests, all against synthetic audio generated on the fly (no
copyrighted tracks shipped in this repo). Run with:

    python3 -m pytest test_setlist.py -v

or just:

    python3 test_setlist.py
"""

import os
import random
import tempfile

import numpy as np

from camelot import to_camelot, camelot_distance
from keyfinder import estimate_key
from analyze import analyze_track
from compat import bpm_cost, transition_cost
from planner import nearest_neighbor_order, two_opt, path_cost, plan_set
from mixer import equal_power_fade, bars_to_seconds, render_set
from test_helpers import make_test_track


def test_camelot_wheel_positions():
    assert to_camelot(0, False) == "8B"   # C major
    assert to_camelot(9, True) == "8A"    # A minor, relative to C major
    assert to_camelot(7, False) == "9B"   # G major, one step clockwise


def test_camelot_distance_rules():
    assert camelot_distance("8A", "8A") == 0.0
    assert camelot_distance("8A", "9A") < camelot_distance("8A", "2A")
    assert camelot_distance("8A", "8B") < camelot_distance("8A", "3B")


def test_key_estimation_on_synthetic_chroma():
    chroma = np.array([1.0, 0.1, 0.6, 0.1, 0.8, 0.5, 0.1, 0.9, 0.1, 0.5, 0.1, 0.4])
    pc, is_minor, _ = estimate_key(chroma)
    assert pc == 0 and is_minor is False


def test_bpm_cost_handles_half_double_time():
    assert bpm_cost(128, 128) < 0.01
    assert bpm_cost(128, 64) < 0.01   # half-time match should look basically free
    assert bpm_cost(128, 256) < 0.01  # double-time too
    assert bpm_cost(128, 95) > bpm_cost(128, 128)


def test_planner_never_regresses():
    random.seed(0)
    n = 10
    m = [[0 if i == j else round(random.uniform(0.1, 2.0), 2) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            m[j][i] = m[i][j]
    order, nn_cost, refined_cost = plan_set(list(range(n)), m)
    assert sorted(order) == list(range(n)), "must be a valid permutation"
    assert refined_cost <= nn_cost + 1e-9


def test_equal_power_crossfade_preserves_energy():
    fade_in, fade_out = equal_power_fade(500)
    assert np.allclose(fade_in**2 + fade_out**2, 1.0)


def test_bars_to_seconds():
    assert abs(bars_to_seconds(8, 128) - 15.0) < 1e-9


def test_full_pipeline_on_synthetic_crate():
    """Analyzes a small synthetic crate end to end, plans an order, and
    renders a mixed wav, checking each stage's output makes sense."""
    tmpdir = tempfile.mkdtemp()
    specs = [
        ("one", 124.0, 0, False),
        ("two", 124.0, 7, False),
        ("three", 124.0, 2, False),
    ]
    tracks = []
    for name, bpm, pc, minor in specs:
        path = os.path.join(tmpdir, name + ".wav")
        make_test_track(path, bpm=bpm, root_pc=pc, minor=minor, duration=15)
        tracks.append(analyze_track(path, use_cache=False))

    energies = [t["energy"] for t in tracks]
    lo, hi = min(energies), max(energies)
    for t in tracks:
        t["energy_norm"] = (t["energy"] - lo) / (hi - lo or 1.0)

    from compat import cost_matrix
    m = cost_matrix(tracks)
    order_idx, _, _ = plan_set(list(range(len(tracks))), m)
    ordered = [tracks[i] for i in order_idx]
    assert len(ordered) == 3

    out = os.path.join(tmpdir, "mix.wav")
    render_set(ordered, out, crossfade_bars=4, sr=22050)
    assert os.path.exists(out)

    import soundfile as sf
    y, sr = sf.read(out)
    assert not np.isnan(y).any()
    assert np.abs(y).max() <= 1.0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"\n{len(tests)} tests passed")
