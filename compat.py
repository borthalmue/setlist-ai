"""Scores how clean a transition between two tracks would be. Lower is
better. 0 would be an identical track."""

from camelot import camelot_distance


def bpm_cost(bpm_a, bpm_b, tolerance=0.06):
    """DJs pitch-shift a track a few percent to match tempo, and half/double
    time tracks mix fine too (a 174 bpm DnB track sits right on top of an
    87 bpm hip-hop track), so this checks the target against bpm_b itself
    *and* half/double, and takes whichever is closest."""
    candidates = [bpm_b, bpm_b * 2, bpm_b / 2]
    best_relative_gap = min(abs(bpm_a - c) / bpm_a for c in candidates)
    return min(best_relative_gap / tolerance, 2.0)


def transition_cost(track_a, track_b, w_key=1.0, w_bpm=1.0, w_energy=0.3):
    key_c = camelot_distance(track_a["camelot"], track_b["camelot"])
    bpm_c = bpm_cost(track_a["bpm"], track_b["bpm"])
    energy_c = abs(track_a.get("energy_norm", 0) - track_b.get("energy_norm", 0))
    return w_key * key_c + w_bpm * bpm_c + w_energy * energy_c


def cost_matrix(tracks, **weights):
    n = len(tracks)
    return [[transition_cost(tracks[i], tracks[j], **weights) if i != j else 0.0
             for j in range(n)] for i in range(n)]
