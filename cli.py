"""
Point this at a folder of tracks and it analyzes everything, figures out a
harmonically and tempo-smart play order, and can either hand you an m3u
playlist to drop into your actual DJ software, or just render the whole
thing as one continuous mixed wav file.

    python3 cli.py ~/Music/some_crate
    python3 cli.py ~/Music/some_crate --export set.m3u
    python3 cli.py ~/Music/some_crate --render set.wav
"""

import argparse
import os

from analyze import analyze_folder
from compat import cost_matrix, transition_cost
from planner import plan_set
from camelot import camelot_distance


def effective_bpm_gap(bpm_a, bpm_b):
    """Beat trackers octave-error constantly (a 129 bpm track reads as 64.5),
    and half/double time tracks genuinely do mix fine, so report whichever
    comparison (straight, half, or double) is actually closest instead of
    the literal difference, which is often meaningless on its own."""
    candidates = [bpm_b, bpm_b * 2, bpm_b / 2]
    return min(candidates, key=lambda c: abs(bpm_a - c)) - bpm_a


def describe_transition(a, b):
    key_gap = camelot_distance(a["camelot"], b["camelot"])
    bpm_gap = effective_bpm_gap(a["bpm"], b["bpm"])
    if key_gap == 0.0:
        key_note = "same key"
    elif key_gap <= 0.2:
        key_note = f"{a['camelot']} -> {b['camelot']}, clean harmonic move"
    elif key_gap <= 0.3:
        key_note = f"{a['camelot']} -> {b['camelot']}, relative major/minor swap"
    else:
        key_note = f"{a['camelot']} -> {b['camelot']}, rougher key jump"
    return f"{key_note}, {bpm_gap:+.1f} bpm"


def write_m3u(tracks_in_order, out_path):
    with open(out_path, "w") as f:
        f.write("#EXTM3U\n")
        for t in tracks_in_order:
            f.write(f"#EXTINF:-1,{t['title']}\n{t['path']}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="folder of audio files to plan a set from")
    parser.add_argument("--export", help="write the suggested order as an m3u playlist")
    parser.add_argument("--render", help="render the suggested order as one continuous mixed wav")
    parser.add_argument("--crossfade-bars", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true", help="re-analyze even if cached")
    args = parser.parse_args()

    print(f"analyzing {args.folder} ...")
    tracks = analyze_folder(args.folder, use_cache=not args.no_cache)
    if len(tracks) < 2:
        print("need at least 2 tracks to plan a set")
        return

    print(f"\n{'title':40s} {'bpm':>6s} {'key':>5s}  energy")
    for t in tracks:
        print(f"{t['title'][:40]:40s} {t['bpm']:6.1f} {t['camelot']:>5s}  {t['energy_norm']:.2f}")

    m = cost_matrix(tracks)
    order_idx, nn_cost, final_cost = plan_set(list(range(len(tracks))), m)
    ordered = [tracks[i] for i in order_idx]

    print(f"\nsuggested set order (cost {nn_cost:.2f} -> {final_cost:.2f} after 2-opt):\n")
    print(f"1. {ordered[0]['title']}")
    for i in range(len(ordered) - 1):
        note = describe_transition(ordered[i], ordered[i + 1])
        print(f"      {note}")
        print(f"{i + 2}. {ordered[i + 1]['title']}")

    if args.export:
        write_m3u(ordered, args.export)
        print(f"\nwrote playlist to {args.export}")

    if args.render:
        from mixer import render_set
        print(f"\nrendering mixed set to {args.render} ...")
        render_set(ordered, args.render, crossfade_bars=args.crossfade_bars)
        print("done")


if __name__ == "__main__":
    main()
