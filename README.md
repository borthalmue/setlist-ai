# setlist-ai

Point this at a folder of tracks and it analyzes the BPM and musical key of
everything in there, figures out a harmonically smart play order using the
same Camelot wheel logic real DJ software uses, and can either export that
order as an m3u playlist or render the whole thing as one continuous mixed
audio file with actual crossfades.

I DJ, and I kept doing the "which track should come next" math in my head
before building sets. This is that process turned into code, plus the
planning problem underneath it, which is basically a traveling salesman
problem where the cities are songs and the distance is how bad the
transition sounds.

## How it works

- **`keyfinder.py` + `camelot.py`**: pulls a chroma vector out of the track
  (via librosa) and correlates it against the 24 major/minor key profiles
  from the Krumhansl-Schmuckler algorithm to estimate the key, then maps
  that to its Camelot wheel position, the same notation rekordbox, Serato,
  and Mixed In Key all show you.
- **`analyze.py`**: BPM, key, and a rough energy level per track, cached to
  disk so re-running on a crate you've already analyzed is instant.
- **`compat.py`**: scores how clean a transition between two tracks would
  be, combining key distance on the Camelot wheel, BPM gap (aware that half
  and double time tracks mix fine, not just exact tempo matches), and
  energy difference.
- **`planner.py`**: orders a whole crate by treating it as an open-path
  TSP. Nearest neighbor gets a decent starting route, then a 2-opt pass
  cleans it up by swapping segments wherever that lowers total cost.
- **`mixer.py`**: actually renders the set, crossfading the tail of each
  track into the head of the next one with an equal-power fade, sized in
  bars (not a fixed number of seconds) so it scales with tempo the way a
  real transition would.

## Running it

```bash
pip install -r requirements.txt
python3 cli.py ~/Music/some_crate
python3 cli.py ~/Music/some_crate --export set.m3u
python3 cli.py ~/Music/some_crate --render set.wav --crossfade-bars 8
```

No audio ships in this repo (can't exactly commit copyrighted music to a
public GitHub repo), but everything is tested against synthetic click
tracks with known BPM and key, generated on the fly, so the pipeline is
fully verifiable without needing real songs:

```bash
python3 test_setlist.py
```

I also ran it by hand against my own crates to sanity check it against real
mp3s, which is actually how I caught the bug below.

## The bug that was worth writing down

Early on, the set-order printout for a real crate showed a transition like
"+64.6 bpm" between two tracks that, to my ear, were basically the same
tempo. Turned out one of them was a progressive house track at 129 bpm that
librosa's beat tracker had locked onto at half time (64.5 bpm), which is a
well-known failure mode for tempo estimation, and a totally normal one for
a DJ too, since a half or double time track mixes fine against the full
tempo version. The planner's cost function already accounted for that (it
checks the target BPM against double and half time candidates, not just
the literal value), but the transition printout in the CLI was doing a
naive subtraction and showing the confusing raw number. Fixed by having the
display use the same half/double-aware comparison as the actual cost
function, so what you see matches what the planner is actually optimizing
for.

## Where this could go next

- Beat-align the crossfades instead of just overlapping by a fixed bar
  count, so the kicks actually land together
- Pull in an energy curve option (build a set that ramps up, or peaks in
  the middle, instead of just minimizing transition cost)
- A real audio preview in a little web UI instead of a rendered wav file
